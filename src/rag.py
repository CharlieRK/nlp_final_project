import json
import re
from typing import Dict, List, Tuple

import faiss
import requests
from sentence_transformers import SentenceTransformer

from src.config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    CHUNKS_FILE,
    EMBED_MODEL_NAME,
    GENERATOR_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    TOP_K,
    VECTOR_DIR,
)


def load_chunks() -> List[Dict]:
    chunks = []
    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


class RagEngine:
    def __init__(self) -> None:
        index_path = VECTOR_DIR / "index.faiss"
        if not index_path.exists() or not CHUNKS_FILE.exists():
            raise FileNotFoundError(
                "Missing vector index. Please run: python src/ingest.py"
            )
        self.index = faiss.read_index(str(index_path))
        self.chunks = load_chunks()
        self.model = SentenceTransformer(EMBED_MODEL_NAME)

    def _extract_paper_hints(self, question: str) -> List[str]:
        q = question.lower().strip()
        hints = set()

        # arXiv-like ids, e.g. 1706.03762v7 or 2401.06104v2
        for m in re.findall(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", q):
            hints.add(m)

        # quoted title fragments: "attention is all you need"
        for m in re.findall(r'"([^"]+)"', q):
            m = m.strip().lower()
            if len(m) >= 4:
                hints.add(m)

        # lightweight "paper XYZ" capture
        m = re.search(r"\bpaper\s+([a-z0-9\.\- ]{4,80})", q)
        if m:
            hints.add(m.group(1).strip())

        return list(hints)

    def _matches_hints(self, chunk: Dict, hints: List[str]) -> bool:
        if not hints:
            return True
        haystack = f"{chunk.get('paper_title', '')} {chunk.get('source_file', '')}".lower()
        return any(h in haystack for h in hints)

    def _tokenize_query(self, text: str) -> List[str]:
        tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
        stop = {
            "paper",
            "what",
            "which",
            "does",
            "this",
            "that",
            "with",
            "from",
            "into",
            "about",
            "main",
            "objective",
        }
        return [t for t in tokens if t not in stop]

    def _question_intent(self, question: str) -> str:
        q = question.lower()
        if any(k in q for k in ["problem", "objective", "goal", "motivation", "solve"]):
            return "problem"
        if any(k in q for k in ["method", "approach", "model", "architecture"]):
            return "method"
        if any(k in q for k in ["result", "performance", "evidence", "effective"]):
            return "result"
        return "general"

    def _chunk_quality_score(self, chunk: Dict, question: str) -> float:
        text = " ".join(chunk.get("text", "").lower().split())
        q_tokens = self._tokenize_query(question)
        overlap = sum(1 for t in q_tokens if t in text)
        intent = self._question_intent(question)

        # Prefer intro-like chunks and declarative cues.
        cue_bonus = 0
        for cue in ("we propose", "we present", "our method", "introduction", "problem"):
            if cue in text:
                cue_bonus += 1

        local_id = int(chunk.get("local_chunk_id", 9999))
        intro_bonus = 2 if local_id < 10 else 0
        if intent == "problem":
            if local_id < 20:
                intro_bonus += 4
            if any(k in text for k in ["problem", "challenge", "motivation", "difficult", "need"]):
                cue_bonus += 3
        elif intent == "method":
            if any(k in text for k in ["method", "approach", "model", "architecture", "we propose"]):
                cue_bonus += 3
        elif intent == "result":
            if any(k in text for k in ["result", "experiment", "evaluation", "improve", "outperform"]):
                cue_bonus += 3

        noise_penalty = 2 if "<pad>" in text else 0
        return float(overlap + cue_bonus + intro_bonus - noise_penalty)

    def retrieve(self, question: str, top_k: int = TOP_K) -> List[Dict]:
        q_emb = self.model.encode([question], convert_to_numpy=True).astype("float32")
        hints = self._extract_paper_hints(question)

        # If paper hints exist, over-retrieve and then filter to target paper(s).
        candidate_k = top_k if not hints else min(len(self.chunks), max(top_k * 60, 200))
        _, indices = self.index.search(q_emb, candidate_k)
        candidates = []
        for i in indices[0]:
            if 0 <= i < len(self.chunks):
                chunk = self.chunks[int(i)]
                if self._matches_hints(chunk, hints):
                    candidates.append(chunk)

        # Rerank candidates with lightweight lexical/quality heuristics.
        if candidates:
            candidates.sort(
                key=lambda c: self._chunk_quality_score(c, question),
                reverse=True,
            )
        results = candidates[:top_k]

        # Fallback: if hint filtering returns too little, keep best global hits.
        if len(results) < top_k:
            _, global_indices = self.index.search(q_emb, top_k)
            for i in global_indices[0]:
                if 0 <= i < len(self.chunks):
                    chunk = self.chunks[int(i)]
                    if chunk not in results:
                        results.append(chunk)
                if len(results) >= top_k:
                    break
        return results

    def _build_prompt(self, question: str, retrieved_chunks: List[Dict]) -> str:
        context_blocks = []
        for c in retrieved_chunks:
            context_blocks.append(
                f"[{c['paper_title']} | chunk {c['local_chunk_id']}]\n{c['text']}"
            )
        context = "\n\n".join(context_blocks)

        prompt = f"""
You are PaperPal, an assistant that answers questions about academic papers.
Use only the retrieved context. If context is insufficient, say you are not sure.
Keep the answer concise and student-friendly.

Question:
{question}

Retrieved context:
{context}
""".strip()
        return prompt

    def _generate_with_ollama(self, prompt: str) -> str:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        }
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=180
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

    def _generate_with_anthropic(self, prompt: str) -> str:
        if not ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Please export your Claude API key."
            )

        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": ANTHROPIC_MODEL,
            "max_tokens": 512,
            "temperature": 0.2,
            "messages": [{"role": "user", "content": prompt}],
        }
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=180,
        )
        if response.status_code >= 400:
            detail = response.text[:500]
            raise RuntimeError(
                f"Anthropic API error {response.status_code}: {detail}"
            )
        body = response.json()
        content_blocks = body.get("content", [])
        text_parts = [
            block.get("text", "")
            for block in content_blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(text_parts).strip()

    def _generate_with_openai(self, prompt: str) -> str:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set. Please export your OpenAI API key.")

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": OPENAI_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 512,
        }
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=180,
        )
        if response.status_code >= 400:
            detail = response.text[:500]
            raise RuntimeError(f"OpenAI API error {response.status_code}: {detail}")
        body = response.json()
        choices = body.get("choices", [])
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "").strip()

    def generate(self, question: str, retrieved_chunks: List[Dict]) -> str:
        prompt = self._build_prompt(question, retrieved_chunks)
        provider = GENERATOR_PROVIDER.lower().strip()

        if provider in {"local", "extractive"}:
            return self._generate_local_extractive(question, retrieved_chunks)
        if provider == "anthropic":
            return self._generate_with_anthropic(prompt)
        if provider == "openai":
            return self._generate_with_openai(prompt)
        if provider == "ollama":
            return self._generate_with_ollama(prompt)
        raise ValueError(
            f"Unsupported GENERATOR_PROVIDER='{GENERATOR_PROVIDER}'. Use 'local', 'anthropic', 'openai', or 'ollama'."
        )

    def _generate_local_extractive(self, question: str, retrieved_chunks: List[Dict]) -> str:
        if not retrieved_chunks:
            return "I could not find relevant context in the indexed papers."

        q_tokens = self._tokenize_query(question)
        intent = self._question_intent(question)
        candidate_sentences = []
        for c in retrieved_chunks[:3]:
            text = " ".join(
                c["text"].replace("<pad>", " ").replace("<EOS>", " ").split()
            )
            # Split lightly by sentence delimiters to extract one direct answer sentence.
            raw_sentences = re.split(r"(?<=[\.\?\!])\s+", text)
            for s in raw_sentences:
                s = s.strip()
                if len(s) < 40:
                    continue
                overlap = sum(1 for t in q_tokens if t in s.lower())
                bonus = 0
                s_lower = s.lower()
                if intent == "problem" and any(k in s_lower for k in ["problem", "challenge", "goal", "objective", "motivation", "difficult"]):
                    bonus += 3
                if intent == "method" and any(k in s_lower for k in ["method", "approach", "model", "architecture", "we propose"]):
                    bonus += 3
                if intent == "result" and any(k in s_lower for k in ["result", "experiment", "evaluation", "improve", "outperform"]):
                    bonus += 3
                penalty = 1 if "chunk" in s_lower else 0
                score = overlap + bonus - penalty
                candidate_sentences.append((score, s))

        if candidate_sentences:
            candidate_sentences.sort(key=lambda x: x[0], reverse=True)
            best = candidate_sentences[0][1]
        else:
            best = " ".join(retrieved_chunks[0]["text"].split())[:240].strip()

        return best

    def ask(self, question: str) -> Tuple[str, List[Dict]]:
        retrieved = self.retrieve(question, top_k=TOP_K)
        answer = self.generate(question, retrieved)
        return answer, retrieved
