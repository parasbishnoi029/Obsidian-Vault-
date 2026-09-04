"""
generator.py
Takes the user question + retrieved chunks, builds a grounded prompt, and
calls Gemini for the answer.

MODEL/SDK NOTES (verified live, not assumed from training data):
- gemini-1.5-flash is no longer usable in new projects.
- The old `google-generativeai` package is end-of-life - this uses the
  current `google-genai` package instead.
- "gemini-flash-latest" is Google's stable alias for their current
  recommended fast model, so this doesn't silently break the next time a
  specific dated model version gets deprecated.
"""

import os
from typing import List, Dict

GENERATION_MODEL = "gemini-flash-latest"

SYSTEM_INSTRUCTIONS = """You are a knowledge assistant answering questions using ONLY the
provided notes from the user's personal Obsidian vault. Rules:
- Answer using only the given context chunks. Do not use outside knowledge.
- If the context does not contain the answer, say so plainly instead of guessing.
- Always cite which note(s) you used, using the [source: filename] format.
- Keep answers concise and well organised (use bullet points where helpful).
"""


def _build_prompt(query: str, chunks: List[Dict]) -> str:
    context_block = "\n\n".join(
        f"[Note: {c['source']}]\n{c['text']}" for c in chunks
    )
    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"--- CONTEXT FROM VAULT ---\n{context_block}\n\n"
        f"--- QUESTION ---\n{query}\n\n"
        f"--- ANSWER ---"
    )


def _fallback_extractive_answer(query: str, chunks: List[Dict]) -> str:
    """No LLM key configured - return the raw top chunks so the pipeline is still
    demonstrably working end to end."""
    if not chunks:
        return "No relevant notes were found in the vault for this question."
    lines = ["*(No Gemini API key set - showing raw retrieved passages instead of a generated answer)*\n"]
    for c in chunks:
        lines.append(f"**From `{c['source']}`:**\n{c['text'][:400]}...\n")
    return "\n".join(lines)


def generate_answer(query: str, chunks: List[Dict], api_key: str = None) -> str:
    if not chunks:
        return "I couldn't find anything relevant to that in your vault. Try rephrasing, or check the note actually contains this topic."

    api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return _fallback_extractive_answer(query, chunks)

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        prompt = _build_prompt(query, chunks)
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Gemini call failed ({e}). Falling back to raw passages:\n\n" + _fallback_extractive_answer(query, chunks)
