# Prompt Engineering Notes

Random notes from reading about prompting LLMs well.

## Grounding
Always tell the model explicitly what it's allowed to use. "Answer
using only the following context" massively reduces hallucination
compared to just pasting context with no instruction.

## Few-shot vs zero-shot
Few-shot examples help a lot for structured output tasks (JSON,
specific formats) but cost extra tokens. For simple Q&A, zero-shot
with a clear system instruction is usually enough.

## Chunking strategy
Smaller chunks (300-800 chars) retrieve more precisely but can lose
surrounding context. Overlap between chunks (100-200 chars) helps
avoid cutting a sentence exactly at a chunk boundary.

## Citation trick
Asking the model to cite [source: filename] after each claim makes it
easier to verify answers and also seems to reduce made-up facts,
maybe because it has to "commit" to where the info came from.

#prompting #llm #rag
