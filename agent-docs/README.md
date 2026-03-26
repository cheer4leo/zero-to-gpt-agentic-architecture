# Agent-Readable Documentation

This folder contains self-contained documentation for every module in the project. Each file is designed so that **any AI agent** can read it and give you a personalized, accurate explanation.

## How to Use These Docs

1. Pick any module you want to understand
2. Copy the corresponding agent-doc (or point your agent at the file)
3. Tell your agent: "Read this document and explain [module] to me"
4. Ask follow-up questions — the agent has full context to answer accurately

The agent will tailor its explanation to YOUR level. You can ask it to:

- "Explain this like I'm a beginner who only knows Python basics"
- "Explain the math behind this with the linear algebra"
- "Compare this to how it works in the original Transformer paper"
- "Walk me through what happens to a specific input tensor step by step"
- "Explain this in [your language]"

## Why This Exists

When you feed raw source code to an AI agent, it can explain syntax but often misses the deeper *why*. It might hallucinate about design choices or give a shallow explanation.

Each agent-doc solves this by providing the agent with everything it needs upfront:

- **Where this fits** in the overall GPT pipeline
- **The core intuition** — the mental model that makes the concept click
- **The algorithm** described in plain language
- **Key design decisions** and their rationale
- **The complete source code** so the doc is self-contained
- **Common questions** with accurate answers
- **Suggested exercises** for deeper understanding

## Document Index

### Chapter 2: Working with Text Data

| File | Module | Concept |
|------|--------|---------|
| `ch02/01-word-embeddings.md` | — | Why LLMs need embeddings (conceptual) |
| `ch02/02-tokenizer.md` | `tokenizer.py` | Text tokenization and vocabulary |
| `ch02/03-special-tokens.md` | `tokenizer.py` | Special tokens (endoftext, unk, pad) |
| `ch02/04-bpe.md` | `tokenizer.py` | Byte Pair Encoding via tiktoken |
| `ch02/05-sliding-window.md` | `data.py` | Sliding window data sampling |
| `ch02/06-embeddings.md` | `embeddings.py` | Token + positional embeddings |

*More chapters will be added as the project progresses.*
