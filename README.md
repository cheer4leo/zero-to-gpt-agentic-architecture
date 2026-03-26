# Zero to GPT: The Agentic Architecture

Build a GPT from scratch using Agentic Engineering — not manual coding, not vibe coding.

## What Is This?

A YouTube series and open-source project that teaches two things simultaneously:

1. **How LLMs work** — by building a GPT from scratch, following the standard 3-stage pipeline (build → pretrain → fine-tune)
2. **How to do Agentic Engineering** — by using AI agents to implement every component with human oversight and understanding

Every module in this repo was built by directing AI agents with carefully crafted prompts, then reviewed, corrected, and approved by a human who understands the underlying concepts. The agent prompts, review notes, and agent-readable documentation are all checked into the repo — so you can see exactly how the code was built, not just the final result.

## Philosophy

Inspired by Andrej Karpathy's [microGPT](http://karpathy.github.io/2026/02/12/microgpt/) and his insight that in the AI era, the human expert's job is to *distill core intuitions* and let agents handle personalized explanation at scale.

- **The human distills; the agent explains.** Each module includes an agent-doc with the core intuition, complete source code, and everything an AI agent needs to give you a personalized walkthrough.
- **Write Markdown for agents, not HTML for humans.** The `agent-docs/` folder is designed so you can feed any file to your own AI and get an explanation at your level.
- **Show the spectrum.** Early episodes use single-agent workflows. Later episodes demonstrate parallel multi-agent pipelines. You see the full progression.

## How to Follow Along

**Watch**: [YouTube Playlist](#) (link coming soon)

**Learn at your own pace**: Pick any file in `agent-docs/`, feed it to your AI agent, and ask it to explain at your level.

**Code along**: Each episode corresponds to a Pull Request. Check the PR history to see exactly what changed.

## Project Structure

```
zero-to-gpt-agentic-architecture/
├── src/llm_from_scratch/    # The GPT implementation (production code)
├── tests/                   # Test suite
├── notebooks/               # Jupyter notebooks for each episode
├── agent-prompts/           # The prompts used to direct AI agents
├── agent-docs/              # Agent-readable docs (feed to your AI for explanations)
├── scripts/                 # Runnable training and generation scripts
└── docs/                    # Project documentation
```

## Episode Guide

### Stage 1: Building a GPT

| # | Episode | Topic |
|---|---------|-------|
| 1 | Project Kickoff | What is Agentic Engineering? |
| 2 | What Are Word Embeddings? | Why LLMs can't read text |
| 3 | Building a Text Tokenizer | From scratch, with an agent |
| 4 | Special Tokens | How LLMs handle unknowns and boundaries |
| 5 | Byte Pair Encoding | The tokenizer behind GPT |
| 6 | Sliding Window | How to create LLM training data |
| 7 | Embeddings & Position | The complete GPT input pipeline |
| 8–13 | Attention & Transformer | Self-attention → multi-head → full GPT |

### Stage 2: Foundation Model

| # | Episode | Topic |
|---|---------|-------|
| 14–18 | Pretraining | Training loop, evaluation, weight loading |

### Stage 3: Fine-tuning

| # | Episode | Topic |
|---|---------|-------|
| 19–25 | Fine-tuning | Classification and instruction following |

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/zero-to-gpt-agentic-architecture.git
cd zero-to-gpt-agentic-architecture
pip install -e ".[dev]"
make test
```

## Acknowledgments

This project was inspired by the pedagogical approach in Sebastian Raschka's
"Build a Large Language Model (From Scratch)" (Manning, 2024). The concepts
taught here — transformer architectures, attention mechanisms, pretraining,
and fine-tuning — are drawn from the broader machine learning research literature.
All code, documentation, examples, and explanations in this repository are
original work.

## License

MIT
