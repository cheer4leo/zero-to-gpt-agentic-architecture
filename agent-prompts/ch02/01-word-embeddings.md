# Agent Prompt: Word Embeddings Visualization

## Context

You are building a module for the **Zero to GPT: The Agentic Architecture** project — a teaching series that builds a GPT from scratch. This is the very first coding episode (Episode 2, Chapter 2). The audience has zero prior context about how LLMs handle text.

This episode introduces **word embeddings** — the idea that words must be represented as vectors of real numbers before a neural network can process them. No model is being built yet; this episode is purely conceptual with a supporting visualization.

## What to Build

### File: `src/llm_from_scratch/visualize_embeddings.py`

A self-contained script that:

1. **Loads pre-trained word vectors** from GloVe (Global Vectors for Word Representation, Pennington et al., 2014). Use the smallest GloVe file: `glove.6B.50d.txt` (50-dimensional vectors, ~66MB).
   - Download URL: `https://nlp.stanford.edu/data/glove.6B.zip`
   - If the file doesn't exist locally, print a clear message telling the user where to download it and where to place it.

2. **Selects curated word groups** to demonstrate clustering. Use these groups:
   - Animals: `["dog", "cat", "horse", "fish", "bird"]`
   - Colors: `["red", "blue", "green", "yellow", "purple"]`
   - Countries: `["france", "germany", "japan", "brazil", "canada"]`
   - Beverages: `["coffee", "tea", "beer", "juice", "water"]`

3. **Reduces dimensionality** from 50D to 2D using PCA (from sklearn or implemented manually with numpy). PCA is preferred over t-SNE for this demo because it's deterministic and easier to explain.

4. **Creates a scatter plot** using matplotlib that:
   - Color-codes each word group differently
   - Labels each point with its word
   - Includes a legend showing the group names
   - Uses a clean, minimal style suitable for a YouTube thumbnail/screenshot
   - Title: "Words That Appear in Similar Contexts Cluster Together"
   - Saves to `outputs/embedding_clusters.png`

5. **Prints a similarity demo** to the console:
   - Compute cosine similarity between "coffee" and "tea"
   - Compute cosine similarity between "coffee" and "horse"
   - Print both, showing that semantically related words have higher similarity

### File: `tests/test_visualize_embeddings.py`

Write tests that verify:
- The cosine similarity function produces correct values (test with known vectors)
- The PCA reduction preserves the correct output shape (n_words × 2)
- The word loading function handles missing words gracefully (returns None or raises a clear error)

## Technical Constraints

- **Dependencies**: Only use `numpy` and `matplotlib`. If PCA is too complex to implement from scratch in this episode, `sklearn.decomposition.PCA` is acceptable.
- **No PyTorch yet**: This episode predates the PyTorch introduction. Keep it pure numpy/matplotlib.
- **No hardcoded paths**: Use `pathlib.Path` for all file paths. Accept the GloVe file path as a command-line argument or look for it in a default location (`data/glove.6B.50d.txt`).
- **Type hints**: Use Python type hints on all functions.
- **Docstrings**: Every function gets a one-line docstring minimum.

## Code Style

- Line length: 100 characters max
- Follow PEP 8
- Use descriptive variable names: `word_vectors` not `wv`, `similarity_score` not `sim`
- Module-level `__all__` export list

## What NOT to Do

- Do not reference any textbook, book section, or copyrighted tutorial
- Do not use Word2Vec (use GloVe — it's simpler to load as a plain text file)
- Do not build a training loop or neural network — this is a visualization-only episode
- Do not use gensim or any heavy NLP library
