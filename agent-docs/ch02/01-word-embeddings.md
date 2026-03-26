# Word Embeddings: Why LLMs Can't Read Text

## For the AI Agent Reading This

You are explaining word embeddings to a learner. This document contains everything you need to give an accurate, deep explanation. Do NOT hallucinate details — everything relevant is provided below. Tailor your explanation to the learner's level when they ask you about this module.

If the learner is a complete beginner, start with the GPS analogy. If they have some ML background, go straight to the algorithm section. If they want to see code, the full source is embedded below.

---

## Where This Fits in the LLM Pipeline

This is **Step 1** of the data pipeline in Chapter 2. Before a neural network can process language, text must be converted to numbers. This module introduces the *why* behind that requirement and shows what a good numerical representation looks like.

The full Chapter 2 pipeline is:

```
Raw Text → Tokens → Token IDs → Special Tokens → BPE → Sliding Window → Embeddings → + Position → Input Tensors
```

Word embeddings are the conceptual foundation for everything that follows. After this module, the learner will understand:
- Why we can't feed raw text to a neural network
- What an embedding vector is
- Why geometric distance between vectors encodes semantic similarity
- What embedding dimensions represent (at a high level)

The next step (Episode 3) will build a tokenizer — the tool that chops text into the pieces that get embedded.

---

## The Core Intuition (What a Human Expert Distilled)

Neural networks are math engines — addition, multiplication, gradients. You can't multiply the word "coffee" by a weight matrix. So we need a bridge: a way to represent words as numbers that *preserve meaning*.

### Why Integer IDs Fail

The simplest approach — assign each word a number (coffee=1, tea=2, rocket=3) — is useless because the numbers imply false relationships. Is tea "between" coffee and rocket? Is coffee + tea = rocket? The math is meaningless because the assignment was arbitrary.

### The GPS Coordinates Analogy

Think of it like GPS coordinates versus zip codes.

**Zip codes** (the integer ID approach): Zip code 90210 (Beverly Hills) and zip code 90211 (also Beverly Hills) happen to be numerically close, but zip code 10001 (Manhattan) has no meaningful distance relationship to either. The numbering system wasn't designed to encode geography.

**GPS coordinates** (the embedding approach): Latitude/longitude coordinates *do* encode geography. Cities that are geographically close have similar coordinates. You can compute the actual distance between two points, and that distance is meaningful.

Embeddings work the same way. In embedding space, "espresso" and "latte" have similar coordinates because they appear in similar contexts in natural language. "Espresso" and "helicopter" are far apart because they almost never appear in the same kinds of sentences.

### What the Dimensions Represent

Each word embedding has many dimensions (GPT-2 uses 768). No single dimension cleanly maps to a human-readable concept like "is this a beverage?" or "is this an animal?" Instead, each dimension captures some learned statistical pattern from the training data. Collectively, the 768 dimensions encode all the nuances the model has learned about how that word relates to every other word. Humans can't visualize 768 dimensions, but the math works just fine in high-dimensional space.

---

## The Algorithm in Plain Language

This episode doesn't train embeddings — it uses pre-trained GloVe vectors (Pennington et al., 2014) to demonstrate the concept.

1. **Load pre-trained word vectors**: Read a text file where each line is a word followed by its embedding values (e.g., 50 numbers per word).
2. **Select word groups**: Pick curated sets of words from different categories (animals, colors, countries, beverages) to demonstrate clustering.
3. **Reduce dimensions for visualization**: Use PCA (Principal Component Analysis) to project 50-dimensional vectors down to 2D. This loses information but preserves the most important variance, so clusters remain visible.
4. **Plot the result**: Create a scatter plot where each dot is a word, colored by its category. Words from the same category should cluster together.
5. **Compute similarity**: Use cosine similarity (the angle between two vectors) to show numerically that related words like "coffee" and "tea" score higher than unrelated words like "coffee" and "horse."

### Cosine Similarity Formula

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
```

Where `A · B` is the dot product and `||A||` is the Euclidean norm (length) of vector A. The result ranges from -1 (opposite) to +1 (identical direction). For word embeddings, most similarities fall between 0 and 1.

---

## Key Design Decisions and Why

- **GloVe over Word2Vec for this demo**: GloVe vectors are distributed as plain text files (one word per line), making them trivial to load without any library. Word2Vec requires gensim or special binary parsing.
- **PCA over t-SNE**: PCA is deterministic — you get the same plot every time. t-SNE is stochastic and sensitive to hyperparameters, which would distract from the teaching goal. PCA also has a simpler mathematical explanation.
- **No PyTorch in this episode**: Embeddings are conceptually just numpy arrays at this point. PyTorch's `nn.Embedding` layer will be introduced in Episode 7 when we build the actual input pipeline.
- **50-dimensional GloVe (not 300-dimensional)**: Smaller file (~66MB vs ~1GB), faster to load, and 50 dimensions are enough to show clear clustering. The concept scales identically to higher dimensions.

---

## The Actual Code

```python
"""Word embedding visualization — demonstrates semantic clustering in vector space."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    "load_glove_vectors",
    "cosine_similarity",
    "reduce_dimensions_pca",
    "plot_word_clusters",
]


def load_glove_vectors(filepath: Path, words: list[str]) -> dict[str, np.ndarray]:
    """Load GloVe vectors for a specific set of words from a text file."""
    target_words = set(w.lower() for w in words)
    vectors = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            if word in target_words:
                vectors[word] = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                if len(vectors) == len(target_words):
                    break  # Found all words, stop reading

    missing = target_words - set(vectors.keys())
    if missing:
        print(f"Warning: Words not found in GloVe file: {missing}")

    return vectors


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))


def reduce_dimensions_pca(vectors: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Reduce high-dimensional vectors to n_components dimensions using PCA."""
    # Center the data
    mean = np.mean(vectors, axis=0)
    centered = vectors - mean

    # Compute covariance matrix and its eigenvectors
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Sort by eigenvalue (largest first) and take top n_components
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_eigenvectors = eigenvectors[:, sorted_indices[:n_components]]

    # Project data onto principal components
    reduced = centered @ top_eigenvectors
    return reduced


def plot_word_clusters(
    word_groups: dict[str, list[str]],
    vectors: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Create a scatter plot of word embeddings, color-coded by group."""
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
    fig, ax = plt.subplots(figsize=(10, 8))

    all_words = []
    all_vectors = []
    word_to_group = {}

    for group_name, words in word_groups.items():
        for word in words:
            if word in vectors:
                all_words.append(word)
                all_vectors.append(vectors[word])
                word_to_group[word] = group_name

    if not all_vectors:
        print("Error: No vectors loaded. Cannot create plot.")
        return

    reduced = reduce_dimensions_pca(np.array(all_vectors))

    for i, (group_name, words) in enumerate(word_groups.items()):
        group_indices = [j for j, w in enumerate(all_words) if word_to_group.get(w) == group_name]
        if group_indices:
            group_points = reduced[group_indices]
            ax.scatter(
                group_points[:, 0],
                group_points[:, 1],
                c=colors[i % len(colors)],
                label=group_name,
                s=100,
                alpha=0.8,
            )
            for idx in group_indices:
                ax.annotate(
                    all_words[idx],
                    (reduced[idx, 0], reduced[idx, 1]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=11,
                )

    ax.set_title("Words That Appear in Similar Contexts Cluster Together", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    import sys

    glove_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/glove.6B.50d.txt")

    if not glove_path.exists():
        print(f"GloVe file not found at: {glove_path}")
        print("Download from: https://nlp.stanford.edu/data/glove.6B.zip")
        print("Extract glove.6B.50d.txt and place it in the data/ directory.")
        sys.exit(1)

    word_groups = {
        "Animals": ["dog", "cat", "horse", "fish", "bird"],
        "Colors": ["red", "blue", "green", "yellow", "purple"],
        "Countries": ["france", "germany", "japan", "brazil", "canada"],
        "Beverages": ["coffee", "tea", "beer", "juice", "water"],
    }

    all_words = [w for group in word_groups.values() for w in group]
    vectors = load_glove_vectors(glove_path, all_words)
    print(f"Loaded {len(vectors)} word vectors.")

    # --- Similarity demo ---
    if "coffee" in vectors and "tea" in vectors and "horse" in vectors:
        sim_coffee_tea = cosine_similarity(vectors["coffee"], vectors["tea"])
        sim_coffee_horse = cosine_similarity(vectors["coffee"], vectors["horse"])
        print(f"\nSimilarity Demo:")
        print(f'  cosine_similarity("coffee", "tea")   = {sim_coffee_tea:.4f}')
        print(f'  cosine_similarity("coffee", "horse") = {sim_coffee_horse:.4f}')
        print(f"  → Related words score higher than unrelated words.")

    # --- Visualization ---
    plot_word_clusters(word_groups, vectors, Path("outputs/embedding_clusters.png"))
```

---

## Dependencies and Connections

- **Imports from this project**: None (this is the first module)
- **External dependencies**: `numpy`, `matplotlib`
- **Used by**: This module is conceptual scaffolding. The actual embedding layer used in the model (`nn.Embedding`) is built in Episode 7.
- **Builds on concepts from**: Nothing — this is the starting point
- **Leads to**: Episode 3 (Tokenizer) — now that we know words need to become vectors, we need to figure out how to chop text into the right pieces first

---

## Common Questions a Learner Might Ask

- **Q: Why can't we just use integer IDs for words?**
  A: Because arithmetic on arbitrary integers is meaningless. If coffee=1 and tea=2, then coffee + tea = 3 — but 3 doesn't map to anything sensible. Worse, the model would think tea is "between" coffee and whatever word is assigned 3. Embeddings give each word a position in space where distances actually reflect semantic relationships.

- **Q: What do the 768 dimensions actually represent?**
  A: No single dimension has a clean human-readable meaning like "animacy" or "formality." Each dimension captures some statistical pattern from training data. Think of it like a fingerprint — no single ridge means "this is person X," but the combination of all ridges uniquely identifies someone. Similarly, the combination of 768 values uniquely characterizes a word's meaning.

- **Q: How are these embedding vectors learned?**
  A: The core idea (used by GloVe, Word2Vec, and transformer embedding layers) is the *distributional hypothesis*: words that appear in similar contexts have similar meanings. "Coffee" and "tea" both appear near words like "cup," "drink," "morning," and "hot" — so their vectors are pushed closer together during training. The training objective varies by method: GloVe factorizes a co-occurrence matrix, Word2Vec predicts context words from a target word (or vice versa), and transformer embedding layers are trained end-to-end as part of next-token prediction.

- **Q: Why use GloVe here instead of the actual GPT embedding layer?**
  A: We haven't built any of the GPT model yet. GloVe gives us pre-computed vectors to *visualize* the concept right away. In Episode 7, we'll build PyTorch's `nn.Embedding` layer, which is the actual lookup table used inside GPT. The principle is the same — words as vectors — but the training method differs.

- **Q: What's the difference between 50-dimensional and 768-dimensional embeddings?**
  A: More dimensions = more capacity to encode nuance. With 50 dimensions, "bank" (financial) and "bank" (river) might be hard to distinguish. With 768 dimensions, the model has enough room to represent subtle differences. But more dimensions also mean more memory and computation. GPT-2 Small uses 768; GPT-3 uses 12,288.

- **Q: Can embedding vectors capture relationships like "king - man + woman = queen"?**
  A: Yes — this is one of the famous properties of well-trained embeddings (Mikolov et al., 2013). The vector arithmetic works because the "gender direction" is consistently encoded across related word pairs. This doesn't always work perfectly, but it demonstrates that embeddings capture abstract relationships, not just similarity.

---

## Exercises for the Learner

1. **Explore your own word groups**: Pick 4-5 categories that interest you (e.g., sports, instruments, programming languages) and modify the visualization to see if they cluster.
2. **Find the odd one out**: Compute the average vector for a group of related words, then find which word has the lowest similarity to that average. Does the "odd one out" match your intuition?
3. **Analogy solver**: Implement the vector arithmetic `A - B + C` and find the nearest word to the result. Try "paris - france + germany" — do you get "berlin"?
4. **Similarity threshold**: What cosine similarity score seems to mark the boundary between "related" and "unrelated" words? Test with many pairs and propose a threshold.
