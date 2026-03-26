# Episode 2: "What Are Word Embeddings? Why LLMs Can't Read Text"

**Series**: Zero to GPT: The Agentic Architecture
**Chapter**: 2 — Working with Text Data
**Episode Duration**: ~10-12 minutes
**Agentic Phase**: Phase 1 (Single Agent)
**PR**: `ch02/01-word-embeddings`

---

## VIDEO SCRIPT

---

### COLD OPEN [0:00–0:45]

**[ON CAMERA — you at your desk, code editor visible behind you]**

> Here's something that tripped me up when I started learning about LLMs. I kept hearing "the model processes text" — and I just... assumed it read words the way we do. It doesn't. It can't. A neural network has never seen a word in its life. All it knows is numbers — addition, multiplication, gradients.
>
> So today's question is simple: how do you turn the word "coffee" into something a math engine can actually work with? And more importantly — how do you do it without losing what "coffee" *means*?
>
> This is Episode 2 of Zero to GPT. Let's figure this out.

**[TITLE CARD]**: "What Are Word Embeddings? Why LLMs Can't Read Text"

---

### SECTION 1: THE PROBLEM [0:45–2:30]

**[SCREEN RECORDING — show a simple Python REPL or notebook]**

> Let me show you the problem concretely.

**[Type and run]:**
```python
text = "coffee"
text * 3.5
```
**→ Error (or "coffeecoffeecoffeecoffee..." depending on framing)**

> Python doesn't know what it means to "multiply" a word by a weight. But that's exactly what a neural network does thousands of times per second — matrix multiplications. Every single layer of a transformer is: take an input, multiply it by a weight matrix, add a bias, apply an activation. Repeat.
>
> So before anything else can happen, we need to convert text into numbers. But not just *any* numbers.

---

### SECTION 2: THE WRONG WAY — INTEGER IDs [2:30–4:00]

**[SCREEN RECORDING — notebook or slides]**

> The most obvious approach: assign each word a number.

**[Show on screen]:**
```
coffee → 1
tea    → 2
rocket → 3
```

> Seems reasonable, right? But now think about what the math does with these.
>
> If the model computes "coffee + tea" — that's 1 + 2 = 3. And 3 is... rocket? Does that make sense? Is tea somehow "between" coffee and rocket on some meaningful scale?
>
> No. The assignment was arbitrary. We could just as easily have said coffee=7429, tea=12, rocket=3. The math would be completely different, but the words haven't changed.
>
> Integer IDs tell the model *which* word it's looking at — like a name tag — but they say nothing about what the word *means*. And meaning is exactly what the model needs.

---

### SECTION 3: THE RIGHT WAY — THE GPS ANALOGY [4:00–6:30]

**[VISUAL — show a simple map graphic, then transition to a coordinate plane]**

> Here's the analogy that made it click for me.
>
> Think about zip codes versus GPS coordinates.
>
> **Zip codes** are like integer IDs. Beverly Hills is 90210. Manhattan is 10001. Those numbers don't tell you anything about the actual distance between them. It's just an arbitrary labeling system.
>
> **GPS coordinates** are different. Every place on Earth gets a latitude and a longitude — two numbers that *actually encode where it is*. Cities that are close together have similar coordinates. You can compute the real distance between two places just by doing math on their coordinates. The numbers are *meaningful*.
>
> Word embeddings work exactly like GPS, but instead of latitude and longitude — two dimensions — each word gets a position in a space with *hundreds* of dimensions. GPT-2 uses 768. GPT-3 uses over 12,000.

**[VISUAL — transition to a 2D scatter plot showing word clusters]**

> And just like nearby cities on a map share a region, words that appear in similar contexts end up near each other in embedding space. "Espresso" and "latte" — close together, because they show up near words like "cup," "barista," "morning." "Espresso" and "helicopter" — far apart, because they almost never appear in the same kinds of sentences.
>
> That's the core idea. An embedding is a set of coordinates that places a word in a meaningful space where distance equals similarity.

---

### SECTION 4: THE AGENTIC BUILD [6:30–9:00]

**[SCREEN RECORDING — show your code editor + terminal, agent conversation visible]**

> Now here's where the Agentic Engineering comes in. I'm not going to write this visualization code by hand. I understand the concept — I know what I want to show — so I'm going to direct an AI agent to build it.

**[Show the agent prompt on screen — briefly, not reading every line]**

> I've written a prompt that describes exactly what to build: load pre-trained GloVe vectors — these are word embeddings that were trained on billions of words of web text — select four groups of words, reduce the dimensions down to 2D so we can actually plot them, and create a scatter plot.
>
> Watch what I'm doing here. I'm not saying "write me some code." I'm specifying the algorithm, the visualization requirements, the file structure, what libraries to use, and what NOT to do. That's the difference between agentic engineering and vibe coding.

**[Run the agent — show it generating the code. You can time-lapse or cut this]**

> The agent writes the code. Let me review it.

**[Scroll through the generated code, pausing at key points]**

> Cosine similarity function — looks correct. PCA implementation using numpy — clean. The plotting function uses matplotlib with color-coding per group. Good.
>
> One thing I always check: does it handle the case where a word isn't found in the GloVe file? *[scroll to that section]* Yes — it prints a warning and continues. That's the kind of robustness I look for in review.

**[Run the script — show the output]**

> Let's run it.

**[Show terminal output]:**
```
Loaded 20 word vectors.

Similarity Demo:
  cosine_similarity("coffee", "tea")   = 0.9357
  cosine_similarity("coffee", "horse") = 0.2314
  → Related words score higher than unrelated words.

Plot saved to outputs/embedding_clusters.png
```

**[Show the scatter plot]**

> Look at that. Animals cluster together. Colors cluster together. Countries cluster together. Beverages cluster together. These vectors were not *told* that "dog" and "cat" are both animals — they figured it out purely from context patterns in billions of sentences.
>
> And the similarity numbers confirm it: coffee and tea score 0.94 — almost identical direction in the vector space. Coffee and horse? 0.23. The math matches our intuition.

---

### SECTION 5: THE AGENT-AS-TUTOR DEMO [9:00–10:30]

**[SCREEN RECORDING — show a new agent chat window, paste in the agent-doc]**

> Now here's the Karpathy-inspired part. Everything we just built — the code, the intuition, the Q&A — I've packaged into an agent-doc. This is a self-contained markdown file designed for any AI to read and explain.
>
> Let me show you what I mean. I'll paste this agent-doc into a new chat and ask a question at a completely different level.

**[Paste agent-doc, then type]:**
"I'm a high school student. Can you explain what word embeddings are using simple examples?"

**[Show the agent's response — tailored to a high school level]**

> See that? Same document, but the explanation is completely different from what I gave. It's adapted to the learner. That's the power of writing for agents — you encode the knowledge once, and it scales to every skill level.
>
> The agent-doc is in the repo. Link in the description. Paste it into ChatGPT, Claude, Gemini, whatever you use — and ask it anything about embeddings.

---

### SECTION 6: RECAP & NEXT EPISODE [10:30–11:30]

**[ON CAMERA]**

> Let's recap what we covered.
>
> Neural networks can't read text — they need numbers. The naive approach — integer IDs — fails because the numbers are arbitrary and don't encode meaning. Embeddings solve this by placing each word in a high-dimensional space where distance reflects semantic similarity. We used pre-trained GloVe vectors to visualize this: animals cluster with animals, colors with colors.
>
> On the agentic side, we used a single agent to build the visualization from a detailed prompt. I reviewed the output, checked edge cases, and approved.
>
> But there's a question we skipped: before we can look up a word's embedding, we need to decide what counts as a "word." Is "don't" one word or two? Is "!" a word? That's **tokenization** — and it's what we're building from scratch in Episode 3.

**[END CARD]**: Subscribe, link to repo, link to agent-doc

---

## POST-PRODUCTION NOTES

### Screen Recordings Needed
1. Python REPL showing `text * 3.5` error
2. Agent prompt being written/shown
3. Agent generating code (can be time-lapsed)
4. Code review walkthrough
5. Script execution + terminal output
6. Scatter plot result
7. Agent-doc pasted into a new AI chat with learner question

### Graphics/Slides Needed
1. Title card
2. Integer ID assignment visual (coffee=1, tea=2, rocket=3)
3. Zip code vs GPS analogy visual (map graphic)
4. 2D embedding space with labeled clusters
5. Pipeline diagram showing where embeddings fit: `Raw Text → ... → **Embeddings** → ...`
6. End card with links

### Key Files Referenced in This Episode
- Agent prompt: `agent-prompts/ch02/01-word-embeddings.md`
- Agent-doc: `agent-docs/ch02/01-word-embeddings.md`
- Source code: `src/llm_from_scratch/visualize_embeddings.py`
- Tests: `tests/test_visualize_embeddings.py`
- PR: `ch02/01-word-embeddings`
