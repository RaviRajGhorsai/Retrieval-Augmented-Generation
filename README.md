# RAG-Based Movie Search Engine

A cli based **Retrieval-Augmented Generation (RAG)** system built from scratch that combines **keyword search, semantic search, hybrid ranking, and LLM-powered enhancements** to deliver highly relevant movie results.

This project is a deep dive into modern search systems, covering everything from **classical IR (Information Retrieval)** to **LLM-powered intelligent retrieval**.

---

## Features

-  Keyword-based search (TF-IDF, BM25)
-  Semantic search using embeddings
-  Hybrid search (BM25 + Semantic)
-  Re-ranking with cross-encoders and LLMs
-  Query enhancement using LLMs
-  Evaluation metrics (Precision, Recall, F1)
-  Chunking strategies for better retrieval
-  Multimodal search (image → text retrieval)
-  Agentic & recursive RAG concepts

---

## What I Learned & Implemented

### 1. Data Loading
- Built a dataset of movies with titles and descriptions.

---

### 2. Text Preprocessing

Core NLP pipeline:
- Lowercasing (case normalization)
- Removing punctuation
- Tokenization (text → words)
- Stopword removal
- Stemming (reducing words to root form)

---

### 3. Inverted Index

- Built a mapping of:
  ```
  word → list of documents containing that word
  ```
- Enables efficient keyword lookup.

---

### 4. Boolean Search

- Implemented logical queries:
  - AND
  - OR
  - NOT

---

### 5. TF (Term Frequency)

- Measures how often a term appears in a document.

---

### 6. IDF (Inverse Document Frequency)

- Penalizes common words:
  ```
  IDF = log(N / df)
  ```

---

### 7. TF-IDF

- Combines TF and IDF:
  ```
  TF-IDF = TF × IDF
  ```
- Provides better ranking than raw term frequency.

---

### 8. BM25 (Advanced Keyword Search)

Improved version of TF-IDF with:

- Better TF scaling:
  ```
  tf = (tf * (k1 + 1)) / (tf + k1)
  ```

- Improved IDF:
  ```
  IDF = log((N - df + 0.5) / (df + 0.5) + 1)
  ```

- Document length normalization:
  ```
  length_norm = 1 - b + b * (doc_length / avg_doc_length)
  ```

- Final score:
  ```
  BM25 = TF_component × IDF
  ```

---

### 9. Semantic Search

#### 🔹 Embeddings
- Converted text into vectors representing meaning.

#### 🔹 Model Used
- `all-MiniLM-L6-v2` (pretrained embedding model)

#### 🔹 Workflow
- Generate embeddings for:
  - Documents → saved as `.npy`
  - Queries → computed at runtime

#### 🔹 Similarity
- Used **cosine similarity**:
  ```
  similarity ∈ [-1, 1]
  ```

---

### 10. Chunking Strategies

- Fixed-size chunking
- Chunk overlap (preserve context)
- Semantic chunking (sentence/paragraph-based)

> Explored (but not implemented):
- ColBERT
- Late Chunking

---

### 11. Hybrid Search

Combining keyword + semantic search:

#### 🔹 Score Normalization
- Min-Max normalization:
  ```
  normalized = (score - min) / (max - min)
  ```

#### 🔹 Weighted Fusion
```
hybrid_score = α * bm25 + (1 - α) * semantic
```

#### 🔹 Reciprocal Rank Fusion (RRF)
```
score = 1 / (k + rank)
```

---

### 12. LLM Integration

Used LLMs to improve search quality:

- Spelling correction
- Query rewriting
- Query expansion

---

### 13. Re-Ranking

Improves top results quality:

#### 🔹 Two-Stage Retrieval
1. Retrieve top ~25 candidates (fast)
2. Re-rank top results (slow but accurate)

#### 🔹 Methods
- LLM-based re-ranking
- Cross-encoder models

---

### 14. Evaluation

#### 🔹 Metrics

- **Precision**
  ```
  precision = relevant_retrieved / total_retrieved
  ```

- **Recall**
  ```
  recall = relevant_retrieved / total_relevant
  ```

- **F1 Score**
  ```
  F1 = 2 * (precision * recall) / (precision + recall)
  ```

#### 🔹 Techniques
- Manual evaluation
- Golden datasets
- Error analysis
- LLM-based evaluation

---

### 15. Retrieval-Augmented Generation (RAG)

- Generate answers using retrieved documents
- LLM-based:
  - Summarization with citations
  - Question answering

---

### 16. Agentic RAG (Conceptual)

- Recursive RAG
- Agent-based search workflows

---

### 17. Multimodal Search 

#### 🔹 Image → Text Retrieval

- Generated embeddings for:
  - Images
  - Text documents

#### 🔹 Model Inspiration
- CLIP (Contrastive Language–Image Pretraining)

#### 🔹 Implementation
- Used `SentenceTransformer` for image embeddings

#### 🔹 Workflow
1. Convert image → embedding
2. Compare with text embeddings
3. Rank using cosine similarity

---

##  Tech Stack

- Python
- NumPy
- SentenceTransformers
- HuggingFace Models
- LLM APIs
- Custom CLI tools

---

##  Project Structure (Example)

```
rag-search-engine/
│
├── cli/
├── lib/
│   ├── semantic_search.py
│   ├── keyword_search.py
│   ├── hybrid_search.py
│   └── rerank.py
│
├── data/
├── embeddings.npy
└── README.md
```

---

##  Future Improvements

- Add full multimodal embeddings (CLIP-based)
- Improve ranking with learning-to-rank models
- Deploy as API / web app
- Add real-time indexing
- Optimize latency for large datasets

---

##  Conclusion

This project covers the **full lifecycle of a modern search system**, including:

- Classical IR (TF-IDF, BM25)
- Neural search (embeddings)
- Hybrid ranking
- LLM-powered enhancements
- Multimodal retrieval

It serves as a **complete foundation for building production-grade RAG systems**.


## Installation
 
### 1. Clone the Repository
```bash
git clone https://github.com/RaviRajGhorsai/Retrieval-Augmented-Generation
cd Retrieval-Augmented-Generation
```
 
### 2. Create a Virtual Environment (recommended)
```bash
uv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```
 
### 3. Install Requirements
```bash
uv sync
```

## Available Commands

## Keyword Search

### 4. Build the Search Index
Before running any search commands, build the cache first:
```bash
uv cli/keyword_search_cli.py build
```
 
---
  
### `build`
Build and save the cache for BM25 search.
```bash
uv run cli/keyword_search_cli.py build
```
 
---
 
### `bm25search`
Search movies using full BM25 scoring with an optional result limit.
```bash
uv run cli/keyword_search_cli.py.py bm25search "<query>" [--limit N]
```
| Argument  | Type   | Required | Default | Description                            |
|-----------|--------|----------|---------|----------------------------------------|
| `query`   | string | Yes      | —       | Search query                           |
| `--limit` | int    | No       | `5`     | Maximum number of results to return    |
 
**Example:**
```bash
uv run cli/keyword_search_cli.py bm25search "space adventure" --limit 10
```
 
---
 
## Quick Reference
 
| Command      | Description                                 |
|--------------|---------------------------------------------|
| `build`      | Build and cache index for BM25 search       |
| `search`     | Basic keyword movie search                  |
| `tf`         | Term Frequency for a doc/term pair          |
| `idf`        | Inverse Document Frequency for a term       |
| `tfidf`      | TF-IDF score for a doc/term pair            |
| `bm25idf`    | BM25 IDF score for a term                   |
| `bm25tf`     | BM25 TF score for a doc/term pair           |
| `bm25search` | Full BM25 ranked search with result limit   |

---

## Semantic Search

### `search_chunked`
Run semantic search over chunked document embeddings.
```bash
uv run cli/semantic_search_cli.py search_chunked "<query>" [--limit N]
```
| Argument  | Type   | Required | Default | Description              |
|-----------|--------|----------|---------|--------------------------|
| `query`   | string | Yes      | —       | Input query              |
| `--limit` | int    | No       | `5`     | Number of top results    |
 
**Example:**
```bash
uv run cli/semantic_search_cli.py search_chunked "climate change effects" --limit 5
```
 
---
 
## Quick Reference
 
| Command            | Description                                          |
|--------------------|------------------------------------------------------|
| `verify_model`     | Verify the embedding model is initialized            |
| `embed_text`       | Embed a single piece of text                         |
| `verify_embeddings`| Generate and verify embeddings for all documents     |
| `embedquery`       | Generate an embedding for a query                    |
| `search`           | Semantic search over document embeddings             |
| `chunk`            | Split text into fixed-size chunks                    |
| `semantic_chunk`   | Split text into chunks respecting language structure |
| `embed_chunks`     | Embed all document chunks                            |
| `search_chunked`   | Semantic search over chunked document embeddings     |

---

## Hybrid Search

### `rrf-search`
Perform a hybrid search using Reciprocal Rank Fusion (RRF), combining BM25 and semantic rankings. Supports optional query enhancement, LLM-based re-ranking, and result evaluation.
```bash
uv run cli/hybrid_search_cli.py rrf-search "<query>" [--k N] [--limit N] [--enhance METHOD] [--rerank-method METHOD] [--evaluate]
```
| Argument          | Type   | Required | Default | Description                                                                                      |
|-------------------|--------|----------|---------|--------------------------------------------------------------------------------------------------|
| `query`           | string | Yes      | —       | Search query                                                                                     |
| `--k`             | int    | No       | `60`    | Controls weighting between higher-ranked vs. lower-ranked results (higher k = more even weighting) |
| `--limit`         | int    | No       | `5`     | Number of top results to return                                                                  |
| `--enhance`       | string | No       | —       | Query enhancement method: `spell`, `rewrite`, or `expand`                                        |
| `--rerank-method` | string | No       | —       | LLM re-ranking method: `individual`, `batch`, or `cross_encoder`                                 |
| `--evaluate`      | flag   | No       | `false` | LLM evaluates result relevance and accuracy                                                      |
 
#### `--enhance` options
| Value     | Description                                      |
|-----------|--------------------------------------------------|
| `spell`   | Correct spelling errors in the query             |
| `rewrite` | Rewrite the query for better search performance  |
| `expand`  | Expand the query with related terms              |
 
#### `--rerank-method` options
| Value           | Description                                      |
|-----------------|--------------------------------------------------|
| `individual`    | Re-rank each result individually using an LLM    |
| `batch`         | Re-rank all results together in a single LLM call|
| `cross_encoder` | Use a cross-encoder model for re-ranking         |
 
**Examples:**
```bash
# Basic RRF search
uv run cli/hybrid_search_cli.py rrf-search "space exploration" --limit 10
 
# With query expansion and batch re-ranking
uv run cli/hybrid_search_cli.py rrf-search "climate change" --enhance expand --rerank-method batch --limit 5
 
# Full pipeline with evaluation
uv run cli/hybrid_search_cli.py rrf-search "dark comedy films" --k 30 --enhance rewrite --rerank-method cross_encoder --evaluate
```
 
---
 
## Quick Reference
 
| Command           | Description                                                   |
|-------------------|---------------------------------------------------------------|
| `normalize`       | Normalize a list of BM25 or semantic scores                   |
| `weighted-search` | Hybrid search with configurable alpha weighting               |
| `rrf-search`      | Hybrid search using Reciprocal Rank Fusion with optional query enhancement, re-ranking, and evaluation |

---

## Augmented Generation

### `rag`
Perform a full RAG pipeline — retrieves relevant documents and generates an AI answer based on the results.
```bash
uv run cli/augmented_generation_cli.py rag "<query>"
```
| Argument | Type   | Description        |
|----------|--------|--------------------|
| `query`  | string | Search query for RAG |
 
**Example:**
```bash
uv run cli/augmented_generation_cli.py rag "What are the effects of climate change on biodiversity?"
```
 
---
 
### `summary`
Retrieve documents matching the query and return an AI-generated summary of the results.
```bash
uv run cli/augmented_generation_cli.py summary "<query>" [--limit N]
```
| Argument  | Type   | Required | Default | Description                     |
|-----------|--------|----------|---------|---------------------------------|
| `query`   | string | Yes      | —       | Search query                    |
| `--limit` | int    | No       | `5`     | Number of top results to summarize |
 
**Example:**
```bash
uv run cli/augmented_generation_cli.py summary "recent advances in renewable energy" --limit 10
```
 
---
 
### `question`
Ask a natural language question and get an AI-generated answer grounded in retrieved data.
```bash
uv run cli/augmented_generation_cli.py question "<question>" [--limit N]
```
| Argument   | Type   | Required | Default | Description                          |
|------------|--------|----------|---------|--------------------------------------|
| `question` | string | Yes      | —       | Question to answer from retrieved data |
| `--limit`  | int    | No       | `5`     | Number of top results to retrieve    |
 
**Example:**
```bash
uv run cli/augmented_generation_cli.py question "Who directed Inception?" --limit 5
```
 
---
 
## Quick Reference
 
| Command    | Description                                              |
|------------|----------------------------------------------------------|
| `rag`      | Retrieve documents and generate an AI answer             |
| `summary`  | Retrieve documents and generate an AI summary            |
| `question` | Ask a question and get an answer from retrieved data     |

---

## MultiModal Search

### `image-search`
Search for movies using an image as the query input.
```bash
uv run cli/multimodel_search_cli.py image-search "<img_path>"
```
| Argument   | Type   | Description             |
|------------|--------|-------------------------|
| `img_path` | string | Path to the input image |
 
**Example:**
```bash
uv run cli/multimodel_search_cli.py image-search ./images/movie_poster.png
```
 
---
 
## Quick Reference
 
| Command                  | Description                                      |
|--------------------------|--------------------------------------------------|
| `verify-image-embedding` | Verify image can be embedded by the model        |
| `image-search`           | Search movies using an image as the query        |
 
