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
