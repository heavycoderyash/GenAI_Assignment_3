# 📚 NotebookLM Clone - Advanced RAG Application

A full-stack, enterprise-grade Retrieval-Augmented Generation (RAG) web application that allows users to upload PDF documents and have highly accurate, grounded conversations with them. 

This project goes beyond basic RAG by implementing an **Advanced RAG Pipeline**. It tackles the most common AI pitfalls—poor user phrasing, noisy context windows, and hallucinations—by utilizing Multi-Query Generation, LLM-as-a-Judge Reranking, and Context Compression before generating the final response.

## 🚀 Features
* **Advanced RAG Architecture:** Orchestrates a multi-step LLM pipeline (Multi-Query → Parallel Retrieval → Reranking → Compression → Generation) for unparalleled accuracy.
* **Dynamic Document Isolation:** Dynamically generates unique Qdrant collections for every uploaded file to prevent cross-document data bleeding.
* **LLM-as-a-Judge Filtering:** Uses `gpt-4o` to mathematically score and filter out irrelevant context chunks before they reach the final prompt.
* **Context Compression:** Strips away surrounding text fluff, extracting only the exact sentences needed to answer the question, keeping the context window pristine.
* **Modern Vanilla UI:** A clean, responsive frontend built with pure HTML, CSS (Flexbox), and JavaScript, featuring typing animations, parallel processing logs, and `marked.js` for beautiful Markdown rendering.

## 🛠️ Tech Stack
* **Frontend:** HTML5, CSS3, Vanilla JavaScript, Marked.js
* **Backend:** Node.js, Express.js, Multer
* **AI Tooling & Ingestion:** LangChain (`@langchain/community`, `@langchain/textsplitters`)
* **LLM & Embeddings:** GitHub Models API (`gpt-4o` for chat and logic steps, `text-embedding-3-large` for dense vectors)
* **Vector Database:** Qdrant (Qdrant Cloud)

## 🧠 The Advanced RAG Pipeline
This application executes a strict, 9-step pipeline for every single user question:

1. **Ingestion:** PDF documents are uploaded via the frontend and temporarily stored on the server using `multer`, then parsed using LangChain's `PDFLoader`.
2. **Chunking Strategy:** Documents are processed using the **`RecursiveCharacterTextSplitter`**. 
   * **Chunk Size:** 1000 characters
   * **Chunk Overlap:** 200 characters
   * *Rationale:* Captures complete semantic thoughts and paragraphs, with a generous 200-character overlap to prevent cutting off crucial context between boundaries.
3. **Embedding:** Chunks are converted into high-dimensional vectors using `text-embedding-3-large`.
4. **Storage:** Vectors and payloads are stored securely in a dynamic, timestamp-isolated collection in Qdrant Cloud.
5. **Multi-Query Generation:** At query time, the LLM generates 3 alternative phrasings of the user's question to capture different keywords and semantic intent.
6. **Parallel Retrieval:** The system queries Qdrant with all 4 questions (original + 3 variants) in parallel. The resulting chunks are merged and deduplicated into a massive candidate pool.
7. **Reranking (LLM-as-a-Judge):** Each candidate chunk is evaluated by the LLM and scored from `0.0` to `1.0` based on direct relevance. Chunks scoring below `0.3` are discarded.
8. **Context Compression:** The surviving high-value chunks are passed through an extraction prompt. The LLM pulls *only* the specific sentences, facts, and data needed to answer the question, discarding all surrounding noise.
9. **Grounded Generation:** The strictly compressed, pristine context is injected into the final system prompt. The LLM (at a low temperature of `0.3`) generates an accurate, hallucination-free response based *only* on the extracted facts.