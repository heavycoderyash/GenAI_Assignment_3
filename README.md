# 📚 NotebookLM Clone - Custom RAG Application

A full-stack Retrieval-Augmented Generation (RAG) web application that allows users to upload PDF documents and ask natural language questions about their content. The system guarantees grounded answers by enforcing strict context boundaries, ensuring the LLM relies solely on the uploaded document and does not hallucinate.

## 🚀 Features
* **End-to-End RAG Pipeline:** Handles file ingestion, chunking, embedding, vector storage, semantic retrieval, and grounded generation.
* **Dynamic Document Isolation:** Dynamically generates unique Qdrant collections for every uploaded file to prevent cross-document data bleeding.
* **Strict Anti-Hallucination Prompting:** The LLM is strictly prompted and temperature-controlled to refuse questions whose answers do not exist within the document context.
* **Modern Vanilla UI:** A clean, responsive frontend built with pure HTML, CSS (Flexbox), and JavaScript, featuring typing animations and `marked.js` for beautiful Markdown rendering.

## 🛠️ Tech Stack
* **Frontend:** HTML5, CSS3, Vanilla JavaScript, Marked.js (Markdown parsing)
* **Backend:** Node.js, Express.js, Multer (File Handling)
* **AI & Orchestration:** LangChain (`@langchain/community`, `@langchain/textsplitters`)
* **LLM & Embeddings:** GitHub Models API (`gpt-4o` for chat, `text-embedding-3-large` for embeddings)
* **Vector Database:** Qdrant (Qdrant Cloud)

## 🧠 The RAG Pipeline Architecture
1. **Ingestion:** PDF documents are uploaded via the frontend and temporarily stored on the server using `multer`, then read using LangChain's `PDFLoader`.
2. **Chunking Strategy:** Documents are processed using the **`RecursiveCharacterTextSplitter`**. 
   * **Chunk Size:** 800 characters
   * **Chunk Overlap:** 100 characters
   * *Rationale:* This recursive strategy attempts to split text by paragraphs, then sentences, then words. This ensures that semantic completeness (whole thoughts or sentences) is maintained within chunks, while the 100-character overlap prevents cutting off context between adjacent chunks.
3. **Embedding:** The chunks are converted into dense vector representations using `text-embedding-3-large` via the GitHub Models inference endpoint.
4. **Storage:** The vectors and their corresponding text payloads are stored securely in a dynamic, timestamp-isolated collection in Qdrant Cloud.
5. **Retrieval:** User queries are embedded using the same model, and a semantic similarity search fetches the top 4 most relevant document chunks from Qdrant.
6. **Generation:** The retrieved chunks are injected into a strict system prompt. `gpt-4o` processes this context at a low temperature (`0.3`) to generate an accurate, grounded response.