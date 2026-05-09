import express from "express";
import multer from "multer";
import "dotenv/config";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OpenAIEmbeddings } from "@langchain/openai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { OpenAI } from "openai";

const app = express();
const upload = multer({ dest: "uploads/" });
app.use(express.json());
app.use(express.static("public"));

let currentCollectionName = "initial_collection";

const githubClient = new OpenAI({
    baseURL: process.env.BASE_URL,
    apiKey: process.env.GITHUB_TOKEN,
});

app.post("/upload", upload.single("pdf"), async (req, res) => {
    try {
        if (!req.file) return res.status(400).json({ error: "No file uploaded" });

        const loader = new PDFLoader(req.file.path);
        const docs = await loader.load();

        const textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: 800,
            chunkOverlap: 100,
        });
        const chunkedDocs = await textSplitter.splitDocuments(docs);

        const embeddings = new OpenAIEmbeddings({
            apiKey: process.env.GITHUB_TOKEN,
            model: "text-embedding-3-large",
            configuration: {
                baseURL: process.env.BASE_URL,
            },
        });

        currentCollectionName = "docs_" + Date.now();

        await QdrantVectorStore.fromDocuments(chunkedDocs, embeddings, {
            url: process.env.QDRANT_URL,
            apiKey: process.env.QDRANT_API_KEY,
            collectionName: currentCollectionName,
        });

        res.json({ message: "New document isolated and indexed successfully!" });
    } catch (err) {
        console.error("Upload error:", err);
        res.status(500).json({ error: err.message });
    }
});

app.post("/chat", async (req, res) => {
    try {
        const { query } = req.body;

        const embeddings = new OpenAIEmbeddings({
            apiKey: process.env.GITHUB_TOKEN,
            model: "text-embedding-3-large",
            configuration: {
                baseURL: process.env.BASE_URL,
            },
        });

        const vectorStore = await QdrantVectorStore.fromExistingCollection(
            embeddings,
            {
                url: process.env.QDRANT_URL,
                apiKey: process.env.QDRANT_API_KEY,
                collectionName: currentCollectionName,
            },
        );

        const results = await vectorStore.similaritySearch(query, 4);
        const context = results.map((r) => r.pageContent).join("\n\n---\n\n");

        const systemPrompt = `You are an AI assistant helping a user understand a document. 
        
        RULES:
        1. You must answer the user's question using ONLY the provided context.
        2. You may synthesize, summarize, and connect ideas found within the context to formulate your answer.
        3. If the context does not contain enough information to answer the question, do not guess. Reply exactly with: I cannot answer this based on the uploaded document.
        
        Context:
        ${context}`;

        const response = await githubClient.chat.completions.create({
            model: process.env.MODEL,
            messages: [
                { role: "system", content: systemPrompt },
                { role: "user", content: query },
            ],
            temperature: 0.3,
        });

        res.json({ answer: response.choices[0].message.content });
    } catch (err) {
        console.error("Chat error:", err);
        res.status(500).json({ error: err.message });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));
