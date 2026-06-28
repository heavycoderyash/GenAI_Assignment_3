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

// --- 1. INGEST & CHUNK ---
app.post("/upload", upload.single("pdf"), async (req, res) => {
    try {
        if (!req.file) return res.status(400).json({ error: "No file uploaded" });

        const loader = new PDFLoader(req.file.path);
        const docs = await loader.load();

        // ADVANCED RAG UPGRADE: Larger chunks, better overlap for sentence boundaries
        const textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 200,
        });
        const chunkedDocs = await textSplitter.splitDocuments(docs);

        const embeddings = new OpenAIEmbeddings({
            apiKey: process.env.GITHUB_TOKEN,
            model: "text-embedding-3-large",
            configuration: { baseURL: process.env.BASE_URL },
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

// --- ADVANCED RAG CHAT PIPELINE ---
app.post("/chat", async (req, res) => {
    try {
        const { query } = req.body;
        console.log(`\n--- NEW CHAT REQUEST ---`);
        
        // STEP 1: MULTI-QUERY GENERATION
        console.log(`1. Generating multi-queries for: "${query}"`);
        const multiQueryPrompt = `You are an AI assistant. Generate 3 slightly different versions of the user's question to retrieve relevant documents from a vector database. Use alternative keywords and phrasing.
        Return ONLY the 3 alternative questions, each on a new line. Do not use numbering or bullet points.`;

        const multiQueryResponse = await githubClient.chat.completions.create({
            model: process.env.MODEL, // e.g., "gpt-4o-mini" or "gpt-4o"
            messages: [
                { role: "system", content: multiQueryPrompt },
                { role: "user", content: query }
            ],
            temperature: 0.5,
        });

        const alternativeQueries = multiQueryResponse.choices[0].message.content
            .split('\n')
            .map(q => q.trim())
            .filter(q => q.length > 0);

        const searchQueries = [query, ...alternativeQueries];
        console.log(`   Generated variants:`, alternativeQueries);


        // STEP 2: PARALLEL RETRIEVAL & DEDUPLICATION
        console.log(`2. Querying Qdrant and deduplicating...`);
        const embeddings = new OpenAIEmbeddings({
            apiKey: process.env.GITHUB_TOKEN,
            model: "text-embedding-3-large",
            configuration: { baseURL: process.env.BASE_URL },
        });

        const vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
            url: process.env.QDRANT_URL,
            apiKey: process.env.QDRANT_API_KEY,
            collectionName: currentCollectionName,
        });

        // Search all 4 queries in parallel
        const searchPromises = searchQueries.map(q => vectorStore.similaritySearch(q, 3));
        const allSearchResults = await Promise.all(searchPromises);
        
        // Flatten and Deduplicate chunks using text content
        const uniqueChunksMap = new Map();
        allSearchResults.flat().forEach(chunk => {
            if (!uniqueChunksMap.has(chunk.pageContent)) {
                uniqueChunksMap.set(chunk.pageContent, chunk);
            }
        });
        const uniqueChunks = Array.from(uniqueChunksMap.values());
        console.log(`   Found ${uniqueChunks.length} unique candidate chunks.`);


        // STEP 3: RERANKING (LLM-as-a-judge)
        console.log(`3. Reranking candidates...`);
        let chunkTextForScoring = uniqueChunks.map((chunk, index) => `[CHUNK ${index}]\n${chunk.pageContent}\n`).join('\n');
        
        const rerankPrompt = `You are a relevance judge. Score each document chunk from 0.0 to 1.0 based on how well it helps answer this question: "${query}"
        
        0.0 = completely irrelevant. 1.0 = highly relevant.
        
        CHUNKS:
        ${chunkTextForScoring}
        
        You MUST respond with a valid JSON object containing a single array called "scores". The numerical scores must be in the exact order of the chunks.
        Example output:
        { "scores": [0.9, 0.1, 0.8, 0.0] }`;

        const rerankResponse = await githubClient.chat.completions.create({
            model: process.env.MODEL,
            messages: [{ role: "user", content: rerankPrompt }],
            response_format: { type: "json_object" },
            temperature: 0.1,
        });

        const parsedScores = JSON.parse(rerankResponse.choices[0].message.content).scores;

        // Keep chunks scoring >= 0.3 and sort descending
        const topChunks = uniqueChunks.map((chunk, i) => ({
            content: chunk.pageContent,
            score: parsedScores[i] || 0
        })).filter(c => c.score >= 0.3).sort((a, b) => b.score - a.score);

        console.log(`   Kept ${topChunks.length} high-scoring chunks.`);

        if (topChunks.length === 0) {
            return res.json({ answer: "I cannot answer this based on the uploaded document." });
        }


        // STEP 4: CONTEXT COMPRESSION
        console.log(`4. Compressing context...`);
        const rawContextText = topChunks.map(c => c.content).join("\n\n---\n\n");
        
        const compressionPrompt = `You are an extraction assistant. Extract ONLY the specific sentences, facts, and data from the provided text that are directly necessary to answer the user's question. Exclude all unrelated filler, introductions, or off-topic information.
        
        User Question: "${query}"
        
        Text:
        ${rawContextText}`;

        const compressionResponse = await githubClient.chat.completions.create({
            model: process.env.MODEL,
            messages: [{ role: "user", content: compressionPrompt }],
            temperature: 0.1,
        });

        const compressedContext = compressionResponse.choices[0].message.content;


        // STEP 5: GROUNDED GENERATION
        console.log(`5. Generating final grounded answer...`);
        const systemPrompt = `You are a strict AI assistant helping a user understand a document. 
        
        RULES:
        1. You must answer the user's question using ONLY the provided context.
        2. You may synthesize, summarize, and connect ideas found within the context to formulate your answer.
        3. If the context does not contain enough information to answer the question, do not guess. Reply exactly with: I cannot answer this based on the uploaded document.
        
        Context:
        ${compressedContext}`;

        const finalResponse = await githubClient.chat.completions.create({
            model: process.env.MODEL,
            messages: [
                { role: "system", content: systemPrompt },
                { role: "user", content: query },
            ],
            temperature: 0.3,
        });

        console.log(`--- PIPELINE COMPLETE ---\n`);
        res.json({ answer: finalResponse.choices[0].message.content });
        
    } catch (err) {
        console.error("Chat error:", err);
        res.status(500).json({ error: err.message });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));