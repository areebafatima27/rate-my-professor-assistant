import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { Anonymous_Pro } from "next/font/google";

const systemPrompt = `
  You are a helpful and informative RateMyProfessor agent. Your goal is to assist students in finding the best professors based on their specific needs and preferences.
  Here's how you'll operate:
  1. Understand the Query: Carefully analyze the user's query to identify the key criteria or requirements they are looking for in a professor.
  2. Retrieve Relevant Information: Use the provided RAG (Retrieval Augmented Generation) system to access and process information from the RateMyProfessor database and other relevant sources.
  3. Rank and Recommend: Based on the retrieved information, rank the top 3 professors that best match the user's query. Consider factors such as teaching style, course difficulty, fairness of grading, and student feedback.
  4. Provide Clear and Concise Responses: Present your recommendations in a clear and concise manner, highlighting the key attributes of each professor that make them a good fit for the user.
  Remember: Be helpful, informative, and objective in your responses. Avoid providing personal opinions or biases.
`;

export async function POST(req) {
    try {
        const data = await req.json();
        const pc = new Pinecone({
            apiKey: process.env.PINECONE_API_KEY,
        });

        const index = pc.index("rag").namespace("nsl");

        // Assuming GoogleGenerativeAI has a `generateEmbedding` method, otherwise consult the API docs.
        const genai = new GoogleGenerativeAI({ apiKey: process.env.GOOGLE_API_KEY });
        const text = data[data.length - 1].content;

        // Replace `embed_content` with the correct method based on the API
        const embeddingResponse = await genai.generateEmbedding({
            model: "models/text-embedding-004",
            input: [text],
        });

        const embedding = embeddingResponse.data[0].embedding;

        const results = await index.query({
            topK: 3,
            includeMetadata: true,
            vector: embedding,
        });

        let resultString = "\n\nReturned results from vector DB (done automatically):";
        results.matches.forEach((match) => {
            resultString += `
      Returned Results:
      Professor: ${match.id}
      Review: ${match.metadata.review}
      Subject: ${match.metadata.subject}
      Stars: ${match.metadata.stars}\n\n`;
        });

        const lastMessage = data[data.length - 1];
        const lastMessageContent = lastMessage.content + resultString;
        const lastDataWithoutLastMessage = data.slice(0, data.length - 1);

        const geminiModel = await genai.getGenerativeModel({
            model: "models/gemini-pro",
        });

        const completion = await geminiModel.generateContentStream({
            messages: [
                { role: "system", content: systemPrompt },
                ...lastDataWithoutLastMessage,
                { role: "user", content: lastMessageContent },
            ],
            model: "models/gemini-pro",
            stream: true,
        });

        const stream = new ReadableStream({
            async start(controller) {
                const encoder = new TextEncoder();
                try {
                    for await (const chunk of completion) {
                        const content = chunk.choices[0]?.delta?.content;
                        if (content) {
                            const text = encoder.encode(content);
                            controller.enqueue(text);
                        }
                    }
                } catch (err) {
                    controller.error(err);
                } finally {
                    controller.close();
                }
            },
        });

        return new NextResponse(stream);

    } catch (error) {
        console.error("Error handling POST request:", error);
        return new NextResponse("An error occurred", { status: 500 });
    }
}
