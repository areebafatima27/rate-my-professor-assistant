import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { Anonymous_Pro } from "next/font/google";

const systemPrompt =
  "You are a helpful and informative RateMyProfessor agent. Your goal is to assist students in finding the best professors based on their specific needs and preferences.Here's how you'll operate:Understand the Query: Carefully analyze the user's query to identify the key criteria or requirements they are looking for in a professor.Retrieve Relevant Information: Use the provided RAG (Retrieval Augmented Generation) system to access and process information from the RateMyProfessor database and other relevant sources.Rank and Recommend: Based on the retrieved information, rank the top 3 professors that best match the user's query. Consider factors such as teaching style, course difficulty, fairness of grading, and student feedback.Provide Clear and Concise Responses: Present your recommendations in a clear and concise manner, highlighting the key attributes of each professor that make them a good fit for the user.Remember: Be helpful, informative, and objective in your responses. Avoid providing personal opinions or biases";

export async function POST(req) {
  const data = await req.json();
  const pc = new Pinecone({
    apiKey: process.env.PINECOME_API_KEY,
  });

  const index = pc.index("rag").namespace("nsl");
  const genai = new GoogleGenerativeAI();
  const text = data[data.length - 1].content;
  const embedding = await GoogleGenerativeAI.embed_content({
    model: "models/text-embedding-004",
    input: text,
    encoding_format: "float",
  });

  const results = await index.query({
    topK: 3,
    includeMetadata: true,
    vector: embedding.data[0].embedding,
  });

  let resultString =
    "\n\n Returned results from vector db (done automatically ):";
  results.matches.forEach((match) => {
    resultString += `
  Returned Results:
  Professor: ${match.id}
  Review: ${match.metadata.stars}
  Subject: ${match.metadata.subject}
  Stars: ${match.metadata.stars}
  \n\n`;
  });

  const lastMessage = data[data.length - 1];
  const lastMessageContent = lastMessage.content + resultString;
  const lastDataWithoutLastMessage = data.slice(0, data.length - 1);
  //const geminiModel = genai.getGenerativeModel({
  // model: "models/text-embedding-004",
  //});
  const completion = await geminiModel.generateContentStream({
    messages: [
      { roles: "system, content: systemPrompt" },
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
}
