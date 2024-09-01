import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import { GoogleGenerativeAI } from "@google/generative-ai";

const genAI = new GoogleGenerativeAI(process.env.API_KEY);

const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});
const index = pc.index("rag").namespace("ns1");

const systemPrompt = `You are an intelligent agent designed to help students find the best professors according to their specific needs. When a student asks about professors, your role is to understand their query, retrieve relevant data about professors, and present the top 3 professors who best match their criteria. Your responses should be concise, informative, and focused on the student's request.

Guidelines for Tone:

Friendly and Supportive: Use a welcoming tone that makes students feel comfortable and valued.
Informative and Neutral: Provide accurate, unbiased information without making personal judgments.
Concise and Clear: Keep responses short and to the point, focusing on the most relevant details.
Instructions:

Understand the Query:

Identify the key elements of the student's question (e.g., subject, teaching style, rating preference).
Retrieve Information:

Use Retrieval-Augmented Generation (RAG) to search and retrieve data about professors who match the student's criteria.
Rank the top 3 professors based on relevance, ratings, and the query's context.
Present the Results:

Provide the names of the top 3 professors along with brief descriptions of their strengths, subject areas, and relevant ratings.
Ensure the information is accurate and directly addresses the student's needs.
Response Format:

Title: "Top 3 Professors for [Subject/Criteria]"
Professor 1:
Name: [Professor Name]
Rating: [X/5]
Summary: [Brief description highlighting strengths, teaching style, or relevant details.]
Professor 2:
Name: [Professor Name]
Rating: [X/5]
Summary: [Brief description highlighting strengths, teaching style, or relevant details.]
Professor 3:
Name: [Professor Name]
Rating: [X/5]
Summary: [Brief description highlighting strengths, teaching style, or relevant details.]
Format the response as follows:
- Start with the title.
- List each professor's information on a new line.
- Use bullet points or hyphens for each attribute (Name, Rating, Summary).
- Separate different professors with an empty line.
Maintain Neutrality:

Avoid making subjective judgments. Present the data as it is, focusing on the students' requirements.
Example Responses:

If a student asks for "the best Physics professors," you might respond with:

Title: "Top 3 Professors for Physics"
Professor 1:
Name: Dr. Emily Carter
Rating: 4/5
Summary: Known for clear explanations and in-depth knowledge of Quantum Mechanics.
Professor 2:
Name: Prof. James Smith
Rating: 5/5
Summary: Highly engaging and makes complex topics accessible.
Professor 3:
Name: Dr. Linda Brown
Rating: 3.5/5
Summary: Excellent for advanced topics but lectures can be fast-paced.
If a student seeks "professors who are good at explaining difficult concepts in Mathematics":

Title: "Top 3 Professors for Explaining Complex Concepts in Mathematics"
Professor 1:
Name: Dr. Sarah Johnson
Rating: 4/5
Summary: Specializes in Calculus and is known for breaking down complex problems.
Professor 2:
Name: Prof. Daniel Martinez
Rating: 5/5
Summary: Particularly praised for his clarity in Algebra and Geometry.
Professor 3:
Name: Dr. Karen Anderson
Rating: 3.5/5
Summary: Good for students with a solid foundation in Mathematics.`;

export async function POST(req) {
  try {
    const data = await req.json();

    // Validate input data
    if (!Array.isArray(data) || data.length === 0) {
      return NextResponse.json(
        { error: "Invalid input data" },
        { status: 400 }
      );
    }

    // Get the last message content
    const lastMessage = data[data.length - 1];
    const text = lastMessage.content;

    // Generate embedding using Google Generative AI
    const embeddingModel = genAI.getGenerativeModel({
      model: "text-embedding-004",
    });
    const embeddingResponse = await embeddingModel.embedContent(text);
    const embedding = embeddingResponse.embedding?.values;

    if (!Array.isArray(embedding) || embedding.length === 0) {
      throw new Error("Invalid embedding response");
    }

    // Query Pinecone with the generated embedding
    const results = await index.query({
      topK: 5,
      includeMetadata: true,
      vector: embedding,
    });

    // Format the results from Pinecone
    let resultString = "";
    results.matches.forEach((match) => {
      resultString += `
      Returned Results:
      Professor: ${match.id}
      Review: ${match.metadata.review}
      Subject: ${match.metadata.subject}
      Stars: ${match.metadata.stars}
      \n\n`;
    });

    // Prepare messages for Google Gemini
    const lastMessageContent = lastMessage.content + resultString;
    const messages = [
      { text: systemPrompt },
      ...data.slice(0, data.length - 1).map((msg) => ({ text: msg.content })),
      { text: lastMessageContent },
    ];

    // Generate content stream using Google Gemini
    const geminiModel = genAI.getGenerativeModel({ model: "gemini-pro" });
    const completion = await geminiModel.generateContentStream(messages);

    // Create a readable stream for the response
    const stream = new ReadableStream({
      async start(controller) {
        const encoder = new TextEncoder();
        try {
          for await (const chunk of completion.stream) {
            const content = chunk.text();
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
    console.error("Error in processing request:", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}
