from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles  # NEW
from fastapi.responses import HTMLResponse    # NEW
from pydantic import BaseModel
import base64
from typing import List, Dict
import asyncio # For async startup
import os

# Import your refactored RAG pipeline module
from rag_pipeline import initialize_rag_components, query_rag_system

# --- FastAPI App Setup ---
app = FastAPI(
    title="TDS Virtual TA API",
    description="Answers student questions based on course content and Discourse posts, with optional image support."
)

# --- Mount Static Files Directory ---
# This tells FastAPI to serve files from the 'static' directory
# when the browser requests something under '/static'.
# For example, http://127.0.0.1:8000/static/style.css will serve your CSS file.
app.mount("/static", StaticFiles(directory="static"), name="static") # NEW LINE

# --- Pydantic Models for Request/Response ---
class QuestionRequest(BaseModel):
    question: str
    image: str = None # Optional base64 image string

class Link(BaseModel):
    url: str
    text: str

class ApiResponse(BaseModel):
    answer: str
    links: List[Link]

# --- Global variable to hold initialized RAG components ---
# This will be populated once when the API starts
rag_chain = None
llm_instance = None # We might not strictly need this exposed globally, but good for clarity
retriever_instance = None


@app.on_event("startup")
async def startup_event():
    """
    Initializes the RAG components when the FastAPI application starts up.
    """
    global rag_chain, llm_instance, retriever_instance
    print("API Startup: Initializing RAG components...")
    try:
        # Call the initialization function from your rag_pipeline module
        rag_chain, llm_instance, retriever_instance = await asyncio.to_thread(initialize_rag_components)
        print("API Startup: RAG components initialized successfully.")
    except Exception as e:
        print(f"API Startup Error: Failed to initialize RAG components: {e}")
        # You might want to raise an exception to prevent the server from starting if critical
        # raise HTTPException(status_code=500, detail=f"Server failed to initialize RAG system: {e}")

# --- NEW ROOT ENDPOINT to serve your HTML ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serves the main chatbot HTML page."""
    # This reads your index.html file and returns it as an HTML response
    with open("static/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# Your existing API endpoint for questions
@app.post("/api/", response_model=ApiResponse)
async def ask_question(request_data: QuestionRequest):
    """
    Receives a student question and optional image, returns an answer and relevant links.
    """
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG system is still initializing or failed to initialize.")

    # Validate input
    if not request_data.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    print(f"Received question: '{request_data.question}'")
    if request_data.image:
        print("Image attachment detected.")

    try:
        # Call the refactored query function
        # It will handle retrieving text and combining with image for LLM
        rag_response = await query_rag_system(
            question=request_data.question,
            image_data_base64=request_data.image
        )

        answer = rag_response["answer"]
        source_documents = rag_response["source_documents"]

        # --- Format Links for API Response ---
        links_list = []
        seen_urls = set()

        for doc in source_documents:
            url = doc.metadata.get("url")
            # Prioritize 'title' from Markdown, then 'topic_title' from Discourse
            text = doc.metadata.get("title") or doc.metadata.get("topic_title")

            if url and url not in seen_urls: # Ensure URL exists and is unique
                if not text: # If no explicit title, create a snippet
                    content_excerpt = doc.page_content[:150] # Take first 150 chars
                    content_excerpt = content_excerpt.replace('\n', ' ').strip() # Clean newlines
                    if len(doc.page_content) > 150:
                        content_excerpt += "..."
                    text = content_excerpt if content_excerpt else "Relevant Document" # Fallback text

                links_list.append({"url": url, "text": text})
                seen_urls.add(url)

        # Sort links for consistent output (optional but good practice)
        links_list.sort(key=lambda x: x['url'])

        return ApiResponse(answer=answer, links=links_list)

    except Exception as e:
        print(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# Your existing health check endpoint
@app.get("/health")
async def health_check():
    """Checks if the API and RAG components are ready."""
    if rag_chain is None:
        return Response(status_code=503, content="RAG system not ready")
    return Response(status_code=200, content="API and RAG system are ready")