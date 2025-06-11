from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware # <--- ADD THIS IMPORT
from pydantic import BaseModel
import base64
from typing import List, Dict
import asyncio
import os

# Import your refactored RAG pipeline module
from rag_pipeline import initialize_rag_components, query_rag_system

# --- FastAPI App Setup ---
app = FastAPI(
    title="TDS Virtual TA API",
    description="Answers student questions based on course content and Discourse posts, with optional image support."
)

# --- CORS Middleware ---
# Define the origins that are allowed to access your API
# It's crucial to include your Vercel domains here.
# For maximum compatibility during debugging, you can use allow_origins=["*"]
# but for production, list specific domains.
origins = [
    "http://localhost:3000",  # Your local frontend development server
    "http://localhost:8000",  # If you run FastAPI locally on this port
    "https://tds-virtual-ta.vercel.app", # Your main Vercel domain
    # Add any Vercel preview/branch domains that might be generated
    # You can find these in your Vercel dashboard under "Domains" for each deployment
    "https://tds-virtual-ta-git-main-pruthvi-prasad-ss-projects.vercel.app",
    # You might also see domains like:
    # "https://tds-virtual-ta-two.vercel.app",
    # "https://tds-virtual-at4177a3m-pruthvi-prasad-ss-projects.vercel.app", # Example from your earlier logs
    # Add any other specific Vercel domains you are testing from
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)
# --- END CORS Middleware ---


# --- Mount Static Files Directory ---
app.mount("/static", StaticFiles(directory="static"), name="static")

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
rag_chain = None
llm_instance = None
retriever_instance = None


@app.on_event("startup")
async def startup_event():
    """
    Initializes the RAG components when the FastAPI application starts up.
    """
    global rag_chain, llm_instance, retriever_instance
    print("API Startup: Initializing RAG components...")
    try:
        # Load environment variables (already done by dotenv in rag_pipeline)
        # We ensure it's loaded before accessing os.getenv
        # No need to call load_dotenv() directly in api.py if rag_pipeline does it,
        # but it doesn't hurt to ensure it's loaded before `initialize_rag_components`.
        # if not os.getenv("GOOGLE_API_KEY"):
        #     raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it before running.")

        # Call the initialization function from your rag_pipeline module
        rag_chain, llm_instance, retriever_instance = await asyncio.to_thread(initialize_rag_components)
        print("API Startup: RAG components initialized successfully.")
    except Exception as e:
        print(f"API Startup Error: Failed to initialize RAG components: {e}")
        # Re-raising for Vercel to show critical error during startup
        raise HTTPException(status_code=500, detail=f"Server failed to initialize RAG system: {e}")

# --- NEW ROOT ENDPOINT to serve your HTML ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serves the main chatbot HTML page."""
    with open("static/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# Your existing API endpoint for questions
@app.post("/api/", response_model=ApiResponse) # Frontend likely calls this path
async def ask_question(request_data: QuestionRequest):
    """
    Receives a student question and optional image, returns an answer and relevant links.
    """
    if rag_chain is None:
        # This will be caught by the client-side error handling if startup failed
        raise HTTPException(status_code=503, detail="RAG system is still initializing or failed to initialize.")

    # Validate input
    if not request_data.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    print(f"Received question: '{request_data.question}'")
    if request_data.image:
        print("Image attachment detected.")

    try:
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
            text = doc.metadata.get("title") or doc.metadata.get("topic_title")

            if url and url not in seen_urls:
                if not text:
                    content_excerpt = doc.page_content[:150]
                    content_excerpt = content_excerpt.replace('\n', ' ').strip()
                    if len(doc.page_content) > 150:
                        content_excerpt += "..."
                    text = content_excerpt if content_excerpt else "Relevant Document"

                links_list.append({"url": url, "text": text})
                seen_urls.add(url)

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