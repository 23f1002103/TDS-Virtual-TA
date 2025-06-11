from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
from typing import List, Dict
import asyncio
import os
# from dotenv import load_dotenv # Assuming rag_pipeline.py handles this

# Import your refactored RAG pipeline module
from rag_pipeline import initialize_rag_retriever_only, query_rag_system # Changed import

# Import ChatGoogleGenerativeAI here
from langchain_google_genai import ChatGoogleGenerativeAI # ADD THIS LINE
from langchain.chains import RetrievalQA # ADD THIS LINE (already imported, just for clarity)
from langchain.prompts import PromptTemplate # ADD THIS LINE (already imported, just for clarity)


# --- FastAPI App Setup ---
app = FastAPI(
    title="TDS Virtual TA API",
    description="Answers student questions based on course content and Discourse posts, with optional image support."
)

# --- CORS Middleware ---
# Define the origins that are allowed to access your API
origins = [
    "http://localhost:3000",  # Your local frontend development server
    "http://localhost:8000",  # If you run FastAPI locally on this port
    "https://tds-virtual-ta.vercel.app", # Your main Vercel domain
    # Add any Vercel preview/branch domains that might be generated dynamically.
    # You might consider using a dynamic origin check for production if needed,
    # but explicitly listing them is safer for Vercel deployments.
    "https://tds-virtual-ta-git-main-pruthvi-prasad-ss-projects.vercel.app",
    "https://tds-virtual-ta-two.vercel.app", # As seen in previous logs
    # Add any other specific Vercel domains you are testing from here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
# rag_chain = None # No longer global in api.py, will be passed to query_rag_system
llm_instance = None
retriever_instance = None
qa_chain_global = None # New global to hold the initialized QA chain

# IMPORTANT: Re-define CUSTOM_QA_PROMPT here as it's used directly in api.py now
CUSTOM_QA_PROMPT = PromptTemplate(
    template="""You are a helpful assistant for university students. Your task is to answer questions based *only* on the provided context and any image provided.
If the context or image does not contain enough information to answer the question, or if you cannot find the answer, please politely state "I am sorry, but I do not have enough information in my knowledge base or the provided image to answer that question." Do not try to make up an answer.

Context:
{context}

Question: {question}
Answer:""",
    input_variables=["context", "question"],
)


@app.on_event("startup")
async def startup_event():
    """
    Initializes the RAG components when the FastAPI application starts up.
    """
    global llm_instance, retriever_instance, qa_chain_global # Updated globals
    print("API Startup: Initializing RAG components...")
    try:
        # Step 1: Initialize retriever (synchronously)
        retriever_instance = await asyncio.to_thread(initialize_rag_retriever_only) # Changed function call
        print("API Startup: Retriever initialized successfully.")

        # Step 2: Initialize LLM (asynchronously)
        print(f"⏳ Initializing Google LLM: {os.getenv('GOOGLE_LLM_MODEL')}...")
        llm_instance = ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_LLM_MODEL"), temperature=0.2) # Use os.getenv
        print("✅ Google LLM loaded.")

        # Step 3: Initialize RetrievalQA chain
        qa_chain_global = RetrievalQA.from_chain_type(
            llm=llm_instance,
            chain_type="stuff",
            retriever=retriever_instance,
            return_source_documents=True,
            chain_type_kwargs={"prompt": CUSTOM_QA_PROMPT}
        )
        print("API Startup: RetrievalQA chain initialized successfully.")

        print("API Startup: All RAG components initialized successfully.")
    except Exception as e:
        print(f"API Startup Error: Failed to initialize RAG components: {e}")
        raise HTTPException(status_code=500, detail=f"Server failed to initialize RAG system: {e}")


# --- NEW ROOT ENDPOINT to serve your HTML ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serves the main chatbot HTML page."""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Frontend HTML file not found. Ensure 'static/index.html' exists.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading HTML file: {e}")


# Your existing API endpoint for questions
@app.post("/api/", response_model=ApiResponse)
async def ask_question(request_data: QuestionRequest):
    """
    Receives a student question and optional image, returns an answer and relevant links.
    """
    global qa_chain_global # Declare global to access the initialized chain

    if qa_chain_global is None: # Check the new global var
        raise HTTPException(status_code=503, detail="RAG system is still initializing or failed to initialize.")

    # Validate input
    if not request_data.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    print(f"Received question: '{request_data.question}'")
    if request_data.image:
        print("Image attachment detected.")

    try:
        # Pass the initialized QA chain to query_rag_system
        rag_response = await query_rag_system(
            question=request_data.question,
            image_data_base64=request_data.image,
            qa_chain_instance=qa_chain_global # Pass the initialized chain
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
    global qa_chain_global # Declare global to access the initialized chain
    if qa_chain_global is None: # Check the new global var
        return Response(status_code=503, content="RAG system not ready")
    return Response(status_code=200, content="API and RAG system are ready")