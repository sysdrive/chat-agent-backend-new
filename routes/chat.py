from fastapi import APIRouter
from pydantic import BaseModel

from openai import OpenAI
from database import chat_collection

from sentence_transformers import SentenceTransformer
import chromadb

import os

# =========================
# ENV SETTINGS
# =========================

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

router = APIRouter()

# =========================
# GROQ CLIENT (SAFE INIT)
# =========================

def get_client():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing in Railway environment variables")

    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )

client = get_client()

# =========================
# EMBEDDING MODEL
# =========================

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# CHROMA DB
# =========================

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="medical_docs"
)

# =========================
# REQUEST MODEL
# =========================

class Chat(BaseModel):
    email: str
    message: str

# =========================
# CHAT ROUTE
# =========================

@router.post("/chat")
def chat(data: Chat):

    print("\n======================")
    print("QUESTION:", data.message)

    # =========================
    # DEBUG ENV (OPTIONAL)
    # =========================
    print("GROQ KEY LOADED:", bool(os.getenv("GROQ_API_KEY")))

    # =========================
    # QUERY EMBEDDING
    # =========================

    query_embedding = embedding_model.encode(data.message).tolist()

    # =========================
    # VECTOR SEARCH
    # =========================

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=8
    )

    docs = results["documents"][0] if results.get("documents") else []

    print("\nRETRIEVED DOCS:")

    for i, d in enumerate(docs):
        print(f"\nDOC {i+1}:\n{d[:300]}")

    context = "\n\n".join(docs)

    # =========================
    # CHAT HISTORY
    # =========================

    previous_chats = list(
        chat_collection.find(
            {"email": data.email},
            {"_id": 0}
        ).sort("_id", 1)
    )

    conversation_history = ""

    for c in previous_chats:
        conversation_history += f"""
USER:
{c["user"]}

ASSISTANT:
{c["assistant"]}

"""

    # =========================
    # PROMPT
    # =========================

    prompt = f"""
You are an advanced multilingual AI assistant.

RULES:
- Use previous conversation history
- Use only provided context
- Do NOT hallucinate
- Reply in user's language (Hindi/English/Hinglish)
- If info not found, say:
"यह जानकारी उपलब्ध डॉक्यूमेंट्स में नहीं मिली।"

=========================
CHAT HISTORY
=========================
{conversation_history}

=========================
DOCUMENT CONTEXT
=========================
{context}

=========================
USER QUESTION
=========================
{data.message}
"""

    # =========================
    # LLM CALL
    # =========================

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful multilingual AI assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=700
    )

    answer = response.choices[0].message.content

    print("\nANSWER:\n", answer)

    # =========================
    # SAVE MEMORY
    # =========================

    chat_collection.insert_one({
        "email": data.email,
        "user": data.message,
        "assistant": answer
    })

    return {
        "answer": answer,
        "sources_used": len(docs)
    }

# =========================
# HISTORY ROUTE
# =========================

@router.get("/history/{email}")
def get_history(email: str):

    chats = list(
        chat_collection.find(
            {"email": email},
            {"_id": 0}
        ).sort("_id", 1)
    )

    return {
        "history": chats
    }