from fastapi import APIRouter
from pydantic import BaseModel

from openai import OpenAI
from dotenv import load_dotenv

from database import chat_collection

from sentence_transformers import SentenceTransformer

import chromadb
import os

# =========================
# ENV SETTINGS
# =========================

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

router = APIRouter()

# =========================
# LLM CLIENT
# =========================

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

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
    # CREATE QUERY EMBEDDING
    # =========================

    query_embedding = embedding_model.encode(
        data.message
    ).tolist()

    # =========================
    # VECTOR SEARCH
    # =========================

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=8
    )

    docs = results["documents"][0]

    print("\nRETRIEVED DOCS:\n")

    for i, d in enumerate(docs):

        print(f"\nDOCUMENT {i+1}:\n")
        print(d[:500])

    # =========================
    # CONTEXT BUILD
    # =========================

    context = "\n\n".join(docs)

    # =========================
    # LOAD CHAT MEMORY
    # =========================

    previous_chats = list(

        chat_collection.find(

            {
                "email": data.email
            },

            {"_id": 0}

        ).sort("_id", 1)

    )

    conversation_history = ""

    for chat_item in previous_chats:

        conversation_history += f"""

USER:
{chat_item["user"]}

ASSISTANT:
{chat_item["assistant"]}

"""

    # =========================
    # PROMPT
    # =========================

    prompt = f"""
You are an advanced conversational AI assistant.

IMPORTANT RULES:

1. Continue conversation naturally using previous chat history.

2. Remember previous user questions and assistant answers.

3. Maintain conversational context.

4. If user asks follow-up questions like:
- "tell more"
- "what about treatment?"
- "explain again"
- "उसका इलाज क्या है?"
then understand previous context automatically.

5. If user asks in Hindi,
reply in Hindi.

6. If user asks in English,
reply in English.

7. If user uses Hinglish,
reply naturally in Hinglish.

8. Use simple language.

9. Answer ONLY using:
- previous conversation
- retrieved document context

10. Never invent fake information.

11. If answer not found in context,
say:
"यह जानकारी उपलब्ध डॉक्यूमेंट्स में नहीं मिली।"

=========================
PREVIOUS CONVERSATION
=========================

{conversation_history}

=========================
DOCUMENT CONTEXT
=========================

{context}

=========================
CURRENT USER QUESTION
=========================

{data.message}
"""

    # =========================
    # LLM RESPONSE
    # =========================

    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[

            {
                "role": "system",
                "content":
                "You are a multilingual AI medical assistant."
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

    print("\nANSWER:\n")
    print(answer)

    # =========================
    # SAVE CHAT MEMORY
    # =========================

    chat_collection.insert_one({

        "email": data.email,

        "user": data.message,

        "assistant": answer

    })

    # =========================
    # RESPONSE
    # =========================

    return {

        "answer": answer,

        "sources": len(docs)

    }

# =========================
# HISTORY ROUTE
# =========================

@router.get("/history/{email}")

def get_history(email: str):

    chats = list(

        chat_collection.find(

            {
                "email": email
            },

            {"_id": 0}

        ).sort("_id", 1)

    )

    return {

        "history": chats

    }