import fitz
import chromadb
import os
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(
    path="./chroma_db"
)

try:
    client.delete_collection("medical_docs")
except:
    pass

collection = client.create_collection(
    name="medical_docs"
)

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

PDF_FOLDER = "./pdfs"

for file in os.listdir(PDF_FOLDER):

    if file.endswith(".pdf"):

        path = os.path.join(PDF_FOLDER, file)

        doc = fitz.open(path)

        text = ""

        for page in doc:
            text += page.get_text()

        print(f"\\nProcessing: {file}")

        chunks = []

        chunk_size = 300

        for i in range(0, len(text), chunk_size):

            chunk = text[i:i + chunk_size]

            if len(chunk.strip()) > 50:
                chunks.append(chunk)

        print(f"Chunks Created: {len(chunks)}")

        for idx, chunk in enumerate(chunks):

            embedding = model.encode(chunk).tolist()

            collection.add(
                documents=[chunk],
                ids=[f"{file}_{idx}"],
                embeddings=[embedding],
                metadatas=[{
                    "source": file
                }]
            )

print("\\nVector DB Created Successfully")
print("Total Documents:", collection.count())