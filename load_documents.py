import psycopg2
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example documents
documents = [
    "Strand Agents is an AI framework for RAG.",
    "DeepSeek is a local LLM server.",
    "This agent retrieves from Postgres and uses DeepSeek to answer."
]

# Connect to Postgres
conn = psycopg2.connect(
    dbname="ai_agent",
    user="postgres",
    password="xxx",   # Replace with your password
    host="localhost",
    port=5432
)
cur = conn.cursor()

# Insert embeddings
for doc in documents:
    emb = model.encode(doc).tolist()
    cur.execute(
        "INSERT INTO documents (content, embedding) VALUES (%s, %s)", 
        (doc, emb)
    )

conn.commit()
cur.close()
conn.close()

print("✅ Documents inserted!")
