import psycopg2
import requests
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import json

# 1️⃣ Embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_fn(text):
    return embed_model.encode(text).tolist()

# 2️⃣ DeepSeek LLM via local REST API
class DeepSeekLLM:
    def __init__(self, api_url="http://localhost:11434/api/generate", model="deepseek-r1:7b"):
        self.api_url = api_url
        self.model = model

    def complete(self, prompt: str) -> str:
        try:
            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json={"model": self.model, "prompt": prompt, "stream": False}
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            return f"Error calling LLM: {str(e)}"

# 3️⃣ PostgreSQL + pgvector Retriever
class PGVectorRetriever:
    def __init__(self, conn_str: str, embed_fn, table: str, content_column: str, embedding_column: str):
        self.conn_str = conn_str
        self.embed_fn = embed_fn
        self.table = table
        self.content_column = content_column
        self.embedding_column = embedding_column

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        try:
            # Get query embedding
            query_embedding = self.embed_fn(query)
            
            # Connect to database
            conn = psycopg2.connect(self.conn_str)
            cur = conn.cursor()
            
            # Search for similar documents using cosine similarity
            cur.execute(f"""
                SELECT {self.content_column}, 
                       1 - ({self.embedding_column} <=> %s) as similarity
                FROM {self.table}
                ORDER BY {self.embedding_column} <=> %s
                LIMIT %s
            """, (json.dumps(query_embedding), json.dumps(query_embedding), top_k))
            
            results = [row[0] for row in cur.fetchall()]
            
            cur.close()
            conn.close()
            
            return results
        except Exception as e:
            print(f"Error retrieving documents: {str(e)}")
            return []

# 4️⃣ Simple Agent
class Agent:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def run(self, query: str) -> str:
        # Retrieve relevant documents
        documents = self.retriever.retrieve(query)
        
        if not documents:
            return "No relevant documents found in the database."
        
        # Create context from retrieved documents
        context = "\n".join(documents)
        
        # Create prompt with context
        prompt = f"""Based on the following context, please answer the user's question:

Context:
{context}

Question: {query}

Answer:"""
        
        # Get response from LLM
        return self.llm.complete(prompt)

# 5️⃣ Initialize components
retriever = PGVectorRetriever(
    conn_str="dbname=ai_agent user=postgres password=xxx host=localhost port=5432",
    embed_fn=embed_fn,
    table="documents",
    content_column="content",
    embedding_column="embedding"
)

agent = Agent(
    llm=DeepSeekLLM(),
    retriever=retriever
)

# 6️⃣ CLI loop
if __name__ == "__main__":
    print("💠 Strand Agent (DeepSeek + PostgreSQL)")
    print("Note: Make sure you have:")
    print("1. PostgreSQL running with pgvector extension")
    print("2. DeepSeek LLM server running on localhost:11434")
    print("3. Documents loaded using load_documents.py")
    print()
    
    while True:
        try:
            query = input("\nYou: ")
            if query.lower() in ['quit', 'exit', 'q']:
                break
            answer = agent.run(query)
            print(f"\nAgent: {answer}")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
