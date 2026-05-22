import json
import os
import time

# ==========================================
# LOAD CONFIG FILE
# ==========================================

with open("config.json", "r") as file:
    config = json.load(file)

# ==========================================
# SET OPENAI API KEY
# ==========================================

os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]

# ==========================================
# IMPORTS
# ==========================================

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS

# ==========================================
# LOAD HUGGINGFACE EMBEDDING MODEL
# MUST BE SAME MODEL USED IN data_prep.py
# ==========================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================================
# LOAD LOCAL FAISS VECTOR DB
# ==========================================

db = FAISS.load_local(
    "faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)

print("FAISS vector DB loaded successfully")

# ==========================================
# LOAD OPENAI LLM
# ==========================================

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

print("OpenAI model loaded successfully")

# ==========================================
# CHAT LOOP
# ==========================================

while True:

    # ==========================================
    # USER INPUT
    # ==========================================

    query = input("\nAsk Question (type 'exit' to quit): ")

    if query.lower() == "exit":
        break

    # ==========================================
    # RETRIEVE RELEVANT DOCUMENTS
    # ==========================================

    retrieved_docs = db.similarity_search(query, k=3)

    context = ""

    print("\nRetrieved Chunks:\n")

    for i, doc in enumerate(retrieved_docs):

        print(f"Chunk {i+1}:")
        print(doc.page_content)
        print("-" * 50)

        context += doc.page_content + "\n"

    # ==========================================
    # CREATE PROMPT
    # ==========================================

    prompt = f"""
    You are a helpful AI assistant.

    Answer the question using ONLY the context below.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    # ==========================================
    # GENERATE RESPONSE
    # ==========================================
    start_time = time.time()
    response = llm.invoke(prompt)
    end_time = time.time()
    print(f"Response generated in {end_time - start_time:.2f} seconds")
    # ==========================================
    # PRINT RESPONSE
    # ==========================================

    print("\nAI Response:\n")

    print(response.content)