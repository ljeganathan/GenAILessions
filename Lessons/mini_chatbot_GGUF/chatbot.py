from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from ctransformers import AutoModelForCausalLM
import time

# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================================
# LOAD FAISS DB
# ==========================================

db = FAISS.load_local(
    "faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)

print("FAISS DB loaded successfully")

# ==========================================
# LOAD GGUF MODEL
# ==========================================
#https://huggingface.co/hieupt/TinyLlama-1.1B-Chat-v1.0-Q4_K_M-GGUF/blob/main/tinyllama-1.1b-chat-v1.0-q4_k_m.gguf
llm = AutoModelForCausalLM.from_pretrained(
    "models",
    model_file="tinyllama-1.1b-chat-v1.0-q4_k_m.gguf",
    model_type="llama",
    gpu_layers=0
)

print("TinyLlama GGUF model loaded")

# ==========================================
# CHAT LOOP
# ==========================================

while True:

    query = input("\nAsk Question (type 'exit' to quit): ")

    if query.lower() == "exit":
        break

    # ==========================================
    # RETRIEVE DOCUMENTS
    # ==========================================

    retrieved_docs = db.similarity_search(query, k=3)

    context = ""

    for doc in retrieved_docs:
        context += doc.page_content + "\n"
    print(context)
    # ==========================================
    # CREATE PROMPT
    # ==========================================

    prompt = f"""
    You are a helpful AI assistant.

    Answer the question based only on the context below.

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
    response = llm(
        prompt,
        max_new_tokens=200,
        temperature=0.5
    )
    end_time = time.time()
    print(f"Response generated in {end_time - start_time:.2f} seconds")

    print("\nAI Response:\n")
    print(response)