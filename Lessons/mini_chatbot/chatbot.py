from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

# ==========================================
# STEP 1 - LOAD EMBEDDING MODEL
# ==========================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================================
# STEP 2 - LOAD LOCAL FAISS DB
# ==========================================

db = FAISS.load_local(
    "faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)

print("FAISS DB loaded successfully")

# ==========================================
# STEP 3 - LOAD OPEN SOURCE LLM
# ==========================================

pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=200
)

llm = HuggingFacePipeline(pipeline=pipe)

# ==========================================
# STEP 4 - CHAT LOOP
# ==========================================

while True:

    query = input("\nAsk Question (type 'exit' to quit): ")

    if query.lower() == "exit":
        break

    # ==========================================
    # STEP 5 - RETRIEVE DOCUMENTS
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
    # STEP 6 - CREATE PROMPT
    # ==========================================

    prompt = f"""
    Answer the question using the context below.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    # ==========================================
    # STEP 7 - GENERATE RESPONSE
    # ==========================================

    response = llm.invoke(prompt)

    print("\nAI Response:\n")
    print(response)