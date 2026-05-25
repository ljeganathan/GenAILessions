from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

# ==========================================
# STEP 1 - LOAD EMBEDDING MODEL
# ==========================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================================
# LOAD CHROMA VECTOR DB
# ==========================================

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

print("Chroma DB loaded successfully")

# ==========================================
# STEP 3 - LOAD OPEN SOURCE LLM
# ==========================================

# pipe = pipeline(
#     "text-generation",
#     model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     max_new_tokens=200
# )

pipe = pipeline(
    "text-generation",
    model="microsoft/phi-1_5",
    max_new_tokens=50,
    temperature=0.3
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