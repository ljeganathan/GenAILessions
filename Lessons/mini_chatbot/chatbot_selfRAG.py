# Simple Self-RAG You Can Build
# 1. Retrieve documents
# 2. Generate answer
# 3. Ask LLM:
#    "Is this answer supported by context?"
# 4. If NO:
#    - retrieve more chunks
#    - regenerate answer
# 5. Final output

import time
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
# STEP 2 - LOAD FAISS VECTOR DB
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
    model="microsoft/phi-1_5",
    max_new_tokens=80,
    temperature=0.2,
    repetition_penalty=1.1
)

llm = HuggingFacePipeline(pipeline=pipe)

print("LLM loaded successfully")

# ==========================================
# USER QUERY
# ==========================================

query = input("\nAsk Question: ")

# ==========================================
# TOTAL TIMER START
# ==========================================

total_start = time.time()

# ==========================================
# FIRST RETRIEVAL
# ==========================================

retrieval_start = time.time()

docs = db.similarity_search(query, k=3)

context = "\n".join([doc.page_content for doc in docs])

retrieval_end = time.time()

# ==========================================
# FIRST ANSWER GENERATION
# ==========================================

generation_start = time.time()

prompt = f"""
Answer using the context below.

Context:
{context}

Question:
{query}

Answer:
"""

answer = llm.invoke(prompt)

generation_end = time.time()

print("\nFirst Answer:\n")
print(answer)

# ==========================================
# SELF EVALUATION
# ==========================================

evaluation_start = time.time()

evaluation_prompt = f"""
Question:
{query}

Context:
{context}

Answer:
{answer}

Is the answer fully supported by the context?

Reply ONLY:
YES
or
NO
"""

evaluation = llm.invoke(evaluation_prompt)

evaluation_end = time.time()

print("\nSelf Evaluation:")
print(evaluation)

# ==========================================
# IF ANSWER NOT GOOD
# ==========================================

if "NO" in str(evaluation).upper():

    print("\nAnswer not reliable.")
    print("Retrieving more context...\n")

    # ==========================================
    # SECOND RETRIEVAL
    # ==========================================

    retry_retrieval_start = time.time()

    docs = db.similarity_search(query, k=6)

    new_context = "\n".join([doc.page_content for doc in docs])

    retry_retrieval_end = time.time()

    # ==========================================
    # IMPROVED ANSWER
    # ==========================================

    retry_generation_start = time.time()

    improved_prompt = f"""
You are an AI assistant.

Use the expanded context below.

Context:
{new_context}

Question:
{query}

Provide a corrected and complete answer.
"""

    improved_answer = llm.invoke(improved_prompt)

    retry_generation_end = time.time()

    print("\nImproved Answer:\n")
    print(improved_answer)

else:

    print("\nAnswer validated successfully.")

# ==========================================
# TOTAL TIMER END
# ==========================================

total_end = time.time()

# ==========================================
# PERFORMANCE METRICS
# ==========================================

print("\nPerformance Metrics:")

print(
    f"Initial Retrieval Time : "
    f"{(retrieval_end - retrieval_start):.2f} sec"
)

print(
    f"First Generation Time  : "
    f"{(generation_end - generation_start):.2f} sec"
)

print(
    f"Evaluation Time        : "
    f"{(evaluation_end - evaluation_start):.2f} sec"
)

if "NO" in str(evaluation).upper():

    print(
        f"Retry Retrieval Time   : "
        f"{(retry_retrieval_end - retry_retrieval_start):.2f} sec"
    )

    print(
        f"Retry Generation Time  : "
        f"{(retry_generation_end - retry_generation_start):.2f} sec"
    )

print(
    f"Total Execution Time   : "
    f"{(total_end - total_start):.2f} sec"
)