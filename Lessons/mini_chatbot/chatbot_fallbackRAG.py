import time
import requests

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

# ==========================================
# CONFIGURATION
# ==========================================

THRESHOLD = 0.7

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
# STEP 4 - EXTERNAL SEARCH FUNCTION
# ==========================================

def external_search(query):

    try:

        url = f"https://api.duckduckgo.com/?q={query}&format=json"

        response = requests.get(url)

        data = response.json()

        if data.get("AbstractText"):

            return data["AbstractText"]

        else:

            return "No external information found."

    except Exception as e:

        return f"External search failed: {str(e)}"

# ==========================================
# STEP 5 - CHAT LOOP
# ==========================================

while True:

    query = input("\nAsk Question (type 'exit' to quit): ")

    if query.lower() == "exit":
        break

    total_start = time.time()

    # ==========================================
    # STEP 6 - SEARCH VECTOR DB WITH SCORES
    # ==========================================

    retrieval_start = time.time()

    docs_with_score = db.similarity_search_with_score(
        query,
        k=3
    )

    retrieval_end = time.time()

    best_score = docs_with_score[0][1]

    print(f"\nBest Similarity Score: {best_score:.4f}")

    # ==========================================
    # STEP 7 - INTERNAL RAG
    # ==========================================

    if best_score < THRESHOLD:

        print("\nSource Type: INTERNAL VECTOR DB")

        context = ""

        print("\nRetrieved Chunks:\n")

        for i, (doc, score) in enumerate(docs_with_score):

            print(f"\nChunk {i+1}")
            print(f"Score: {score:.4f}")
            print(doc.page_content)
            print("-" * 60)

            context += doc.page_content + "\n"

        prompt = f"""
You are an AI assistant.

Answer ONLY using the context below.

If answer is not available in context,
say:
"I could not find the answer in the document."

Context:
{context}

Question:
{query}

Answer:
"""

        llm_start = time.time()

        response = llm.invoke(prompt)

        llm_end = time.time()

        print("\nAI Response:\n")

        print(response)

        print("\nAnswer Source: Internal PDF Documents")

    # ==========================================
    # STEP 8 - EXTERNAL FALLBACK
    # ==========================================

    else:

        print("\nSource Type: EXTERNAL SEARCH")

        llm_start = time.time()

        external_result = external_search(query)

        llm_end = time.time()

        print("\nAI Response:\n")

        print(external_result)

        print("\nAnswer Source: External Web Search")

    total_end = time.time()

    # ==========================================
    # STEP 9 - PERFORMANCE METRICS
    # ==========================================

    print("\nPerformance Metrics:")

    print(f"Retrieval Time : {(retrieval_end - retrieval_start):.2f} sec")

    print(f"LLM/Search Time: {(llm_end - llm_start):.2f} sec")

    print(f"Total Time     : {(total_end - total_start):.2f} sec")