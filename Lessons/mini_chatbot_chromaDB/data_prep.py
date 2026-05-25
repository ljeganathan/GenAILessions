from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ==========================================
# STEP 1 - LOAD PDF
# ==========================================

pdf_path = "sample.pdf"

loader = PyPDFLoader(pdf_path)

documents = loader.load()

print(f"Loaded {len(documents)} pages")

# ==========================================
# STEP 2 - SPLIT TEXT
# ==========================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

print(f"Created {len(docs)} chunks")

# ==========================================
# STEP 3 - LOAD EMBEDDING MODEL
# ==========================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================================
# STEP 4 - CREATE CHROMA VECTOR DB
# ==========================================

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

print("Chroma vector DB created successfully")


print("Chroma vector DB saved locally")