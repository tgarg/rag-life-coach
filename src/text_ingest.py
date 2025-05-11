import os
import time
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# --- Step 0: Print a message to the console ---
start_time = time.time()
print("Starting text ingestion...")

# --- Step 1: Load all .txt or .md files from the journals/ directory ---
journal_dir = "journals"
all_files = [os.path.join(journal_dir, f) for f in os.listdir(journal_dir) if f.endswith(('.txt', '.md'))]

if not all_files:
    raise ValueError("No journal files found in ./journals. Add some .txt or .md files first.")

# Use LangChain's TextLoader to load content
docs = []
for file_path in all_files:
    loader = TextLoader(file_path, encoding="utf-8")
    docs.extend(loader.load())

print(f"Loaded {len(docs)} journal documents.")

# --- Step 2: Split documents into 500-character chunks with 50-character overlap ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,  # Increased from 500
    chunk_overlap=150,  # Increased from 50
    separators=["\n\n", "\n", ". ", " ", ""]  # Prioritize paragraph breaks
)
chunks = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks.")

# --- Step 3: Embed using HuggingFace sentence transformer ---
# We are using the 'all-MiniLM-L6-v2' model for balance of performance + quality
# It is small (80MB), fast, and works well for semantic search tasks.
embedding_model_name = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# --- Step 4: Store embeddings in a persistent Chroma vector database ---
chroma_db_path = "chroma_db"
db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=chroma_db_path)
db.persist()

print(f"✅ Successfully created and stored embeddings in {chroma_db_path}/")

# Print total execution time
total_time = time.time() - start_time
print(f"\nTotal execution time: {total_time:.2f} seconds")
print("Text ingestion complete.")
