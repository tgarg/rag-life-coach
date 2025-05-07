from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# --- Configuration ---
embedding_model_name = "all-MiniLM-L6-v2"
chroma_db_path = "chroma_db"

# --- Load the embedding model and Chroma DB ---
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
db = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 5})  # You can change k as needed

# --- Retrieval function ---
def get_relevant_chunks(query: str):
    """
    Given a user query, return a list of relevant journal chunks.
    """
    docs = retriever.get_relevant_documents(query)
    return [doc.page_content for doc in docs]

# Example use:
if __name__ == "__main__":
    query = input("Ask your journal memory something: ")
    results = get_relevant_chunks(query)
    print("\nTop matching journal excerpts:")
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r}\n")