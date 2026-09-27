import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load the prepared documents from the pickle file
pickle_file = "all_documents.pkl"
print(f"Loading documents from {pickle_file}...")

with open(pickle_file, "rb") as f:
    all_documents = pickle.load(f)

print(f"Successfully loaded {len(all_documents)} documents for the university QA bot.")

# 2. Initialize the embedding model
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Create the Vector Database
persist_directory = "./university_chroma_db"
print("Embedding documents and building ChromaDB...")

# Pass the newly loaded 'all_documents' list right in
vectorstore = Chroma.from_documents(
    documents=all_documents,
    embedding=embeddings,
    persist_directory=persist_directory
)

print(f"Success! Vector database saved to: {persist_directory}")