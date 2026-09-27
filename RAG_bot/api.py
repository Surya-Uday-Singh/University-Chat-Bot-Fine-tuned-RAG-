import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- 1. API Setup ---
app = FastAPI(title="Algoma University RAG API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

# --- 2. Initialize RAG Pipeline ---
print("Waking up Llama 3 ChatQA and connecting to ChromaDB...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
persist_directory = "./university_chroma_db"

vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = OllamaLLM(model="llama3-chatqa:8b")

system_prompt = (
    "You are a helpful and precise university assistant. Use the following pieces of retrieved context to answer the question. "
    "If the answer is not contained in the context, do not guess or make up information. Simply state that you do not have that information.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}"),
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 3. The Chat Endpoint ---
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Invoke your RAG pipeline with the user's message
        response = rag_chain.invoke(request.message)
        return {"reply": response}
    except Exception as e:
        return {"reply": f"Sorry, a server error occurred: {str(e)}"}

