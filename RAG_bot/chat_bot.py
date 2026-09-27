from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- 1. Load the Database ---
print("Connecting to the knowledge base...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
persist_directory = "./university_chroma_db"

vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- 2. Initialize the Local LLM ---
print("Waking up Llama 3 ChatQA...")
llm = Ollama(model="llama3-chatqa:8b")

# --- 3. Construct the Prompt Template ---
system_prompt = (
    "You are a helpful and precise university assistant. Use the following pieces of retrieved context to answer the question. "
    "If the answer is not contained in the context, do not guess or make up information. Simply state that you do not have that information.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}"),
])


# --- 4. Format the Retrieved Documents ---
# This helper function takes the raw metadata chunks and joins them into plain text
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# --- 5. The Modern LCEL RAG Pipeline ---
# This single, elegant block completely replaces the old 'create_retrieval_chain'
rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

# --- 6. The Chat Loop ---
print("\n" + "=" * 40)
print("🎓 University QA Bot is Online!")
print("Type 'exit' or 'quit' to shut down.")
print("=" * 40 + "\n")

while True:
    user_query = input("You: ")

    if user_query.lower() in ['exit', 'quit']:
        print("Shutting down. Goodbye!")
        break

    if not user_query.strip():
        continue

    print("Bot is searching and thinking...")

    # Notice how simple the invoke call is now!
    response = rag_chain.invoke(user_query)

    print(f"\nBot: {response}\n")
    print("-" * 40)