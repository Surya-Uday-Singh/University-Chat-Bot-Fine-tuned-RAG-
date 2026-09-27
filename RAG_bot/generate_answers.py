import json
import pandas as pd
from langchain_ollama import OllamaLLM  # <-- Updated import
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ==========================================
# 1. INITIALIZE RAG PIPELINE
# ==========================================
print("Connecting to the knowledge base and waking up Llama 3 ChatQA...")
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

# ==========================================
# 2. RUN THE GENERATION LOOP
# ==========================================
if __name__ == "__main__":
    print("\nLoading test cases from JSONL...")

    test_cases = []
    # Load your golden test cases line by line
    try:
        with open("../golden_test_pairs.jsonl", "r") as file:
            for line in file:
                if line.strip():  # Skip any blank lines
                    test_cases.append(json.loads(line))
    except FileNotFoundError:
        print("Error: 'golden_test_cases.jsonl' not found in the current directory.")
        exit(1)

    results = []
    total_cases = len(test_cases)
    print(f"Generating answers for {total_cases} instructions...\n")

    for idx, item in enumerate(test_cases):
        instruction = item["instruction"]
        golden_answer = item["output"]

        print(f"Processing {idx + 1}/{total_cases}: {instruction[:50]}...")

        # Get answer from the RAG architecture
        rag_ans = rag_chain.invoke(instruction)

        # Store the results, including the golden answer for later evaluation
        results.append({
            "Instruction": instruction,
            "Golden_Answer": golden_answer,
            "RAG_Answer": rag_ans
        })

    # ==========================================
    # 3. SAVE THE RESULTS
    # ==========================================
    df_results = pd.DataFrame(results)
    output_filename = "rag_outputs.csv"
    df_results.to_csv(output_filename, index=False)

    print(f"\nSuccessfully saved all generated answers to {output_filename}")