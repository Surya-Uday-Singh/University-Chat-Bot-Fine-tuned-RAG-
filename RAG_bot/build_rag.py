import os
import json
import pickle
from langchain_core.documents import Document

def load_chunked_jsonl(file_path):
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            documents.append(
                Document(
                    page_content=data.get("text", ""),
                    metadata={"source": file_path, **data.get("metadata", {})}
                )
            )
    return documents

def load_qa_jsonl(file_path):
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            question = data.get("instruction", "")
            answer = data.get("output", "")
            combined_text = f"Question: {question}\nAnswer: {answer}"
            documents.append(
                Document(
                    page_content=combined_text,
                    metadata={"source": file_path, "type": "qa_pair"}
                )
            )
    return documents

def build_documents():
    chunk_files = [
        "../train_chunks_Tuitionfee.jsonl",
        "../train_chunks_courses.jsonl",
        "../train_chunks_program_requirements.jsonl",
    ]
    qa_files = [
        "../train.jsonl",
        "../train_tuition.jsonl",
        "../train_program_requirements.jsonl"
    ]

    documents = []
    for path in chunk_files:
        documents.extend(load_chunked_jsonl(path))

    for path in qa_files:
        documents.extend(load_qa_jsonl(path))

    return documents

if __name__ == "__main__":
    pickle_file = "all_documents.pkl"

    if os.path.exists(pickle_file):
        print("Loading combined documents from saved file...")
        with open(pickle_file, "rb") as f:
            all_documents = pickle.load(f)
        print(f"Loaded {len(all_documents)} documents.")
    else:
        print("Loading chunked JSONL files and QA JSONL file...")
        all_documents = build_documents()

        print("Saving combined documents to file...")
        with open(pickle_file, "wb") as f:
            pickle.dump(all_documents, f)
        print(f"Saved {len(all_documents)} documents.")

    print(f"Total documents prepared: {len(all_documents)}")