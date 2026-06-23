import argparse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
import os
import shutil
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from datetime import datetime

# Modify embedding function for instruct version
class DocumentEmbedder(HuggingFaceEmbeddings):
    def embed_documents(self, texts):
        # Adds 'passage: ' prefix
        prefixed_texts = ["passage: " + text for text in texts]
        return super().embed_documents(prefixed_texts)

# Initialize with proper settings
embedding_function = DocumentEmbedder(
    model_name="intfloat/multilingual-e5-large-instruct",
    model_kwargs={
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
    },
    encode_kwargs={
        'batch_size': 1024, # for embedder
        'normalize_embeddings': True  # Must include for E5 models
    }
)

CHROMA_PATH = "chroma_sample"

BATCH_SIZE = 512 # for database

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the vector database")
    args = parser.parse_args()
    
    if args.reset:
        print("✨ Clearing vector database")
        clear_database()
    
    generate_data_store()

def generate_data_store():
    print("📂 Loading documents...")
    documents = load_documents()
    
    print("✂️ Splitting text into chunks...")
    chunks = split_text(documents)
    
    print("💾 Storing chunks in vector database...")
    batch_and_add_to_chroma(chunks)

def load_documents():
    file_path = "sample_database.csv"

    all_documents = []

    try:
        df = pd.read_csv(file_path)

        documents = []
        for index, row in df.iterrows():
            dt_string = row["datetime"] if pd.notna(row["datetime"]) else None
            timestamp = None

            if dt_string:
                try:
                    # Adjust format
                    dt = datetime.strptime(dt_string, "%Y-%m-%d")
                    timestamp = dt.timestamp()
                except Exception as e:
                    print(f"⚠️ Invalid datetime '{dt_string}' in {file_path} row {index}: {e}")

            metadata = {
                "url": row["url"],
                "title": row["title"],
                "author": row["author"] if pd.notna(row["author"]) else "",
                "website": row["website"],
                "datetime": timestamp if timestamp else 0,  # fallback or filter later
                "section": row["section"] if pd.notna(row["section"]) else "",
                "row_index": index
            }

            documents.append(Document(page_content=row["full-text"], metadata=metadata))

        print(f"📄 Loaded {len(documents)} documents from {file_path}")
        all_documents.extend(documents)

    except Exception as e:
        print(f"❌ Error loading {file_path}: {str(e)}")

    print(f"📚 Total documents loaded: {len(all_documents)}")
    return all_documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Due to embedding model limits
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✂️ Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks

def batch_and_add_to_chroma(chunks: list[Document]):
    chunks_with_ids = calculate_chunk_ids(chunks)
    
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )
    
    existing_items = db.get(include=["metadatas"])
    existing_ids = {item.get("id") for item in existing_items["metadatas"] if "id" in item}
    
    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]
    
    if not new_chunks:
        print("✅ No new documents to add")
        return
    
    print(f"🆕 Adding {len(new_chunks)} new chunks to database")
    
    for i in range(0, len(new_chunks), BATCH_SIZE):
        batch = new_chunks[i:i + BATCH_SIZE]
        batch_ids = [chunk.metadata["id"] for chunk in batch]
        
        # Show progress for each batch
        print(f"🔧 Processing batch {i//BATCH_SIZE + 1}/{(len(new_chunks)-1)//BATCH_SIZE + 1}")
        db.add_documents(
            documents=batch,
            ids=batch_ids
        )
        print(f"✅ Added batch {i//BATCH_SIZE + 1}/{(len(new_chunks)-1)//BATCH_SIZE + 1}")

def calculate_chunk_ids(chunks):
    last_row_id = None
    current_chunk_index = 0
    
    for chunk in chunks:
        url = chunk.metadata.get("url", "no_url")
        row_idx = chunk.metadata.get("row_index", 0)
        
        current_row_id = f"{url}:{row_idx}"
        
        if current_row_id == last_row_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        
        chunk.metadata["id"] = f"{current_row_id}:{current_chunk_index}"
        last_row_id = current_row_id
    
    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("🧹 Database cleared successfully")

if __name__ == "__main__":
    main()
