import os
import json
from pymilvus import MilvusClient
from dotenv import load_dotenv
import time
from tqdm import tqdm

load_dotenv()

client = MilvusClient(uri=os.getenv("MILVUS_HOST"), token="root:Milvus")
collection_name = "estin_docs"
data_folder = "/home/melissa-ghemari/estin-chatbot/data-pipeline/pipeline/test_outputs/chunked"

def truncate_to_bytes(text, max_bytes):
    """Truncate text to fit within max_bytes when encoded as UTF-8"""
    if not text:
        return ""
    
    text_str = str(text)
    encoded = text_str.encode('utf-8')
    
    if len(encoded) <= max_bytes:
        return text_str
    
    for i in range(max_bytes, 0, -1):
        try:
            return encoded[:i].decode('utf-8')
        except UnicodeDecodeError:
            continue
    
    return ""

def load_chunks_to_milvus_optimized(data_folder, collection_name):
    # Create dummy vector once (reuse for memory efficiency)
    dummy_vector = [0.0] * 768
    
    batch_size = 150
    total_inserted = 0
    current_batch = []
    
    json_files = [f for f in os.listdir(data_folder) if f.endswith(".json")]
    
    print(f"Found {len(json_files)} JSON files to process")
    
    for file_name in tqdm(json_files, desc="Processing files"):
        file_path = os.path.join(data_folder, file_name)
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
                
                for chunk in chunks:
                    row = {
                        "chunk": truncate_to_bytes(chunk["content"], 512),
                        #"vector": dummy_vector, #comment this line when embedding function is enabled
                        "level": truncate_to_bytes(chunk["metadata"].get("level", "L1"), 3),
                        "semester": truncate_to_bytes(chunk["metadata"].get("semester", "S1"), 2),
                        "year_of_study": int(chunk["metadata"].get("year", 2025)),
                        "document_type": truncate_to_bytes(chunk["metadata"].get("type", "lecture"), 16),
                        "data_type": truncate_to_bytes(chunk["metadata"].get("content_type", "text"), 10),
                        "subject_code": truncate_to_bytes(chunk["metadata"].get("module", "UNKNOWN"), 128),
                        "title": truncate_to_bytes(chunk["metadata"].get("original_filename", "untitled"), 256),
                    }
                    current_batch.append(row)
                    
                    # Insert when batch is full
                    if len(current_batch) >= batch_size:
                        success = insert_batch(current_batch, collection_name, total_inserted)
                        if success:
                            total_inserted += len(current_batch)
                        current_batch = []  # Clear batch
                        
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
            continue
    
    # Insert remaining batch
    if current_batch:
        success = insert_batch(current_batch, collection_name, total_inserted)
        if success:
            total_inserted += len(current_batch)
    
    print(f"Successfully inserted {total_inserted} chunks into collection '{collection_name}'.")
    
    # Flush and verify
    flush_and_verify(collection_name)

def insert_batch(batch, collection_name, current_total):
    """Insert a batch with retry logic"""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            client.insert(collection_name=collection_name, data=batch)
            batch_num = (current_total // len(batch)) + 1
            print(f"✓ Inserted batch {batch_num}: {len(batch)} chunks (Total: {current_total + len(batch)})")
            return True
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠ Retry {attempt + 1}/{max_retries} for batch (Error: {e})")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"✗ Failed to insert batch after {max_retries} attempts: {e}")
                return False
    
    return False

def flush_and_verify(collection_name):
    """Flush collection and verify data"""
    try:
        print("Flushing collection...")
        client.flush(collection_name=collection_name)
        
        print("Loading collection...")
        client.load_collection(collection_name=collection_name)
        time.sleep(3)
        
        # Verify
        collection_stats = client.get_collection_stats(collection_name=collection_name)
        print(f"Final collection stats: {collection_stats}")
        
        # Count actual records
        results = client.query(
            collection_name=collection_name,
            filter="",
            output_fields=["id"],
            limit=16384
        )
        print(f"Verified: {len(results)} records are queryable")
        
    except Exception as e:
        print(f"Error in flush/verify: {e}")

if __name__ == "__main__":
    if not client.has_collection(collection_name=collection_name):
        print(f"Collection '{collection_name}' does not exist. Please run setup.py first.")
        exit(1)
    
    print(f"Collection '{collection_name}' found. Starting optimized data loading...")
    load_chunks_to_milvus_optimized(data_folder, collection_name)