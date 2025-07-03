import os
from pymilvus import MilvusClient
from dotenv import load_dotenv

load_dotenv()

client = MilvusClient(uri=os.getenv("MILVUS_HOST"), token="root:Milvus")
collection_name = "estin_docs"

def check_vectors():
    # Query a few records and include the vector field
    results = client.query(
        collection_name=collection_name,
        filter="",
        output_fields=["id", "chunk", "vector"],
        limit=3
    )
    
    print(f"Checking vectors for {len(results)} records:\n")
    
    for i, result in enumerate(results):
        print(f"Record {i+1}:")
        print(f"  ID: {result['id']}")
        print(f"  Chunk: {result['chunk'][:50]}...")
        
        vector = result['vector']
        print(f"  Vector length: {len(vector)}")
        print(f"  Vector sample (first 5 dims): {vector[:5]}")
        print(f"  Vector range: min={min(vector):.6f}, max={max(vector):.6f}")
        print()

if __name__ == "__main__":
    check_vectors()