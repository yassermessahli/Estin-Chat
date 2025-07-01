import os
from pymilvus import MilvusClient
from dotenv import load_dotenv

load_dotenv()

client = MilvusClient(uri=os.getenv("MILVUS_HOST"), token="root:Milvus")

collection_name = "estin_docs"

def verify_and_load_collection():
    try:
        # Load the collection to make data searchable
        client.load_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' loaded successfully.")
        
        # Wait a moment for the load to complete
        import time
        time.sleep(2)
        
        # Get updated collection stats
        collection_stats = client.get_collection_stats(collection_name=collection_name)
        print(f"Updated collection stats: {collection_stats}")
        
        # Test a simple query to verify data is accessible
        results = client.query(
            collection_name=collection_name,
            filter="",  # Empty filter to get all records
            output_fields=["chunk", "level", "subject_code"],
            limit=5
        )
        
        print(f"Sample query returned {len(results)} results:")
        for i, result in enumerate(results[:3]):
            print(f"  {i+1}. Level: {result.get('level')}, Subject: {result.get('subject_code')}")
            print(f"     Chunk: {result.get('chunk')[:100]}...")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    verify_and_load_collection()

# Query by specific criteria
results = client.query(
    collection_name="estin_docs",
    filter="level == '1CP' AND subject_code == 'ELEC'",
    output_fields=["chunk", "title", "document_type"],
    limit=10
)

print(f"Query returned {len(results)} results:")

# Get total count by querying all records
results = client.query(
    collection_name="estin_docs",
    filter="",
    output_fields=["id"],
    limit=16384  
)
print(f"Total records found: {len(results)}")