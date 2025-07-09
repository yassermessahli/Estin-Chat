import os
from pymilvus import MilvusClient
from dotenv import load_dotenv

load_dotenv()

client = MilvusClient(uri=os.getenv("MILVUS_HOST"), token="root:Milvus")

def print_record_simple():
    result = client.query(
        collection_name="estin_docs",
        filter="",
        output_fields=["*"],
        limit=1
    )[0]
    
    print("COMPLETE RECORD:")
    print("=" * 60)
    
    for field_name, field_value in result.items():
        if field_name == "vector":
            print(f"{field_name}: [vector with {len(field_value)} dimensions]")
        else:
            print(f"{field_name}: {field_value}")
    
    print("=" * 60)

if __name__ == "__main__":
    print_record_simple()