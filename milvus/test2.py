from pymilvus import MilvusClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MILVUS_HOST = os.getenv("MILVUS_HOST", "http://localhost:19530")

# Connect to Milvus
client = MilvusClient(uri=MILVUS_HOST, token="root:Milvus")

# List all collections
collections = client.list_collections()
print(f"📚 Found {len(collections)} collection(s):\n")

for name in collections:
    print(f"🔎 Collection: {name}")

    # Get collection schema
    schema = client.describe_collection(name)

    print("  Fields:")
    for field in schema["fields"]:
        field_name = field.get("name")
        field_type = field.get("type")
        is_primary = field.get("is_primary", False)
        auto_id = field.get("auto_id", False)
        dim = field.get("params", {}).get("dim")

        print(f"    • {field_name} ({field_type})", end="")
        if dim:
            print(f" [dim={dim}]", end="")
        if is_primary:
            print(" [Primary Key]", end="")
        if auto_id:
            print(" [Auto ID]", end="")
        print()

    print()
