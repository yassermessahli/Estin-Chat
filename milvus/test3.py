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
    target_fields = []
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

        # Conditions to exclude:
        if not dim and not is_primary and field_name.lower() not in {"id", "chunk"}:
            target_fields.append(field_name)

    # Fetch sample data
    print("\n  🧬 Unique Values (filtered fields):")
    try:
        results = client.query(name, output_fields=target_fields, limit=1000)

        # Gather unique values
        field_values = {field: set() for field in target_fields}
        for row in results:
            for field in target_fields:
                value = row.get(field)
                field_values[field].add(value)

        for field, values in field_values.items():
            sorted_vals = sorted(values)
            print(f"    - {field}: {sorted_vals[:10]}", end="")
            if len(sorted_vals) > 10:
                print(f" ... (+{len(sorted_vals) - 10} more)")
            else:
                print()
    except Exception as e:
        print(f"    ⚠️ Could not fetch data: {e}")

    print()
