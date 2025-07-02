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
    chunk_field = None
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

        # Identify chunk field
        if field_name.lower() in {"chunk", "text", "content", "document"}:
            chunk_field = field_name

        # Conditions to exclude:
        if not dim and not is_primary and field_name.lower() not in {"id", "chunk", "text", "content", "document"}:
            target_fields.append(field_name)

    # Fetch sample data
    print("\n  🧬 Unique Values (filtered fields):")
    try:
        # Include chunk field in output if it exists
        output_fields = target_fields.copy()
        if chunk_field:
            output_fields.append(chunk_field)
        
        results = client.query(name, output_fields=output_fields, limit=1000)

        # Gather unique values for non-chunk fields
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

        # Print chunk content samples
        if chunk_field and results:
            print(f"\n  📝 Sample Chunk Content ({chunk_field}):")
            for i, row in enumerate(results[:3]):  # Show first 3 chunks
                chunk_content = row.get(chunk_field, "")
                if chunk_content:
                    # Truncate long content for readability
                    if len(chunk_content) > 200:
                        truncated_content = chunk_content[:200] + "..."
                    else:
                        truncated_content = chunk_content
                    
                    print(f"    [{i+1}] {truncated_content}")
                    print()
            
            if len(results) > 3:
                print(f"    ... and {len(results) - 3} more chunks")
        elif not chunk_field:
            print("\n  📝 No chunk/text content field found")

    except Exception as e:
        print(f"    ⚠️ Could not fetch data: {e}")

    print("-" * 60)  # Separator between collections
    print()