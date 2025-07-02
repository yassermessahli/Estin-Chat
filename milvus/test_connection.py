from pymilvus import connections

try:
    # Connect to Milvus
    connections.connect("default", host="localhost", port="19530")
    print("✅ Connected to Milvus successfully!")

except Exception as e:
    print("❌ Failed to connect to Milvus:", e)
