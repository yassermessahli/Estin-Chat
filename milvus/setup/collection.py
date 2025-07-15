from .client import client
from .schema import schema, index_params

# Check if collection exists and delete it
collection_name = "estin_docs"

if client.has_collection(collection_name=collection_name):
    print(f"Found existing collection with name '{collection_name}', deleting it...")
    client.drop_collection(collection_name=collection_name)
    print(f"Collection '{collection_name}' deleted successfully!")


# Create Collection
print(f"Creating new collection '{collection_name}' with given params...")
client.create_collection(
    collection_name=collection_name,
    schema=schema,
    auto_id=True,
    enable_dynamic_field=True,
    index_params=index_params,
    num_partitions=128,
)

# Verify collection
if client.has_collection(collection_name=collection_name):
    print(f"Collection '{collection_name}' created successfully!")
    collection_stats = client.get_collection_stats(collection_name=collection_name)
    print(f"Collection stats: {collection_stats}")
else:
    print(f"Failed to create collection '{collection_name}'. Please check the logs.")
