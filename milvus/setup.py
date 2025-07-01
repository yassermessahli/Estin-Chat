from pymilvus import MilvusClient, DataType, Function, FunctionType
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Setup Connection
# Connect to Milvus server as root
client = MilvusClient(uri=os.getenv("MILVUS_HOST"), token="root:Milvus")


# Check if collection exists and delete it
collection_name = "estin_docs"

if client.has_collection(collection_name=collection_name):
    print(f"Found existing collection with name '{collection_name}', deleting it...")
    client.drop_collection(collection_name=collection_name)
    print(f"Collection '{collection_name}' deleted successfully!")


# Build Schema
# Instantiate schema
schema = MilvusClient.create_schema()

# Add fields to schema
schema.add_field(
    field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True
)
schema.add_field(field_name="chunk", datatype=DataType.VARCHAR, max_length=512)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
schema.add_field(field_name="level", datatype=DataType.VARCHAR, max_length=3)
schema.add_field(field_name="semester", datatype=DataType.VARCHAR, max_length=2)
schema.add_field(field_name="year_of_study", datatype=DataType.INT16)
schema.add_field(field_name="document_type", datatype=DataType.VARCHAR, max_length=16)
schema.add_field(field_name="data_type", datatype=DataType.VARCHAR, max_length=10)

schema.add_field(
    field_name="subject_code",
    datatype=DataType.VARCHAR,
    max_length=128,
    is_partition_key=True,  # partitioning by subject_code
)

schema.add_field(
    field_name="title",
    datatype=DataType.VARCHAR,
    max_length=256,
    enable_analyzer=True,
    analyzer_params={
        "type": "english",  # Built-in for english content
    },
    enable_match=True,
)


# Add embedding function
embedding_function = Function(
    name="e5-embedding",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["chunk"],
    output_field_names=["vector"],
    params={
        "provider": "TEI",
        "endpoint": os.getenv("EMBEDDING_ENDPOINT"),  # container endpoint
    },
)
schema.add_function(embedding_function)

# Indexing
# Prepare index parameters
index_params = client.prepare_index_params()


index_params.add_index(
    field_name="title", index_type="AUTOINDEX"
)  # Auto index for title field

index_params.add_index(
    field_name="vector",
    index_name="dense_vector_index",
    index_type="AUTOINDEX",  # Auto index for vector field
    metric_type="COSINE",  # cosine score for similarity search
)

for field in [
    "level",
    "semester",
    "subject_code",
    "document_type",
    "data_type",
    "year_of_study",
]:
    index_params.add_index(
        field_name=field,
        index_name=f"{field}_inverted_index",
        index_type="INVERTED",  # Inverted index for categorical fields
    )


# Create Collection
print(f"Creating new collection '{collection_name}' with given params...")
client.create_collection(
    collection_name=collection_name,
    schema=schema,
    auto_id=True,
    enable_dynamic_field=True,
    index_params=index_params,
    num_partitions=128,  # balanced distribution of subject_codes across partitions
    properties={"partitionkey.isolation": True},  # isolation for partition keys
)

# Verify collection
if client.has_collection(collection_name=collection_name):
    print(f"Collection '{collection_name}' created successfully!")
    collection_stats = client.get_collection_stats(collection_name=collection_name)
    print(f"Collection stats: {collection_stats}")
else:
    print(f"Failed to create collection '{collection_name}'. Please check the logs.")
