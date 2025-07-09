from pymilvus import CollectionSchema, FieldSchema, DataType, Function, FunctionType
from client import client
import os

# Build Schema
# Instantiate schema
id_field = FieldSchema(
    name="id",
    dtype=DataType.INT64,
    is_primary=True,
    auto_id=True,
    description="Primary key for the document chunk",
)


schema = CollectionSchema(
    fields=[
        id_field,
    ]
)

# Add fields to schema

schema.add_field(
    field_name="chunk",
    datatype=DataType.VARCHAR,
    max_length=512,
    description="Small piece of text from the original document",
)
schema.add_field(
    field_name="vector",
    datatype=DataType.FLOAT_VECTOR,
    dim=1024,
    description="Vector embedding of the text chunk",
)
schema.add_field(
    field_name="level",
    datatype=DataType.VARCHAR,
    max_length=3,
    description="Academic level (e.g., '1CP', '2CS', '3CS')",
)
schema.add_field(
    field_name="semester",
    datatype=DataType.VARCHAR,
    max_length=2,
    description="Academic semester (e.g., 'S1', 'S2')",
)
schema.add_field(
    field_name="year_of_study",
    datatype=DataType.INT16,
    description="The academic year the document belongs to",
)
schema.add_field(
    field_name="document_type",
    datatype=DataType.VARCHAR,
    max_length=16,
    description="Type of the document (e.g., 'cours', 'td', 'examen')",
)
schema.add_field(
    field_name="page",
    datatype=DataType.INT16,
    description="Page number of the document",
)
schema.add_field(
    field_name="data_type",
    datatype=DataType.VARCHAR,
    max_length=10,
    description="The format of the data ('text', 'table', 'image')",
)

schema.add_field(
    field_name="subject_code",
    datatype=DataType.VARCHAR,
    max_length=128,
    is_partition_key=True,  # partitioning by subject_code
    description="The code of the subject (e.g., 'ELEC', 'BDD', 'AI')",
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
    description="Title of the document",
)


# Add embedding function
embedding_function = Function(
    name="e5-embedding",
    function_type=FunctionType.TEXTEMBEDDING,
    description="Embedding function for text chunks",
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
    "page",
    "data_type",
    "year_of_study",
]:
    index_params.add_index(
        field_name=field,
        index_name=f"{field}_inverted_index",
        index_type="INVERTED",  # Inverted index for categorical fields
    )
    