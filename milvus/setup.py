from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)


def main():
    """
    Initializes Milvus, creates the necessary collections with their schemas
    and indexes, and loads them into memory.
    """
    try:
        # These can be moved to environment variables or a config file
        MILVUS_HOST = "localhost"
        MILVUS_PORT = "19530"
        COLLECTION_NAMES = [
            "texts_collection",
            "tables_collection",
            "images_collection",
        ]
        VECTOR_DIM = 384  # Example dimension

        # 1. Connect to Milvus
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

        # 2. Create collections
        collection_schemas = {
            "texts_collection": [
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            ],
            "tables_collection": [
                FieldSchema(
                    name="table_description", dtype=DataType.VARCHAR, max_length=65535
                ),
            ],
            "images_collection": [
                FieldSchema(
                    name="image_description", dtype=DataType.VARCHAR, max_length=65535
                ),
            ],
        }

        common_fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
            FieldSchema(name="level", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="semester", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="subject", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="year", dtype=DataType.INT64),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=65535),
        ]

        for name, specific_fields in collection_schemas.items():
            if not utility.has_collection(name):
                schema = CollectionSchema(
                    common_fields + specific_fields, f"Collection for {name}"
                )
                collection = Collection(name, schema)

                index_params = {
                    "metric_type": "L2",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128},
                }
                collection.create_index("vector", index_params)

        # 3. Load collections into memory
        for name in COLLECTION_NAMES:
            if utility.has_collection(name):
                collection = Collection(name)
                collection.load()

    except Exception as e:
        print(f"An error occurred during Milvus setup: {e}")


if __name__ == "__main__":
    main()
