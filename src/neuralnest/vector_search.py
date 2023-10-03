from dataclasses import asdict, fields
from typing import List, Type

from neuralnest.config import MILVUS_HOST, MILVUS_PORT
from neuralnest.vectorizers import FileMetadata, VectorizedFile
from pymilvus import CollectionSchema, DataType, FieldSchema, Milvus

TYPE_MAPPING = {str: DataType.STRING, int: DataType.INT64, float: DataType.FLOAT}

# Connect to Milvus server
client = Milvus(host=MILVUS_HOST, port=MILVUS_PORT)


# Create a collection in Milvus to store the vectors
def create_milvus_collection_from_dataclass(
    collection_name: str, dataclass_type: Type[FileMetadata], vector_dim: int
) -> Milvus:
    # Extract fields from the dataclass
    dataclass_fields = fields(dataclass_type)

    milvus_fields = []

    # Add an 'id' field for the primary key
    milvus_fields.append(
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
    )

    # Convert dataclass fields to Milvus fields
    for field in dataclass_fields:
        milvus_dtype = TYPE_MAPPING.get(field.type)
        if milvus_dtype:
            milvus_fields.append(FieldSchema(name=field.name, dtype=milvus_dtype))

    # Add the vector field
    milvus_fields.append(
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim)
    )

    # Create the collection schema
    schema = CollectionSchema(
        fields=milvus_fields, description=f"Collection for {dataclass_type.__name__}"
    )

    # Create the collection in Milvus
    if not client.has_collection(collection_name):
        client.create_collection(collection_name, schema=schema)

    return client


# Insert vectors into the collection
def insert_vectors(collection_name: str, vectorized_files: List[VectorizedFile]):
    # Generate the metadata insertion data dynamically and add vectors
    data_to_insert = {
        **{
            field.name: [asdict(f.metadata)[field.name] for f in vectorized_files]
            for field in fields(FileMetadata)
        },
        "vector": [f.vectorized_content for f in vectorized_files],
    }

    if client.has_collection(collection_name):
        client.insert(collection_name, data_to_insert)
    else:
        raise ValueError(f"Collection {collection_name} does not exist.")


# Search for similar vectors in the collection
def search_vector(collection_name: str, vector: List[int], top_k: int = 5):
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = client.search(collection_name, [vector], top_k, params=search_params)
    return results


# Example usage:
# vectorized_file = vectorize_file('path_to_file.txt')
# insert_vectors([vectorized_file])
# similar_files = search_vector(vectorized_file.vectorized_content)
