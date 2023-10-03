from unittest.mock import Mock, patch

import pytest
from neuralnest.load_models import vector_dimension
from neuralnest.vector_search import (
    create_milvus_collection_from_dataclass,
    insert_vectors,
    search_vector,
)
from neuralnest.vectorizers import FileMetadata

COLLECTION_NAME = "test_collection"


# Mocking the Milvus client
@pytest.fixture
def mock_client() -> Mock:
    with patch("neuralnest.vector_search.client") as mock:
        yield mock


def test_create_milvus_collection_from_dataclass(mock_client, vectorized_test_file):
    # Set return_value before calling the function
    mock_client.has_collection.return_value = False

    create_milvus_collection_from_dataclass(
        COLLECTION_NAME, FileMetadata, vector_dimension
    )

    # Check that has_collection was called with the right argument
    mock_client.has_collection.assert_called_once_with(COLLECTION_NAME)
    # Check that create_collection was called if has_collection returns False
    mock_client.create_collection.assert_called_once()


def test_insert_vectors(mock_client, vectorized_test_file):
    insert_vectors(COLLECTION_NAME, [vectorized_test_file])

    # Ensure that client.insert is called with appropriate parameters
    mock_client.insert.assert_called_once_with(
        COLLECTION_NAME,
        records=[vectorized_test_file.vectorized_content],
    )


def test_search_vector(mock_client, vectorized_test_file):
    vector = vectorized_test_file.vectorized_content
    top_k = 5

    search_vector(COLLECTION_NAME, vector, top_k)

    # Ensure that client.search is called with appropriate parameters
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    mock_client.search.assert_called_once_with(
        COLLECTION_NAME,
        [vector],
        top_k,
        params=search_params,
    )
