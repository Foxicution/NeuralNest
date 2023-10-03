import pytest

from neuralnest.vectorizers import VectorizedFile, vectorize_file


def test_vectorize_non_existent_file():
    with pytest.raises(FileNotFoundError):
        vectorize_file("non_existent_file.txt")


def test_vectorize_directory():
    with pytest.raises(ValueError):
        vectorize_file(".")


def test_vectorize_file(fake_filesystem):
    result = vectorize_file(fake_filesystem)
    assert isinstance(result, VectorizedFile)
    assert result.metadata.name == "test_file.txt"
    assert len(result.vectorized_content) > 0
