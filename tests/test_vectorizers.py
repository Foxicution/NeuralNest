import pytest
from pyfakefs.fake_filesystem_unittest import Patcher

from neuralnest.vectorizers import VectorizedFile, vectorize_file


@pytest.fixture
def fake_filesystem():
    with Patcher() as patcher:
        # Create a fake file in the virtual file system
        test_file_path = "/test_file.txt"
        patcher.fs.create_file(
            test_file_path, contents="This is a test file for NeuralNest."
        )
        yield test_file_path


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
