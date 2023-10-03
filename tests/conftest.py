import pytest
from pyfakefs.fake_filesystem_unittest import Patcher

from neuralnest.vectorizers import VectorizedFile, vectorize_file


@pytest.fixture
def fake_filesystem() -> str:
    with Patcher() as patcher:
        test_file_path = "/test_file.txt"
        patcher.fs.create_file(
            test_file_path, contents="This is a test file for NeuralNest."
        )
        yield test_file_path


@pytest.fixture
def vectorized_test_file(fake_filesystem) -> VectorizedFile:
    yield vectorize_file(fake_filesystem)
