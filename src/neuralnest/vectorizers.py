from dataclasses import dataclass
from pathlib import Path
from typing import List

from neuralnest.load_models import tokenizer


@dataclass(frozen=True)
class FileMetadata:
    """
    Represents metadata for a file.

    Attributes:
        name (str): Name of the file.
        parent (str): Parent directory of the file.
        stem (str): File name without the suffix.
        suffix (str): File extension.
        file_size (int): Size of the file in bytes.
        created (int): Timestamp of file creation.
        last_modified (int): Timestamp of the last modification.
        last_accessed (int): Timestamp of the last access.
        absolute_path (str): Absolute path to the file.
    """

    name: str
    parent: str
    stem: str
    suffix: str
    file_size: int
    created: int
    last_modified: int
    last_accessed: int
    absolute_path: str


@dataclass(frozen=True)
class VectorizedFile:
    """
    Represents a vectorized file along with its metadata.

    Attributes:
        metadata (FileMetadata): Metadata of the file.
        vectorized_content (List[int]): Vectorized content of the file.
    """

    metadata: FileMetadata
    vectorized_content: List[int]


def vectorize_file(file_path: str) -> VectorizedFile:
    """
    Vectorizes the content of a file and gathers its metadata.

    Args:
        file_path (str): Path to the file to be vectorized.

    Returns:
        VectorizedFile: An object containing the file's vectorized content and metadata.

    Raises:
        FileNotFoundError: If the provided file path does not exist.
        ValueError: If the provided path is a directory.
    """
    # Convert the file path to a Path object
    path = Path(file_path)

    # Ensure the file exists
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Raise an exception if the path is a directory
    if path.is_dir():
        raise ValueError(
            f"The provided path is a directory, expected a file: {file_path}"
        )

    # Gather file metadata
    metadata = FileMetadata(
        name=path.name,
        parent=str(path.parent),
        stem=path.stem,
        suffix=path.suffix,
        file_size=path.stat().st_size,
        created=int(path.stat().st_ctime),
        last_modified=int(path.stat().st_mtime),
        last_accessed=int(path.stat().st_atime),
        absolute_path=str(path.resolve()),
    )

    # Read and vectorize the file content
    content = path.read_text(encoding="utf-8")
    vectorized_content = tokenizer.encode(content, return_tensors="pt")[0].tolist()

    return VectorizedFile(metadata=metadata, vectorized_content=vectorized_content)
