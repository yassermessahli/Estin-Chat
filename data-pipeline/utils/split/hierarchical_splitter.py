from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from typing import List


class HierarchicalSplitter:
    def __init__(
        self,
        chunk_size: int = 300,  # In characters
        chunk_overlap: int = 60,  # In characters
    ):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._splitter = self._create_splitter()

    def _create_splitter(self) -> TextSplitter:
        """Creates the underlying RecursiveCharacterTextSplitter."""
        return RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
            keep_separator=False,
        )

    def split_text(self, text: str) -> List[str]:
        """
        Splits the given text into chunks.
        """
        return self._splitter.split_text(text)
