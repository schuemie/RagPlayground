from pathlib import Path
from typing import Iterator, Union, List, Optional, Dict
from collections import defaultdict
import re

import pypdf
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.pdf import BasePDFLoader
from langchain_core.documents import Document
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders.parsers.pdf import PyPDFParser


def _remove_numbers_start_end(text: str) -> str:
    text = re.sub(r"^\d+\s*|\s*\d+$", "", text)
    text = re.sub(r"\n^\d+\s*|\s*\d+\n", "\n", text)
    text = re.sub("  +", " ", text)
    return text


def _detect_repetitive_lines(pages: list[str]) -> tuple[str, str]:
    """Detects repetitive first and last 1 or 2 lines across pages, excluding page numbers."""
    first_lines = defaultdict(int)
    last_lines = defaultdict(int)

    # Count occurrences of first and last 1 or 2 lines, ignoring likely page numbers
    for page in pages:
        lines = page.strip().split('\n')
        lines = [_remove_numbers_start_end(line) for line in lines]

        if len(lines) > 2:  # Ensure there's more than just a header/footer candidate
            first_lines['\n'.join(lines[:1])] += 1
            first_lines['\n'.join(lines[:2])] += 2
            last_lines['\n'.join(lines[-1:])] += 1
            last_lines['\n'.join(lines[-2:])] += 2

    # Identify the most common first and last lines, excluding page numbers
    common_first = max(first_lines, key=first_lines.get) if first_lines else ""
    common_last = max(last_lines, key=last_lines.get) if last_lines else ""

    return common_first, common_last


def _remove_repetitive_lines(text: str, common_first: str, common_last: str) -> str:
    """Removes repetitive first and last lines from a page's text."""
    text_without_page_numbers = _remove_numbers_start_end(text)
    n_first_lines = len(common_first.strip().split('\n'))
    n_last_lines = len(common_last.strip().split('\n'))
    lines = text.strip().split('\n')
    if text_without_page_numbers.startswith(common_first):
        lines = lines[n_first_lines:]
    if text_without_page_numbers.endswith(common_last):
        lines = lines[:-n_last_lines]
    return "\n".join(lines)


class CustomPDFParser(PyPDFParser):
    """
    Load `PDF` using `pypdf` but preserving layout and removing headers and footers.
    """

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        with blob.as_bytes_io() as pdf_file_obj:
            reader = pypdf.PdfReader(pdf_file_obj, password=self.password)
            pages_text = [reader.pages[i].extract_text(extraction_mode="layout") for i in range(len(reader.pages))]
            # pages_text = [reader.pages[i].extract_text() for i in range(len(reader.pages))]
            common_first, common_last = _detect_repetitive_lines(pages_text)
            pages_text = [_remove_repetitive_lines(page, common_first, common_last) for page in pages_text]
            yield from [
                Document(
                    page_content=pages_text[page_number]
                    + self._extract_images_from_page(page),
                    metadata={"source": blob.source, "page": page_number},
                )
                for page_number, page in enumerate(reader.pages)
            ]


class CustomPDFLoader(BasePDFLoader):
    """Load PDF using pypdf into list of documents.

    Loader chunks by page and stores page numbers in metadata.

    Based on `langchain_community.document_loaders.pdf.PyPDFLoader`.
    """

    def load(self) -> List[Document]:
        return list(self.lazy_load())

    def __init__(
        self,
        file_path: str,
        password: Optional[Union[str, bytes]] = None,
        headers: Optional[Dict] = None,
        extract_images: bool = False,
    ) -> None:
        """Initialize with a file path."""
        try:
            import pypdf  # noqa:F401
        except ImportError:
            raise ImportError(
                "pypdf package not found, please install it with " "`pip install pypdf`"
            )
        super().__init__(file_path, headers=headers)
        self.parser = CustomPDFParser(password=password, extract_images=extract_images)

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """Lazy load given path as pages."""
        if self.web_path:
            blob = Blob.from_data(open(self.file_path, "rb").read(), path=self.web_path)
        else:
            blob = Blob.from_path(self.file_path)
        yield from self.parser.parse(blob)


class CustomPDFDirectoryLoader(BaseLoader):
    """Load a directory with `PDF` files using `pypdf` and chunks at character level.

    Loader also stores page numbers in metadata.
    Based on `langchain_community.document_loaders.pdf.PyPDFDirectoryLoader`.
    """

    def __init__(
        self,
        path: Union[str, Path],
        glob: str = "**/[!.]*.pdf",
        silent_errors: bool = False,
        load_hidden: bool = False,
        recursive: bool = False,
        extract_images: bool = False,
    ):
        self.path = path
        self.glob = glob
        self.load_hidden = load_hidden
        self.recursive = recursive
        self.silent_errors = silent_errors
        self.extract_images = extract_images

    @staticmethod
    def _is_visible(path: Path) -> bool:
        return not any(part.startswith(".") for part in path.parts)

    def load(self) -> List[Document]:
        p = Path(self.path)
        docs = []
        items = p.rglob(self.glob) if self.recursive else p.glob(self.glob)
        for i in items:
            if i.is_file():
                if self._is_visible(i.relative_to(p)) or self.load_hidden:
                    try:
                        loader = CustomPDFLoader(str(i), extract_images=self.extract_images)
                        sub_docs = loader.load()
                        for doc in sub_docs:
                            doc.metadata["source"] = str(i)
                        docs.extend(sub_docs)
                    except Exception as e:
                        raise e
        return docs
