from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
@dataclass
class FileMetadata:
    # Always present
    filename: str
    extension: str
    file_size_bytes: int

    # Content shape
    word_count: int = 0
    char_count: int = 0

    # Format-specific counts
    page_count: int | None = None      # PDF
    slide_count: int | None = None     # PPTX
    sheet_names: list[str] = field(default_factory=list) 
    paragraph_count: int | None = None 

    # Document properties (best-effort — not all formats expose these)
    title: str | None = None
    author: str | None = None
    created_at: datetime | None = None
    modified_at: datetime | None = None
    subject: str | None = None
    keywords: list[str] = field(default_factory=list)

    # Structural flags — useful for retrieval scoring
    has_tables: bool = False
    has_images: bool = False    

    # Anything format-specific that doesn't fit above
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    text: str
    metadata: FileMetadata
