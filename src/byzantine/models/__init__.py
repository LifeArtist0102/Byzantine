"""Shared, serialisable models used by all Byzantine research workflows."""

from byzantine.models.document import BibliographicMetadata, DocumentRecord
from byzantine.models.evidence import Evidence, SourceRegion

__all__ = ["BibliographicMetadata", "DocumentRecord", "Evidence", "SourceRegion"]
