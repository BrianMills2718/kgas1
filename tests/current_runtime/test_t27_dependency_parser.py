"""Runtime contract checks for T27 dependency-parser extraction."""

import logging

import spacy

from src.tools.phase1.t27_relationship_extractor_unified import T27RelationshipExtractorUnified


def test_t27_dependency_parser_extracts_subject_object_relationship() -> None:
    """T27 should produce relationships from spaCy subject-verb-object parses."""
    extractor = object.__new__(T27RelationshipExtractorUnified)
    extractor.logger = logging.getLogger("test.t27")
    extractor.dependency_extractions = 0
    nlp = spacy.load("en_core_web_sm")
    text = "Alice admired Bob."
    entities = [
        {"text": "Alice", "entity_type": "PERSON", "start": 0, "end": 5, "confidence": 0.9},
        {"text": "Bob", "entity_type": "PERSON", "start": 14, "end": 17, "confidence": 0.9},
    ]

    relationships = extractor._extract_dependency_relationships(
        text=text,
        entities=entities,
        chunk_ref="chunk-1",
        confidence_threshold=0.5,
        nlp=nlp,
    )

    assert len(relationships) == 1
    relationship = relationships[0]
    assert relationship["extraction_method"] == "dependency_parsing"
    assert relationship["relationship_type"] == "RELATED_TO"
    assert relationship["subject"]["text"] == "Alice"
    assert relationship["object"]["text"] == "Bob"
    assert relationship["verb"] == "admire"
