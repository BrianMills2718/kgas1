"""Runtime dependency checks for local NLP models."""

import spacy


def test_en_core_web_sm_model_is_available() -> None:
    """T27 dependency parsing needs the small English spaCy model installed."""
    nlp = spacy.load("en_core_web_sm")

    assert nlp.meta["name"] == "core_web_sm"
    assert nlp.meta["version"] == "3.8.0"
    assert "parser" in nlp.pipe_names
