"""Entity-shape adapter for the T27 relationship extractor contract."""

from typing import Any, Dict, List


def normalize_entities_for_t27(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize entity records into the T27 relationship extractor contract."""
    normalized = []
    for index, entity in enumerate(entities):
        if all(field in entity for field in ("text", "entity_type", "start", "end")):
            normalized.append({
                **entity,
                "text": entity["text"],
                "entity_type": entity["entity_type"],
                "start": entity["start"],
                "end": entity["end"],
                "confidence": entity.get("confidence", 0.8),
            })
            continue

        if all(field in entity for field in ("surface_form", "entity_type", "start_pos", "end_pos")):
            converted = {
                **entity,
                "text": entity["surface_form"],
                "entity_type": entity["entity_type"],
                "start": entity["start_pos"],
                "end": entity["end_pos"],
                "confidence": entity.get("confidence", 0.8),
            }
            normalized.append(converted)
            continue

        raise ValueError(
            "Entity "
            f"{index} is not T27-compatible; expected text/entity_type/start/end "
            "or T23A surface_form/entity_type/start_pos/end_pos"
        )

    return normalized
