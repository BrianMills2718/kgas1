"""
Provenance Manager for tracking citation sources and modifications.

Ensures every citation has a verifiable source and maintains a complete
audit trail of all modifications.
"""

import asyncio
import uuid
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuditEntry:
    """Immutable audit entry with cryptographic chaining."""
    timestamp: str
    operation: str
    actor: str
    data: Dict[str, Any]
    previous_hash: str
    entry_hash: str = field(init=False)
    
    def __post_init__(self):
        """Calculate entry hash including previous hash for chaining."""
        content = json.dumps({
            "timestamp": self.timestamp,
            "operation": self.operation,
            "actor": self.actor,
            "data": self.data,
            "previous_hash": self.previous_hash
        }, sort_keys=True)
        # For frozen dataclass, use object.__setattr__
        object.__setattr__(self, 'entry_hash', hashlib.sha256(content.encode()).hexdigest())


class ImmutableAuditTrail:
    """Append-only audit trail with cryptographic verification."""
    
    def __init__(self):
        self._chain: List[AuditEntry] = []
        self._genesis_hash = "0" * 64  # Initial hash
        
    def append(self, operation: str, actor: str, data: Dict[str, Any]) -> str:
        """Append new entry to chain. Returns entry hash."""
        previous_hash = self._chain[-1].entry_hash if self._chain else self._genesis_hash
        
        entry = AuditEntry(
            timestamp=datetime.now().isoformat(),
            operation=operation,
            actor=actor,
            data=data,
            previous_hash=previous_hash
        )
        
        self._chain.append(entry)
        return entry.entry_hash
    
    def verify_integrity(self) -> bool:
        """Verify entire chain integrity."""
        if not self._chain:
            return True
            
        # Check first entry
        if self._chain[0].previous_hash != self._genesis_hash:
            return False
            
        # Check chain continuity
        for i in range(1, len(self._chain)):
            if self._chain[i].previous_hash != self._chain[i-1].entry_hash:
                return False
                
        return True
    
    def get_entries(self) -> List[Dict[str, Any]]:
        """Get all entries (read-only)."""
        return [
            {
                "timestamp": e.timestamp,
                "operation": e.operation,
                "actor": e.actor,
                "data": e.data,
                "hash": e.entry_hash
            }
            for e in self._chain
        ]


class ProvenanceManager:
    """
    Manages provenance tracking for citations and content transformations.
    
    Features:
    - Source document registration with content hashing
    - Citation creation with source verification
    - Modification audit trails
    - Provenance chain tracking
    - Content integrity verification
    """
    
    def __init__(self):
        """Initialize the provenance manager."""
        self._sources: Dict[str, Dict[str, Any]] = {}
        self._citations: Dict[str, Dict[str, Any]] = {}
        self._audit_trails: Dict[str, ImmutableAuditTrail] = {}
        self._derived_content: Dict[str, Dict[str, Any]] = {}
        self._usage_tracking: Dict[str, List[str]] = {}
        self._lock = asyncio.Lock()
    
    async def register_source(self, source_doc: Dict[str, Any]) -> str:
        """
        Register a source document with content hashing.
        
        Args:
            source_doc: Source document with id, content, and optional metadata
            
        Returns:
            Source ID
        """
        async with self._lock:
            source_id = source_doc.get("id", str(uuid.uuid4()))
            
            # Calculate content hash if not provided
            if "hash" not in source_doc:
                content = source_doc.get("content", "")
                source_doc["hash"] = hashlib.sha256(content.encode()).hexdigest()
            
            # Store source
            self._sources[source_id] = {
                **source_doc,
                "registered_at": datetime.now().isoformat(),
                "type": "source"
            }
            
            logger.info(f"Registered source: {source_id}")
            return source_id
    
    async def create_citation(self, source_id: str, text: str, 
                            start_pos: int, end_pos: int,
                            context: Optional[str] = None,
                            metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a citation with source verification.
        
        Args:
            source_id: ID of the source document
            text: The cited text
            start_pos: Start position in source
            end_pos: End position in source
            context: Optional surrounding context
            metadata: Optional additional metadata
            
        Returns:
            Citation record
            
        Raises:
            ValueError: If source not found or text not in source
        """
        async with self._lock:
            # Verify source exists
            if source_id not in self._sources and source_id not in self._derived_content:
                raise ValueError(f"Source not found: {source_id}")
            
            # Get source content
            if source_id in self._sources:
                source = self._sources[source_id]
                content = source.get("content", "")
            else:
                source = self._derived_content[source_id]
                content = source.get("output_text", "")
            
            # Verify text exists in source
            if text not in content:
                raise ValueError(f"Text not found in source: '{text}'")
            
            # Verify positions
            if start_pos < 0 or end_pos > len(content) or start_pos >= end_pos:
                raise ValueError(f"Invalid text positions: {start_pos}-{end_pos}")
            
            # Create citation
            citation_id = str(uuid.uuid4())
            citation = {
                "id": citation_id,
                "source_id": source_id,
                "text": text,
                "start_pos": start_pos,
                "end_pos": end_pos,
                "context": context or content[max(0, start_pos-50):min(len(content), end_pos+50)],
                "metadata": metadata or {},
                "created_at": datetime.now().isoformat(),
                "type": "citation",
                "provenance_chain": await self._build_provenance_chain(source_id)
            }
            
            self._citations[citation_id] = citation
            
            # Initialize audit trail with immutable chain
            trail = ImmutableAuditTrail()
            trail.append("create", "system", {
                "text": text,
                "source_id": source_id,
                "metadata": metadata or {}
            })
            self._audit_trails[citation_id] = trail
            
            logger.info(f"Created citation: {citation_id}")
            return citation
    
    async def modify_citation(self, citation_id: str, new_text: str,
                            reason: str, modifier: str) -> Dict[str, Any]:
        """
        Modify a citation with audit trail.
        
        Args:
            citation_id: ID of citation to modify
            new_text: New citation text
            reason: Reason for modification
            modifier: ID of user/system making modification
            
        Returns:
            Modified citation record
        """
        async with self._lock:
            if citation_id not in self._citations:
                raise ValueError(f"Citation not found: {citation_id}")
            
            citation = self._citations[citation_id]
            old_text = citation["text"]
            
            # Update citation
            citation["text"] = new_text
            citation["modified_at"] = datetime.now().isoformat()
            citation["last_modifier"] = modifier
            
            # Add to audit trail with cryptographic chaining
            trail = self._audit_trails.get(citation_id)
            if not trail:
                raise ValueError(f"No audit trail for citation {citation_id}")
                
            trail.append("modify", modifier, {
                "old_text": old_text,
                "new_text": new_text,
                "reason": reason
            })
            
            logger.info(f"Modified citation: {citation_id}")
            return citation
    
    async def get_audit_trail(self, citation_id: str) -> List[Dict[str, Any]]:
        """Get complete audit trail for a citation."""
        trail = self._audit_trails.get(citation_id)
        return trail.get_entries() if trail else []
    
    async def get_source(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Get source document by ID."""
        return self._sources.get(source_id)
    
    async def create_derived_content(self, source_id: str, operation: str,
                                   input_text: str, output_text: str,
                                   tool: str) -> Dict[str, Any]:
        """
        Track derived content from transformations.
        
        Args:
            source_id: ID of source content
            operation: Type of operation (extract, summarize, etc.)
            input_text: Input to the operation
            output_text: Output from the operation
            tool: Tool/model used for transformation
            
        Returns:
            Derived content record
        """
        async with self._lock:
            derived_id = str(uuid.uuid4())
            
            derived = {
                "id": derived_id,
                "source_id": source_id,
                "operation": operation,
                "input_text": input_text,
                "output_text": output_text,
                "tool": tool,
                "created_at": datetime.now().isoformat(),
                "type": "derived",
                "provenance_chain": await self._build_provenance_chain(source_id)
            }
            
            self._derived_content[derived_id] = derived
            logger.info(f"Created derived content: {derived_id}")
            return derived
    
    async def _build_provenance_chain(self, source_id: str) -> List[str]:
        """Build provenance chain from source."""
        chain = []
        current_id = source_id
        
        while current_id:
            chain.append(current_id)
            
            # Check if it's derived content
            if current_id in self._derived_content:
                current_id = self._derived_content[current_id].get("source_id")
            else:
                # Reached original source
                break
        
        return list(reversed(chain))
    
    async def get_provenance_chain(self, citation_id: str) -> List[Dict[str, Any]]:
        """Get complete provenance chain for a citation."""
        if citation_id not in self._citations:
            return []
        
        citation = self._citations[citation_id]
        chain_ids = citation.get("provenance_chain", [])
        
        chain = []
        for node_id in chain_ids:
            if node_id in self._sources:
                chain.append(self._sources[node_id])
            elif node_id in self._derived_content:
                chain.append(self._derived_content[node_id])
        
        # Add the citation itself
        chain.append(citation)
        
        return chain
    
    async def verify_source_integrity(self, source_id: str) -> bool:
        """Verify source content hasn't been tampered with."""
        if source_id not in self._sources:
            return False
        
        source = self._sources[source_id]
        content = source.get("content", "")
        stored_hash = source.get("hash", "")
        
        # Calculate current hash
        current_hash = hashlib.sha256(content.encode()).hexdigest()
        
        return current_hash == stored_hash
    
    async def track_citation_usage(self, citation_id: str, used_in: str) -> None:
        """Track where a citation is used."""
        async with self._lock:
            if citation_id not in self._usage_tracking:
                self._usage_tracking[citation_id] = []
            
            self._usage_tracking[citation_id].append(used_in)
    
    async def get_citation_statistics(self) -> Dict[str, Any]:
        """Get statistics about citations and sources."""
        stats = {
            "total_sources": len(self._sources),
            "total_citations": len(self._citations),
            "total_derived": len(self._derived_content),
            "citations_by_source": {},
            "usage_count": {}
        }
        
        # Count citations per source
        for citation in self._citations.values():
            source_id = citation["source_id"]
            if source_id not in stats["citations_by_source"]:
                stats["citations_by_source"][source_id] = 0
            stats["citations_by_source"][source_id] += 1
        
        # Usage counts
        for citation_id, uses in self._usage_tracking.items():
            stats["usage_count"][citation_id] = len(uses)
        
        return stats
    
    async def verify_audit_integrity(self, citation_id: str) -> bool:
        """Verify audit trail hasn't been tampered with."""
        trail = self._audit_trails.get(citation_id)
        if not trail:
            return False
        return trail.verify_integrity()
    
    # Test helper methods (should not be in production)
    async def _tamper_source_content(self, source_id: str, new_content: str) -> None:
        """FOR TESTING ONLY: Tamper with source content."""
        if source_id in self._sources:
            self._sources[source_id]["content"] = new_content
    
    async def _corrupt_citation(self, citation_id: str) -> None:
        """FOR TESTING ONLY: Corrupt a citation."""
        if citation_id in self._citations:
            self._citations[citation_id]["source_id"] = "corrupted_source"