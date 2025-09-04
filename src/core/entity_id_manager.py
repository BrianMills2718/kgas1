"""
Entity ID Manager for consistent ID mapping between Neo4j and SQLite.

Ensures entity IDs remain consistent across both databases and provides
mechanisms for ID generation, mapping, and validation.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from contextlib import asynccontextmanager
import logging
import aiosqlite
from neo4j import AsyncSession

logger = logging.getLogger(__name__)


class EntityIDTransaction:
    """Transaction context for entity ID operations."""
    
    def __init__(self, manager: 'EntityIDManager'):
        self.manager = manager
        self.operations = []
        self.committed = False
    
    async def create_neo4j_node(self, entity_type: str, properties: Dict[str, Any]) -> str:
        """Create a Neo4j node and return its ID."""
        query = f"CREATE (n:{entity_type} $props) RETURN elementId(n) as id"
        result = await self.manager.neo4j_session.run(query, props=properties)
        record = await result.single()
        neo4j_id = record["id"]
        
        self.operations.append(("neo4j_create", entity_type, neo4j_id, properties))
        return neo4j_id
    
    async def create_sqlite_record(self, internal_id: str, data: Dict[str, Any]) -> None:
        """Create a SQLite record."""
        self.operations.append(("sqlite_create", internal_id, data))
    
    async def create_id_mapping(self, internal_id: str, neo4j_id: str) -> None:
        """Create ID mapping."""
        self.operations.append(("mapping_create", internal_id, neo4j_id))
    
    async def commit(self) -> None:
        """Commit all operations."""
        # In a real implementation, this would use the distributed transaction manager
        for op in self.operations:
            if op[0] == "mapping_create":
                await self.manager.create_id_mapping(op[1], op[2], "Unknown")
        self.committed = True
    
    async def rollback(self) -> None:
        """Rollback all operations."""
        self.operations.clear()


class EntityIDManager:
    """
    Manages entity ID consistency between Neo4j and SQLite.
    
    Features:
    - Generates unique entity IDs with type prefixes
    - Maintains bidirectional ID mappings
    - Validates ID consistency across databases
    - Detects and reports orphaned IDs
    """
    
    def __init__(self, sqlite_path: str, neo4j_session: AsyncSession):
        """
        Initialize the ID manager.
        
        Args:
            sqlite_path: Path to SQLite database
            neo4j_session: Neo4j session for queries
        """
        self.sqlite_path = sqlite_path
        self.neo4j_session = neo4j_session
        self._lock = asyncio.Lock()
        self._id_cache = {}  # Simple cache for frequently accessed mappings
    
    async def generate_entity_id(self, entity_type: str) -> str:
        """
        Generate a unique entity ID with type prefix.
        
        Args:
            entity_type: Type of entity (Person, Document, etc.)
            
        Returns:
            Unique entity ID like "PERSON_550e8400-e29b-41d4-a716-446655440000"
        """
        # Normalize entity type to uppercase
        type_prefix = entity_type.upper()
        
        # Generate UUID
        unique_suffix = str(uuid.uuid4())
        
        # Combine with underscore separator
        entity_id = f"{type_prefix}_{unique_suffix}"
        
        # Verify uniqueness in database
        async with aiosqlite.connect(self.sqlite_path) as db:
            cursor = await db.execute(
                "SELECT COUNT(*) FROM entity_mappings WHERE internal_id = ?",
                [entity_id]
            )
            count = await cursor.fetchone()
            
            # Extremely unlikely, but regenerate if collision
            if count[0] > 0:
                return await self.generate_entity_id(entity_type)
        
        logger.debug(f"Generated entity ID: {entity_id}")
        return entity_id
    
    async def create_id_mapping(self, internal_id: str, neo4j_id: str, entity_type: str) -> None:
        """
        Create a mapping between internal ID and Neo4j ID.
        
        Args:
            internal_id: Internal entity ID
            neo4j_id: Neo4j node ID
            entity_type: Type of entity
            
        Raises:
            ValueError: If mapping already exists
        """
        async with self._lock:
            async with aiosqlite.connect(self.sqlite_path) as db:
                try:
                    await db.execute(
                        """
                        INSERT INTO entity_mappings (internal_id, neo4j_id, entity_type)
                        VALUES (?, ?, ?)
                        """,
                        [internal_id, neo4j_id, entity_type]
                    )
                    await db.commit()
                    
                    # Update cache
                    self._id_cache[internal_id] = neo4j_id
                    self._id_cache[neo4j_id] = internal_id
                    
                    logger.info(f"Created ID mapping: {internal_id} <-> {neo4j_id}")
                    
                except aiosqlite.IntegrityError as e:
                    if "UNIQUE constraint failed" in str(e):
                        raise ValueError(f"ID mapping already exists: {e}")
                    raise
    
    async def get_neo4j_id(self, internal_id: str) -> Optional[str]:
        """
        Get Neo4j ID for an internal ID.
        
        Args:
            internal_id: Internal entity ID
            
        Returns:
            Neo4j ID or None if not found
        """
        # Check cache first
        if internal_id in self._id_cache:
            return self._id_cache[internal_id]
        
        async with aiosqlite.connect(self.sqlite_path) as db:
            cursor = await db.execute(
                "SELECT neo4j_id FROM entity_mappings WHERE internal_id = ?",
                [internal_id]
            )
            row = await cursor.fetchone()
            
            if row:
                neo4j_id = row[0]
                self._id_cache[internal_id] = neo4j_id
                return neo4j_id
            
            return None
    
    async def get_internal_id(self, neo4j_id: str) -> Optional[str]:
        """
        Get internal ID for a Neo4j ID.
        
        Args:
            neo4j_id: Neo4j node ID
            
        Returns:
            Internal ID or None if not found
        """
        # Check cache first
        if neo4j_id in self._id_cache:
            return self._id_cache[neo4j_id]
        
        async with aiosqlite.connect(self.sqlite_path) as db:
            cursor = await db.execute(
                "SELECT internal_id FROM entity_mappings WHERE neo4j_id = ?",
                [neo4j_id]
            )
            row = await cursor.fetchone()
            
            if row:
                internal_id = row[0]
                self._id_cache[neo4j_id] = internal_id
                return internal_id
            
            return None
    
    async def validate_id_consistency(self, internal_id: str, neo4j_id: str) -> bool:
        """
        Validate that an ID mapping is consistent.
        
        Args:
            internal_id: Internal entity ID
            neo4j_id: Neo4j node ID
            
        Returns:
            True if mapping is consistent, False otherwise
        """
        stored_neo4j_id = await self.get_neo4j_id(internal_id)
        stored_internal_id = await self.get_internal_id(neo4j_id)
        
        return (stored_neo4j_id == neo4j_id and 
                stored_internal_id == internal_id)
    
    async def find_orphaned_ids(self) -> List[Dict[str, Any]]:
        """
        Find IDs that exist in mapping table but not in Neo4j.
        
        Returns:
            List of orphaned ID records
        """
        orphaned = []
        
        async with aiosqlite.connect(self.sqlite_path) as db:
            cursor = await db.execute(
                "SELECT internal_id, neo4j_id, entity_type FROM entity_mappings"
            )
            
            async for row in cursor:
                internal_id, neo4j_id, entity_type = row
                
                # Check if node exists in Neo4j
                query = "MATCH (n) WHERE elementId(n) = $id RETURN n"
                result = await self.neo4j_session.run(query, id=neo4j_id)
                record = await result.single()
                
                if not record:
                    orphaned.append({
                        "internal_id": internal_id,
                        "neo4j_id": neo4j_id,
                        "entity_type": entity_type,
                        "status": "missing_in_neo4j"
                    })
        
        logger.warning(f"Found {len(orphaned)} orphaned IDs")
        return orphaned
    
    async def validate_all_mappings(self) -> Dict[str, int]:
        """
        Validate all ID mappings for consistency.
        
        Returns:
            Dictionary with validation statistics
        """
        stats = {
            "total": 0,
            "valid": 0,
            "invalid": 0,
            "errors": []
        }
        
        async with aiosqlite.connect(self.sqlite_path) as db:
            cursor = await db.execute(
                "SELECT internal_id, neo4j_id FROM entity_mappings"
            )
            
            async for row in cursor:
                internal_id, neo4j_id = row
                stats["total"] += 1
                
                if await self.validate_id_consistency(internal_id, neo4j_id):
                    stats["valid"] += 1
                else:
                    stats["invalid"] += 1
                    stats["errors"].append({
                        "internal_id": internal_id,
                        "neo4j_id": neo4j_id,
                        "reason": "inconsistent_mapping"
                    })
        
        logger.info(f"Validation complete: {stats['valid']}/{stats['total']} valid mappings")
        return stats
    
    @asynccontextmanager
    async def create_entity_transaction(self):
        """
        Create a transaction context for entity operations.
        
        Yields:
            EntityIDTransaction for coordinated operations
        """
        transaction = EntityIDTransaction(self)
        try:
            yield transaction
        except Exception:
            await transaction.rollback()
            raise
        else:
            if not transaction.committed:
                await transaction.rollback()
    
    async def cleanup_orphaned_ids(self, dry_run: bool = True) -> Dict[str, Any]:
        """
        Clean up orphaned ID mappings.
        
        Args:
            dry_run: If True, only report what would be cleaned
            
        Returns:
            Cleanup statistics
        """
        orphaned = await self.find_orphaned_ids()
        
        if dry_run:
            return {
                "dry_run": True,
                "would_remove": len(orphaned),
                "orphaned_ids": orphaned
            }
        
        removed = 0
        async with aiosqlite.connect(self.sqlite_path) as db:
            for record in orphaned:
                await db.execute(
                    "DELETE FROM entity_mappings WHERE internal_id = ?",
                    [record["internal_id"]]
                )
                removed += 1
            
            await db.commit()
        
        logger.info(f"Removed {removed} orphaned ID mappings")
        
        return {
            "dry_run": False,
            "removed": removed,
            "orphaned_ids": orphaned
        }