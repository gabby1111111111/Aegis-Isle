"""
Agent Memory System - Manages persistent memory and context for agents.
"""

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, String, DateTime, Text, Integer, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

from .base import AgentMessage
from ..core.config import settings
from ..core.logging import logger


Base = declarative_base()


class MemoryEntry(Base):
    """Database model for agent memory entries."""

    __tablename__ = "agent_memory"

    id = Column(String, primary_key=True)
    agent_id = Column(String, index=True, nullable=False)
    message_id = Column(String, index=True, nullable=False)
    content = Column(Text, nullable=False)
    message_type = Column(String, default="text")
    meta_data = Column(Text)  # JSON string (renamed from metadata to avoid SQLAlchemy reserved word)
    importance_score = Column(Integer, default=1)
    is_system = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    accessed_count = Column(Integer, default=0)
    last_accessed = Column(DateTime, default=datetime.utcnow)


class AgentMemory:
    """Manages persistent memory for agents."""

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or settings.database_url
        self.engine = create_engine(self.database_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

        # In-memory cache for recent messages
        self._cache: Dict[str, List[MemoryEntry]] = {}
        self._cache_size = 50  # Number of recent entries to keep in cache per agent

        logger.info("AgentMemory initialized")

    def store_message(
        self,
        agent_id: str,
        message: AgentMessage,
        importance_score: int = 1,
        is_system: bool = False
    ) -> None:
        """Store a message in the agent's memory."""
        try:
            entry = MemoryEntry(
                id=f"{agent_id}_{message.id}",
                agent_id=agent_id,
                message_id=message.id,
                content=message.content,
                message_type=message.message_type,
                meta_data=json.dumps(message.metadata),
                importance_score=importance_score,
                is_system=is_system,
                created_at=message.timestamp
            )

            self.session.add(entry)
            self.session.commit()

            # Update cache
            self._update_cache(agent_id, entry)

            logger.debug(f"Stored message {message.id} for agent {agent_id}")

        except Exception as e:
            self.session.rollback()
            logger.error(f"Error storing message for agent {agent_id}: {e}")

    def retrieve_recent_messages(
        self,
        agent_id: str,
        limit: int = 10,
        include_system: bool = True
    ) -> List[AgentMessage]:
        """Retrieve recent messages for an agent."""
        try:
            # Try cache first
            if agent_id in self._cache and len(self._cache[agent_id]) >= limit:
                cached_entries = self._cache[agent_id][-limit:]
                if include_system or not any(entry.is_system for entry in cached_entries):
                    return self._entries_to_messages(cached_entries)

            # Query database
            query = self.session.query(MemoryEntry).filter(
                MemoryEntry.agent_id == agent_id
            )

            if not include_system:
                query = query.filter(MemoryEntry.is_system == False)

            entries = query.order_by(MemoryEntry.created_at.desc()).limit(limit).all()

            # Update access statistics
            for entry in entries:
                entry.accessed_count += 1
                entry.last_accessed = datetime.utcnow()

            self.session.commit()

            return self._entries_to_messages(entries)

        except Exception as e:
            logger.error(f"Error retrieving messages for agent {agent_id}: {e}")
            return []

    def retrieve_important_messages(
        self,
        agent_id: str,
        min_importance: int = 5,
        limit: int = 20
    ) -> List[AgentMessage]:
        """Retrieve important messages for an agent."""
        try:
            entries = self.session.query(MemoryEntry).filter(
                MemoryEntry.agent_id == agent_id,
                MemoryEntry.importance_score >= min_importance
            ).order_by(
                MemoryEntry.importance_score.desc(),
                MemoryEntry.created_at.desc()
            ).limit(limit).all()

            return self._entries_to_messages(entries)

        except Exception as e:
            logger.error(f"Error retrieving important messages for agent {agent_id}: {e}")
            return []

    def search_messages(
        self,
        agent_id: str,
        query: str,
        limit: int = 10
    ) -> List[AgentMessage]:
        """Search messages by content."""
        try:
            entries = self.session.query(MemoryEntry).filter(
                MemoryEntry.agent_id == agent_id,
                MemoryEntry.content.contains(query)
            ).order_by(MemoryEntry.created_at.desc()).limit(limit).all()

            return self._entries_to_messages(entries)

        except Exception as e:
            logger.error(f"Error searching messages for agent {agent_id}: {e}")
            return []

    def update_importance_score(
        self,
        agent_id: str,
        message_id: str,
        new_score: int
    ) -> bool:
        """Update the importance score of a message."""
        try:
            entry = self.session.query(MemoryEntry).filter(
                MemoryEntry.agent_id == agent_id,
                MemoryEntry.message_id == message_id
            ).first()

            if entry:
                entry.importance_score = new_score
                self.session.commit()
                logger.debug(
                    f"Updated importance score for message {message_id} "
                    f"to {new_score}"
                )
                return True

            return False

        except Exception as e:
            logger.error(f"Error updating importance score: {e}")
            return False

    def clear_agent_memory(
        self,
        agent_id: str,
        keep_important: bool = True,
        min_importance_to_keep: int = 5
    ) -> int:
        """Clear an agent's memory."""
        try:
            query = self.session.query(MemoryEntry).filter(
                MemoryEntry.agent_id == agent_id
            )

            if keep_important:
                query = query.filter(
                    MemoryEntry.importance_score < min_importance_to_keep
                )

            deleted_count = query.count()
            query.delete()
            self.session.commit()

            # Clear cache
            if agent_id in self._cache:
                if keep_important:
                    self._cache[agent_id] = [
                        entry for entry in self._cache[agent_id]
                        if entry.importance_score >= min_importance_to_keep
                    ]
                else:
                    del self._cache[agent_id]

            logger.info(f"Cleared {deleted_count} memory entries for agent {agent_id}")
            return deleted_count

        except Exception as e:
            logger.error(f"Error clearing memory for agent {agent_id}: {e}")
            return 0

    def cleanup_old_memories(self, days_to_keep: int = 30) -> int:
        """Clean up old memory entries."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

            # Keep important messages even if old
            deleted_count = self.session.query(MemoryEntry).filter(
                MemoryEntry.created_at < cutoff_date,
                MemoryEntry.importance_score < 5  # Keep important messages
            ).count()

            self.session.query(MemoryEntry).filter(
                MemoryEntry.created_at < cutoff_date,
                MemoryEntry.importance_score < 5
            ).delete()

            self.session.commit()

            # Clear cache to force refresh
            self._cache.clear()

            logger.info(f"Cleaned up {deleted_count} old memory entries")
            return deleted_count

        except Exception as e:
            logger.error(f"Error cleaning up old memories: {e}")
            return 0

    def get_memory_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get memory statistics for an agent."""
        try:
            total_entries = self.session.query(MemoryEntry).filter(
                MemoryEntry.agent_id == agent_id
            ).count()

            important_entries = self.session.query(MemoryEntry).filter(
                MemoryEntry.agent_id == agent_id,
                MemoryEntry.importance_score >= 5
            ).count()

            recent_entries = self.session.query(MemoryEntry).filter(
                MemoryEntry.agent_id == agent_id,
                MemoryEntry.created_at >= datetime.utcnow() - timedelta(days=1)
            ).count()

            return {
                "total_entries": total_entries,
                "important_entries": important_entries,
                "recent_entries": recent_entries,
                "cache_size": len(self._cache.get(agent_id, [])),
            }

        except Exception as e:
            logger.error(f"Error getting memory stats for agent {agent_id}: {e}")
            return {}

    def _entries_to_messages(self, entries: List[MemoryEntry]) -> List[AgentMessage]:
        """Convert database entries to AgentMessage objects."""
        messages = []
        for entry in entries:
            try:
                metadata = json.loads(entry.meta_data) if entry.meta_data else {}
                message = AgentMessage(
                    id=entry.message_id,
                    sender_id=entry.agent_id,
                    content=entry.content,
                    message_type=entry.message_type,
                    metadata=metadata,
                    timestamp=entry.created_at
                )
                messages.append(message)
            except Exception as e:
                logger.warning(f"Error converting entry to message: {e}")

        return messages

    def _update_cache(self, agent_id: str, entry: MemoryEntry) -> None:
        """Update the in-memory cache."""
        if agent_id not in self._cache:
            self._cache[agent_id] = []

        self._cache[agent_id].append(entry)

        # Keep only the most recent entries
        if len(self._cache[agent_id]) > self._cache_size:
            self._cache[agent_id] = self._cache[agent_id][-self._cache_size:]

    def close(self) -> None:
        """Close the database session."""
        try:
            self.session.close()
            logger.info("AgentMemory session closed")
        except Exception as e:
            logger.error(f"Error closing AgentMemory session: {e}")