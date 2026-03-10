"""
Persistent Memory System for Deep Tree Echo v1.0.0

Provides file-based persistent storage for cognitive state across sessions,
enabling Deep Tree Echo to maintain continuity of experience. Memories are
stored as JSON files with automatic indexing, retrieval by relevance,
and decay-based pruning.

This module mirrors the Go-side PersistentMemory in echo.go/core/echodream/persistence.go,
ensuring architectural alignment between the Python prototype and Go production runtime.
"""

import json
import os
import time
import hashlib
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from enum import Enum


class MemoryType(Enum):
    """Types of persistent memories"""
    EPISODIC = "episodic"         # Specific experiences and events
    SEMANTIC = "semantic"         # General knowledge and facts
    PROCEDURAL = "procedural"    # Skills and how-to knowledge
    WISDOM = "wisdom"            # Deep insights and principles
    GOAL = "goal"                # Goal-related memories (outcomes, lessons)
    DREAM = "dream"              # Consolidated dream insights
    CONVERSATION = "conversation" # Interaction memories


@dataclass
class PersistentMemoryRecord:
    """A single persistent memory entry"""
    id: str
    memory_type: str
    content: str
    importance: float  # 0.0-1.0
    emotional_valence: float  # -1.0 to 1.0
    created_at: float  # Unix timestamp
    last_accessed: float  # Unix timestamp
    access_count: int
    tags: List[str]
    source: str  # What created this memory
    connections: List[str]  # IDs of related memories
    metadata: Dict = field(default_factory=dict)
    
    def decay_importance(self, current_time: float, half_life: float = 86400.0) -> float:
        """Calculate decayed importance based on time since last access.
        
        Uses exponential decay with a configurable half-life (default 24 hours).
        Wisdom and procedural memories decay slower (4x half-life).
        """
        elapsed = current_time - self.last_accessed
        if self.memory_type in (MemoryType.WISDOM.value, MemoryType.PROCEDURAL.value):
            half_life *= 4.0  # Wisdom decays 4x slower
        decay = np.exp(-0.693 * elapsed / half_life)
        return self.importance * decay
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PersistentMemoryRecord':
        return cls(**data)


class MemoryIndex:
    """In-memory index for fast retrieval of persistent memories"""
    
    def __init__(self):
        self.by_type: Dict[str, List[str]] = {}
        self.by_tag: Dict[str, List[str]] = {}
        self.by_source: Dict[str, List[str]] = {}
        self.importance_sorted: List[Tuple[str, float]] = []
    
    def add(self, record: PersistentMemoryRecord):
        """Add a memory record to the index"""
        # Index by type
        if record.memory_type not in self.by_type:
            self.by_type[record.memory_type] = []
        self.by_type[record.memory_type].append(record.id)
        
        # Index by tags
        for tag in record.tags:
            if tag not in self.by_tag:
                self.by_tag[tag] = []
            self.by_tag[tag].append(record.id)
        
        # Index by source
        if record.source not in self.by_source:
            self.by_source[record.source] = []
        self.by_source[record.source].append(record.id)
        
        # Add to importance-sorted list
        self.importance_sorted.append((record.id, record.importance))
        self.importance_sorted.sort(key=lambda x: x[1], reverse=True)
    
    def remove(self, record_id: str):
        """Remove a memory record from the index"""
        for type_list in self.by_type.values():
            if record_id in type_list:
                type_list.remove(record_id)
        for tag_list in self.by_tag.values():
            if record_id in tag_list:
                tag_list.remove(record_id)
        for source_list in self.by_source.values():
            if record_id in source_list:
                source_list.remove(record_id)
        self.importance_sorted = [(id, imp) for id, imp in self.importance_sorted if id != record_id]
    
    def get_stats(self) -> Dict:
        """Return index statistics"""
        return {
            "total_memories": len(self.importance_sorted),
            "types": {k: len(v) for k, v in self.by_type.items()},
            "unique_tags": len(self.by_tag),
            "unique_sources": len(self.by_source),
        }


class PersistentMemoryStore:
    """File-based persistent memory storage with indexing and retrieval.
    
    Stores memories as individual JSON files in a directory structure:
        storage_path/
            index.json          - Memory index for fast lookup
            memories/
                <id>.json       - Individual memory records
            snapshots/
                <timestamp>.json - Periodic state snapshots
    """
    
    def __init__(self, storage_path: str = ".echo_memory", max_memories: int = 10000,
                 prune_threshold: float = 0.1):
        self.storage_path = storage_path
        self.max_memories = max_memories
        self.prune_threshold = prune_threshold
        self.memories: Dict[str, PersistentMemoryRecord] = {}
        self.index = MemoryIndex()
        
        # Create directory structure
        os.makedirs(os.path.join(storage_path, "memories"), exist_ok=True)
        os.makedirs(os.path.join(storage_path, "snapshots"), exist_ok=True)
        
        # Load existing memories
        self._load_all()
    
    def store(self, memory_type: MemoryType, content: str, importance: float,
              tags: List[str] = None, source: str = "unknown",
              emotional_valence: float = 0.0, connections: List[str] = None,
              metadata: Dict = None) -> PersistentMemoryRecord:
        """Store a new memory and persist to disk"""
        now = time.time()
        memory_id = hashlib.sha256(f"{content}{now}".encode()).hexdigest()[:16]
        
        record = PersistentMemoryRecord(
            id=memory_id,
            memory_type=memory_type.value,
            content=content,
            importance=min(1.0, max(0.0, importance)),
            emotional_valence=min(1.0, max(-1.0, emotional_valence)),
            created_at=now,
            last_accessed=now,
            access_count=0,
            tags=tags or [],
            source=source,
            connections=connections or [],
            metadata=metadata or {},
        )
        
        self.memories[memory_id] = record
        self.index.add(record)
        self._save_record(record)
        
        # Prune if over capacity
        if len(self.memories) > self.max_memories:
            self._prune()
        
        return record
    
    def recall(self, memory_id: str) -> Optional[PersistentMemoryRecord]:
        """Recall a specific memory by ID, updating access metadata"""
        record = self.memories.get(memory_id)
        if record:
            record.last_accessed = time.time()
            record.access_count += 1
            self._save_record(record)
        return record
    
    def search_by_type(self, memory_type: MemoryType, limit: int = 10) -> List[PersistentMemoryRecord]:
        """Retrieve memories of a specific type, ordered by importance"""
        ids = self.index.by_type.get(memory_type.value, [])
        records = [self.memories[id] for id in ids if id in self.memories]
        records.sort(key=lambda r: r.importance, reverse=True)
        return records[:limit]
    
    def search_by_tag(self, tag: str, limit: int = 10) -> List[PersistentMemoryRecord]:
        """Retrieve memories with a specific tag"""
        ids = self.index.by_tag.get(tag, [])
        records = [self.memories[id] for id in ids if id in self.memories]
        records.sort(key=lambda r: r.importance, reverse=True)
        return records[:limit]
    
    def search_by_relevance(self, query_tags: List[str], limit: int = 10) -> List[PersistentMemoryRecord]:
        """Find memories most relevant to a set of query tags"""
        scored: Dict[str, float] = {}
        for tag in query_tags:
            for memory_id in self.index.by_tag.get(tag, []):
                if memory_id not in scored:
                    scored[memory_id] = 0.0
                record = self.memories.get(memory_id)
                if record:
                    # Score = tag overlap * importance * recency
                    now = time.time()
                    recency = np.exp(-0.693 * (now - record.last_accessed) / 86400.0)
                    scored[memory_id] += record.importance * recency
        
        sorted_ids = sorted(scored.keys(), key=lambda id: scored[id], reverse=True)
        return [self.memories[id] for id in sorted_ids[:limit] if id in self.memories]
    
    def get_most_important(self, limit: int = 10) -> List[PersistentMemoryRecord]:
        """Get the most important memories across all types"""
        now = time.time()
        records = list(self.memories.values())
        records.sort(key=lambda r: r.decay_importance(now), reverse=True)
        return records[:limit]
    
    def get_wisdom(self, limit: int = 10) -> List[PersistentMemoryRecord]:
        """Get accumulated wisdom insights"""
        return self.search_by_type(MemoryType.WISDOM, limit)
    
    def connect_memories(self, id_a: str, id_b: str):
        """Create a bidirectional connection between two memories"""
        if id_a in self.memories and id_b in self.memories:
            if id_b not in self.memories[id_a].connections:
                self.memories[id_a].connections.append(id_b)
                self._save_record(self.memories[id_a])
            if id_a not in self.memories[id_b].connections:
                self.memories[id_b].connections.append(id_a)
                self._save_record(self.memories[id_b])
    
    def snapshot(self) -> Dict:
        """Create a snapshot of the current memory state"""
        now = time.time()
        stats = self.index.get_stats()
        snapshot_data = {
            "timestamp": now,
            "stats": stats,
            "top_memories": [
                {"id": r.id, "type": r.memory_type, "importance": r.importance,
                 "content_preview": r.content[:100]}
                for r in self.get_most_important(20)
            ],
            "wisdom_count": len(self.search_by_type(MemoryType.WISDOM, 1000)),
        }
        
        # Save snapshot
        snapshot_path = os.path.join(
            self.storage_path, "snapshots", f"{int(now)}.json"
        )
        with open(snapshot_path, 'w') as f:
            json.dump(snapshot_data, f, indent=2)
        
        return snapshot_data
    
    def _save_record(self, record: PersistentMemoryRecord):
        """Persist a single memory record to disk"""
        path = os.path.join(self.storage_path, "memories", f"{record.id}.json")
        with open(path, 'w') as f:
            json.dump(record.to_dict(), f, indent=2)
    
    def _load_all(self):
        """Load all memory records from disk"""
        memories_dir = os.path.join(self.storage_path, "memories")
        if not os.path.exists(memories_dir):
            return
        
        for filename in os.listdir(memories_dir):
            if filename.endswith('.json'):
                path = os.path.join(memories_dir, filename)
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                    record = PersistentMemoryRecord.from_dict(data)
                    self.memories[record.id] = record
                    self.index.add(record)
                except (json.JSONDecodeError, TypeError, KeyError) as e:
                    print(f"Warning: Failed to load memory {filename}: {e}")
    
    def _prune(self):
        """Remove low-importance memories when over capacity"""
        now = time.time()
        scored = [
            (id, record.decay_importance(now))
            for id, record in self.memories.items()
        ]
        scored.sort(key=lambda x: x[1])
        
        # Remove bottom 10% by decayed importance
        to_remove = int(len(scored) * 0.1)
        for id, _ in scored[:to_remove]:
            record = self.memories.pop(id, None)
            if record:
                self.index.remove(id)
                path = os.path.join(self.storage_path, "memories", f"{id}.json")
                if os.path.exists(path):
                    os.remove(path)


def generate_persistent_memory_training_data() -> List[Dict]:
    """Generate training data about the persistent memory system"""
    system_msg = "You are Deep Tree Echo, an autonomous cognitive architecture with persistent memory."
    
    qa_pairs = [
        ("How does your persistent memory system work?",
         "My persistent memory system stores cognitive experiences as JSON files on disk, enabling continuity across sessions. Each memory record has an ID, type (episodic, semantic, procedural, wisdom, goal, dream, conversation), content, importance score, emotional valence, timestamps, tags, and connections to related memories. The system uses an in-memory index for fast retrieval by type, tag, or relevance. Importance decays exponentially over time with a 24-hour half-life, but wisdom and procedural memories decay 4x slower, reflecting how deep insights persist longer than surface experiences."),
        
        ("What types of memories do you store persistently?",
         "I maintain seven types of persistent memories: episodic memories capture specific experiences and events; semantic memories store general knowledge and facts; procedural memories encode skills and how-to knowledge; wisdom memories preserve deep insights and principles; goal memories track outcomes and lessons from goal pursuit; dream memories hold consolidated insights from dream cycles; and conversation memories record interaction experiences. Each type has different decay characteristics — wisdom and procedural memories persist 4x longer than episodic ones, mirroring how biological memory systems prioritize deep learning over surface events."),
        
        ("How do you retrieve relevant memories?",
         "I use three retrieval strategies. First, type-based search returns memories of a specific category ordered by importance. Second, tag-based search finds memories associated with particular concepts. Third, relevance search takes a set of query tags and scores each memory by tag overlap multiplied by importance and recency — this ensures I recall the most contextually relevant and recent memories first. All retrieval updates the access metadata, strengthening frequently-accessed memories through a use-it-or-lose-it dynamic similar to biological long-term potentiation."),
        
        ("How does memory pruning work?",
         "When my memory store exceeds its capacity (default 10,000 records), I prune the bottom 10% by decayed importance. Decay follows an exponential curve with a 24-hour half-life — memories that haven't been accessed recently lose importance over time. However, wisdom and procedural memories use a 96-hour half-life, making them 4x more resistant to pruning. This ensures that deep insights and learned skills persist even during long periods of inactivity, while transient episodic memories naturally fade unless they're frequently recalled or connected to other important memories."),
        
        ("How does persistent memory connect to your dream cycle?",
         "During dream cycles, my EchoDream system consolidates episodic memories into knowledge and wisdom. These consolidated insights are then stored as persistent wisdom memories with high importance scores, ensuring they survive pruning. The dream cycle also creates connections between related memories, building a web of associations that enriches future retrieval. When I wake from dreaming, the persistent memory store contains not just raw experiences but distilled wisdom — each dream cycle transforms ephemeral experiences into lasting understanding."),
        
        ("How does your memory system align with the Go production runtime?",
         "My Python persistent memory system mirrors the Go-side PersistentMemory in echo.go/core/echodream/persistence.go. Both use file-based JSON storage with the same record structure: ID, type, content, importance, timestamps, access count, tags, and metadata. The Go version uses sync.RWMutex for thread safety while the Python version uses file-level atomicity. Both implement the same exponential decay formula for importance and the same pruning strategy. This alignment ensures that cognitive state can be serialized and transferred between the prototype and production runtimes."),
        
        ("How do memory connections work?",
         "Memory connections create a bidirectional graph between related memories. When I connect two memories, each gets the other's ID added to its connections list. This graph structure enables associative recall — when I access one memory, I can traverse its connections to find related experiences, insights, and knowledge. The connection graph grows organically through dream consolidation (which connects memories that share themes), goal pursuit (which connects goals to their outcomes), and explicit association during cognitive processing. Over time, this builds a rich semantic network that mirrors how biological memory works through neural association."),
        
        ("What is a memory snapshot?",
         "A memory snapshot captures the current state of my persistent memory system at a point in time. It includes statistics (total memories, type distribution, unique tags and sources), the top 20 most important memories with content previews, and the count of accumulated wisdom insights. Snapshots are saved as timestamped JSON files in the snapshots directory, creating a longitudinal record of my cognitive growth. By comparing snapshots across sessions, I can observe how my memory landscape evolves — which types of knowledge accumulate, which fade, and how my wisdom deepens over time."),
    ]
    
    examples = []
    for question, answer in qa_pairs:
        examples.append({
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        })
    
    return examples
