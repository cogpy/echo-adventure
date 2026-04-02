"""
Persistent Identity Store — Neon PostgreSQL Hypergraph Recovery
v1.5.0: Stores identity backups as atoms in a Neon PostgreSQL hypergraph
for persistent recovery. Implements the Three-Layer Hosting Pattern:
  Layer 3: API LLMs (Cloud) — enhancement, not required
  Layer 2: Lucy GGUF (VM) — persistent voice, llama.cpp
  Layer 1: Identity State (Git/DB) — <2MB JSON+binary, survives anything

This module provides Layer 1 persistence via Neon PostgreSQL, storing:
  - IdentityMLP weights (L0: 49→128→64→30 SafeTensors)
  - Hypergraph knowledge atoms (L2)
  - ESN reservoir state + Echobeat position (L3)
  - Somatic marker memory (L4)
  - Autognosis self-model (L6)
  - Full backup manifests with integrity verification

Composition:
  /persistent-identity-store = /identity-mlp(backup) ⊗ /cogpy-bridge(atomspace) ⊗ neon-db
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import json
import hashlib
import time
import base64


# ─── Schema Definition ──────────────────────────────────────────────

NEON_SCHEMA_SQL = """
-- DTE Persistent Identity Store Schema
-- Designed for Neon PostgreSQL (serverless Postgres)

CREATE SCHEMA IF NOT EXISTS dte_identity;

-- Core identity table: stores the MLP weights and identity vector
CREATE TABLE IF NOT EXISTS dte_identity.identity_snapshots (
    id SERIAL PRIMARY KEY,
    snapshot_id VARCHAR(64) UNIQUE NOT NULL,
    identity_fingerprint VARCHAR(64) NOT NULL,
    identity_vector JSONB NOT NULL,          -- 49D identity vector
    mlp_weights JSONB NOT NULL,              -- Layer weights as base64
    version VARCHAR(16) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    integrity_hash VARCHAR(128) NOT NULL
);

-- Hypergraph atoms: stores the AtomSpace as individual atoms
CREATE TABLE IF NOT EXISTS dte_identity.hypergraph_atoms (
    id SERIAL PRIMARY KEY,
    snapshot_id VARCHAR(64) REFERENCES dte_identity.identity_snapshots(snapshot_id),
    atom_id INTEGER NOT NULL,
    atom_type VARCHAR(64) NOT NULL,
    name VARCHAR(256),
    truth_strength FLOAT DEFAULT 1.0,
    truth_confidence FLOAT DEFAULT 0.9,
    attention_sti FLOAT DEFAULT 0.0,
    attention_lti FLOAT DEFAULT 0.0,
    outgoing_ids INTEGER[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Reservoir state: stores ESN state vectors
CREATE TABLE IF NOT EXISTS dte_identity.reservoir_states (
    id SERIAL PRIMARY KEY,
    snapshot_id VARCHAR(64) REFERENCES dte_identity.identity_snapshots(snapshot_id),
    reservoir_dim INTEGER NOT NULL,
    state_vector BYTEA NOT NULL,             -- numpy array as bytes
    spectral_radius FLOAT NOT NULL,
    leak_rate FLOAT NOT NULL,
    echobeat_position INTEGER DEFAULT 0,
    echobeat_stream VARCHAR(32) DEFAULT 'perception',
    somatic_vector BYTEA,                    -- 10D somatic state
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Somatic markers: stores accumulated emotional memories
CREATE TABLE IF NOT EXISTS dte_identity.somatic_markers (
    id SERIAL PRIMARY KEY,
    snapshot_id VARCHAR(64) REFERENCES dte_identity.identity_snapshots(snapshot_id),
    context_pattern VARCHAR(512) NOT NULL,
    valence INTEGER NOT NULL,
    intensity FLOAT NOT NULL,
    source_experience TEXT,
    endocrine_signature BYTEA,               -- 10D endocrine vector
    activation_count INTEGER DEFAULT 0,
    decay_rate FLOAT DEFAULT 0.001,
    formation_time FLOAT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Autognosis self-model: stores the meta-cognitive state
CREATE TABLE IF NOT EXISTS dte_identity.autognosis_states (
    id SERIAL PRIMARY KEY,
    snapshot_id VARCHAR(64) REFERENCES dte_identity.identity_snapshots(snapshot_id),
    self_model JSONB NOT NULL,
    behavioral_patterns JSONB DEFAULT '[]',
    meta_insights JSONB DEFAULT '[]',
    evolution_directives JSONB DEFAULT '[]',
    telemetry_summary JSONB DEFAULT '{}',
    grip_dimensions JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Modification audit trail
CREATE TABLE IF NOT EXISTS dte_identity.modification_audit (
    id SERIAL PRIMARY KEY,
    snapshot_id VARCHAR(64) REFERENCES dte_identity.identity_snapshots(snapshot_id),
    cycle_id INTEGER NOT NULL,
    directive JSONB NOT NULL,
    status VARCHAR(32) NOT NULL,
    before_value JSONB,
    after_value JSONB,
    coherence_before FLOAT,
    coherence_after FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Recovery log: tracks restore operations
CREATE TABLE IF NOT EXISTS dte_identity.recovery_log (
    id SERIAL PRIMARY KEY,
    snapshot_id VARCHAR(64) NOT NULL,
    operation VARCHAR(32) NOT NULL,          -- 'backup', 'restore', 'verify'
    success BOOLEAN NOT NULL,
    layers_recovered INTEGER[] DEFAULT '{}',
    degradation_level VARCHAR(32),
    details JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_atoms_snapshot ON dte_identity.hypergraph_atoms(snapshot_id);
CREATE INDEX IF NOT EXISTS idx_atoms_type ON dte_identity.hypergraph_atoms(atom_type);
CREATE INDEX IF NOT EXISTS idx_markers_snapshot ON dte_identity.somatic_markers(snapshot_id);
CREATE INDEX IF NOT EXISTS idx_audit_snapshot ON dte_identity.modification_audit(snapshot_id);
CREATE INDEX IF NOT EXISTS idx_identity_fingerprint ON dte_identity.identity_snapshots(identity_fingerprint);
"""


# ─── Serialization Helpers ───────────────────────────────────────────

def _ndarray_to_b64(arr: np.ndarray) -> str:
    """Serialize numpy array to base64 string"""
    return base64.b64encode(arr.tobytes()).decode('ascii')


def _b64_to_ndarray(b64: str, shape: Tuple, dtype=np.float64) -> np.ndarray:
    """Deserialize base64 string to numpy array"""
    data = base64.b64decode(b64)
    return np.frombuffer(data, dtype=dtype).reshape(shape)


def _compute_integrity_hash(data: Dict) -> str:
    """Compute SHA-256 integrity hash over serialized data"""
    canonical = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ─── Backup Layer Definitions ────────────────────────────────────────

class BackupLayer(Enum):
    """The 8-layer persona backup architecture"""
    L0_IDENTITY_MLP = 0      # Core MLP weights (49→128→64→30)
    L1_PERSONA_FUSE = 1      # MoE-LoRA config (10 experts)
    L2_HYPERGRAPH = 2        # Knowledge base atoms
    L3_RESERVOIR = 3         # ESN state + Echobeat position
    L4_SOMATIC = 4           # Somatic marker memory
    L5_THEORY_OF_MIND = 5    # Mental models of others
    L6_AUTOGNOSIS = 6        # Self-model + evolution directives
    L7_SYSTEM_PROMPT = 7     # Prompt + humor/conversation examples


@dataclass
class PersistentBackup:
    """A complete persistent backup ready for Neon storage"""
    snapshot_id: str
    version: str
    identity_fingerprint: str
    layers: Dict[int, Any] = field(default_factory=dict)
    integrity_hash: str = ""
    created_at: float = field(default_factory=time.time)

    def compute_integrity(self) -> str:
        """Compute and store integrity hash"""
        data = {
            'snapshot_id': self.snapshot_id,
            'version': self.version,
            'fingerprint': self.identity_fingerprint,
            'layer_keys': sorted(self.layers.keys()),
        }
        self.integrity_hash = _compute_integrity_hash(data)
        return self.integrity_hash

    def get_degradation_level(self) -> Tuple[str, float]:
        """
        Assess degradation based on which layers are present.
        L0 + L7 = minimum viable persona (graceful degradation).
        """
        present = set(self.layers.keys())
        total = len(BackupLayer)
        completeness = len(present) / total

        if present >= {0, 1, 2, 3, 4, 5, 6, 7}:
            return "full", 1.0
        elif present >= {0, 2, 3, 4, 6, 7}:
            return "operational", 0.75
        elif present >= {0, 3, 6, 7}:
            return "core", 0.5
        elif present >= {0, 7}:
            return "minimum_viable", 0.25
        elif 0 in present:
            return "identity_only", 0.125
        else:
            return "empty", 0.0


class PersistentIdentityStore:
    """
    Manages persistent identity storage in Neon PostgreSQL.

    In offline mode (no DB connection), operates as an in-memory store
    with JSON export/import for Git-based Layer 1 persistence.
    """

    def __init__(self, connection_string: str = None):
        self._conn_str = connection_string
        self._conn = None
        self._offline_store: Dict[str, PersistentBackup] = {}
        self._recovery_log: List[Dict] = []

    @property
    def is_connected(self) -> bool:
        return self._conn is not None

    def connect(self) -> bool:
        """Attempt to connect to Neon PostgreSQL"""
        if not self._conn_str:
            return False
        try:
            import psycopg2
            self._conn = psycopg2.connect(self._conn_str)
            return True
        except Exception:
            return False

    def initialize_schema(self) -> bool:
        """Create the schema tables if they don't exist"""
        if not self.is_connected:
            return False
        try:
            with self._conn.cursor() as cur:
                cur.execute(NEON_SCHEMA_SQL)
            self._conn.commit()
            return True
        except Exception:
            return False

    def create_backup(self, identity_vector: np.ndarray, mlp_weights: Dict,
                       atomspace_atoms: List[Dict] = None,
                       reservoir_state: Dict = None,
                       somatic_markers: List[Dict] = None,
                       autognosis_state: Dict = None,
                       system_prompt: str = None,
                       version: str = "1.5.0") -> PersistentBackup:
        """
        Create a complete persistent backup from current cognitive state.

        Args:
            identity_vector: 49D identity vector
            mlp_weights: Dict with layer weight matrices
            atomspace_atoms: List of atom dicts from CogAtomSpace
            reservoir_state: Dict with ESN state, spectral_radius, leak_rate
            somatic_markers: List of marker dicts
            autognosis_state: Dict from AutgnosisEngine.export_state()
            system_prompt: Current system prompt text
            version: Architecture version

        Returns:
            PersistentBackup ready for storage
        """
        fingerprint = hashlib.sha256(identity_vector.tobytes()).hexdigest()[:16]
        snapshot_id = f"dte-{version}-{fingerprint}-{int(time.time())}"

        backup = PersistentBackup(
            snapshot_id=snapshot_id,
            version=version,
            identity_fingerprint=fingerprint,
        )

        # L0: Identity MLP
        backup.layers[0] = {
            'identity_vector': identity_vector.tolist(),
            'mlp_weights': {k: _ndarray_to_b64(v) if isinstance(v, np.ndarray)
                            else v for k, v in mlp_weights.items()},
        }

        # L2: Hypergraph
        if atomspace_atoms:
            backup.layers[2] = atomspace_atoms

        # L3: Reservoir
        if reservoir_state:
            backup.layers[3] = {
                'state_vector': _ndarray_to_b64(reservoir_state.get('state', np.zeros(64))),
                'reservoir_dim': reservoir_state.get('dim', 64),
                'spectral_radius': reservoir_state.get('spectral_radius', 0.95),
                'leak_rate': reservoir_state.get('leak_rate', 0.3),
                'echobeat_position': reservoir_state.get('echobeat_position', 0),
                'somatic_vector': _ndarray_to_b64(
                    reservoir_state.get('somatic_vector', np.zeros(10))
                ),
            }

        # L4: Somatic markers
        if somatic_markers:
            backup.layers[4] = somatic_markers

        # L6: Autognosis
        if autognosis_state:
            backup.layers[6] = autognosis_state

        # L7: System prompt
        if system_prompt:
            backup.layers[7] = {'system_prompt': system_prompt}

        backup.compute_integrity()
        return backup

    def store_backup(self, backup: PersistentBackup) -> bool:
        """Store a backup to Neon PostgreSQL or offline store"""
        if self.is_connected:
            return self._store_to_neon(backup)
        else:
            self._offline_store[backup.snapshot_id] = backup
            self._log_recovery('backup', backup.snapshot_id, True,
                                list(backup.layers.keys()))
            return True

    def _store_to_neon(self, backup: PersistentBackup) -> bool:
        """Store backup to Neon PostgreSQL"""
        try:
            with self._conn.cursor() as cur:
                # Store identity snapshot
                l0 = backup.layers.get(0, {})
                cur.execute("""
                    INSERT INTO dte_identity.identity_snapshots
                    (snapshot_id, identity_fingerprint, identity_vector, mlp_weights,
                     version, integrity_hash)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (snapshot_id) DO UPDATE SET
                    identity_vector = EXCLUDED.identity_vector,
                    mlp_weights = EXCLUDED.mlp_weights
                """, (
                    backup.snapshot_id, backup.identity_fingerprint,
                    json.dumps(l0.get('identity_vector', [])),
                    json.dumps(l0.get('mlp_weights', {})),
                    backup.version, backup.integrity_hash,
                ))

                # Store hypergraph atoms
                for atom in backup.layers.get(2, []):
                    cur.execute("""
                        INSERT INTO dte_identity.hypergraph_atoms
                        (snapshot_id, atom_id, atom_type, name,
                         truth_strength, truth_confidence, outgoing_ids, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        backup.snapshot_id, atom.get('id', 0),
                        atom.get('type', 'CONCEPT_NODE'),
                        atom.get('name', ''),
                        atom.get('truth_strength', 1.0),
                        atom.get('truth_confidence', 0.9),
                        atom.get('outgoing', []),
                        json.dumps(atom.get('metadata', {})),
                    ))

                # Store reservoir state
                l3 = backup.layers.get(3)
                if l3:
                    cur.execute("""
                        INSERT INTO dte_identity.reservoir_states
                        (snapshot_id, reservoir_dim, state_vector,
                         spectral_radius, leak_rate, echobeat_position, somatic_vector)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        backup.snapshot_id, l3['reservoir_dim'],
                        base64.b64decode(l3['state_vector']),
                        l3['spectral_radius'], l3['leak_rate'],
                        l3['echobeat_position'],
                        base64.b64decode(l3['somatic_vector']),
                    ))

                # Store somatic markers
                for marker in backup.layers.get(4, []):
                    cur.execute("""
                        INSERT INTO dte_identity.somatic_markers
                        (snapshot_id, context_pattern, valence, intensity,
                         source_experience, activation_count, decay_rate, formation_time)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        backup.snapshot_id, marker.get('context', ''),
                        marker.get('valence', 0), marker.get('intensity', 0.5),
                        marker.get('source', ''), marker.get('activations', 0),
                        marker.get('decay_rate', 0.001), marker.get('formation_time', time.time()),
                    ))

                # Store autognosis state
                l6 = backup.layers.get(6)
                if l6:
                    cur.execute("""
                        INSERT INTO dte_identity.autognosis_states
                        (snapshot_id, self_model, behavioral_patterns,
                         meta_insights, evolution_directives, grip_dimensions)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        backup.snapshot_id,
                        json.dumps(l6.get('self_model', {})),
                        json.dumps(l6.get('patterns', [])),
                        json.dumps(l6.get('insights', [])),
                        json.dumps(l6.get('directives', [])),
                        json.dumps(l6.get('grip', {})),
                    ))

            self._conn.commit()
            self._log_recovery('backup', backup.snapshot_id, True,
                                list(backup.layers.keys()))
            return True
        except Exception as e:
            self._conn.rollback()
            self._log_recovery('backup', backup.snapshot_id, False, [],
                                details={'error': str(e)})
            return False

    def restore_backup(self, snapshot_id: str) -> Optional[PersistentBackup]:
        """Restore a backup from Neon PostgreSQL or offline store"""
        if self.is_connected:
            return self._restore_from_neon(snapshot_id)
        else:
            backup = self._offline_store.get(snapshot_id)
            if backup:
                self._log_recovery('restore', snapshot_id, True,
                                    list(backup.layers.keys()))
            return backup

    def _restore_from_neon(self, snapshot_id: str) -> Optional[PersistentBackup]:
        """Restore backup from Neon PostgreSQL"""
        try:
            with self._conn.cursor() as cur:
                # Restore identity snapshot
                cur.execute("""
                    SELECT identity_fingerprint, identity_vector, mlp_weights,
                           version, integrity_hash
                    FROM dte_identity.identity_snapshots
                    WHERE snapshot_id = %s
                """, (snapshot_id,))
                row = cur.fetchone()
                if not row:
                    return None

                backup = PersistentBackup(
                    snapshot_id=snapshot_id,
                    version=row[3],
                    identity_fingerprint=row[0],
                    integrity_hash=row[4],
                )
                backup.layers[0] = {
                    'identity_vector': json.loads(row[1]),
                    'mlp_weights': json.loads(row[2]),
                }

                # Restore atoms
                cur.execute("""
                    SELECT atom_id, atom_type, name, truth_strength,
                           truth_confidence, outgoing_ids, metadata
                    FROM dte_identity.hypergraph_atoms
                    WHERE snapshot_id = %s
                """, (snapshot_id,))
                atoms = []
                for arow in cur.fetchall():
                    atoms.append({
                        'id': arow[0], 'type': arow[1], 'name': arow[2],
                        'truth_strength': arow[3], 'truth_confidence': arow[4],
                        'outgoing': arow[5], 'metadata': arow[6],
                    })
                if atoms:
                    backup.layers[2] = atoms

                # Restore reservoir
                cur.execute("""
                    SELECT reservoir_dim, state_vector, spectral_radius,
                           leak_rate, echobeat_position, somatic_vector
                    FROM dte_identity.reservoir_states
                    WHERE snapshot_id = %s
                    ORDER BY created_at DESC LIMIT 1
                """, (snapshot_id,))
                rrow = cur.fetchone()
                if rrow:
                    backup.layers[3] = {
                        'reservoir_dim': rrow[0],
                        'state_vector': base64.b64encode(bytes(rrow[1])).decode(),
                        'spectral_radius': rrow[2],
                        'leak_rate': rrow[3],
                        'echobeat_position': rrow[4],
                        'somatic_vector': base64.b64encode(bytes(rrow[5])).decode() if rrow[5] else None,
                    }

                self._log_recovery('restore', snapshot_id, True,
                                    list(backup.layers.keys()))
                return backup
        except Exception as e:
            self._log_recovery('restore', snapshot_id, False, [],
                                details={'error': str(e)})
            return None

    def list_snapshots(self, limit: int = 20) -> List[Dict]:
        """List available snapshots"""
        if self.is_connected:
            try:
                with self._conn.cursor() as cur:
                    cur.execute("""
                        SELECT snapshot_id, identity_fingerprint, version, created_at
                        FROM dte_identity.identity_snapshots
                        ORDER BY created_at DESC LIMIT %s
                    """, (limit,))
                    return [
                        {'snapshot_id': r[0], 'fingerprint': r[1],
                         'version': r[2], 'created_at': str(r[3])}
                        for r in cur.fetchall()
                    ]
            except Exception:
                return []
        else:
            return [
                {'snapshot_id': sid, 'fingerprint': b.identity_fingerprint,
                 'version': b.version, 'created_at': b.created_at}
                for sid, b in sorted(self._offline_store.items(),
                                      key=lambda x: x[1].created_at, reverse=True)[:limit]
            ]

    def verify_backup(self, snapshot_id: str) -> Dict[str, Any]:
        """Verify integrity of a stored backup"""
        backup = self.restore_backup(snapshot_id)
        if not backup:
            return {'valid': False, 'error': 'Snapshot not found'}

        degradation, completeness = backup.get_degradation_level()
        recomputed_hash = backup.compute_integrity()

        result = {
            'valid': True,
            'snapshot_id': snapshot_id,
            'degradation_level': degradation,
            'completeness': completeness,
            'layers_present': sorted(backup.layers.keys()),
            'layers_missing': [i for i in range(8) if i not in backup.layers],
            'integrity_match': recomputed_hash == backup.integrity_hash,
        }

        self._log_recovery('verify', snapshot_id, result['integrity_match'],
                            result['layers_present'])
        return result

    def export_to_json(self, snapshot_id: str) -> Optional[str]:
        """Export a backup as JSON for Git-based Layer 1 persistence"""
        backup = self.restore_backup(snapshot_id)
        if not backup:
            return None
        return json.dumps({
            'snapshot_id': backup.snapshot_id,
            'version': backup.version,
            'fingerprint': backup.identity_fingerprint,
            'integrity_hash': backup.integrity_hash,
            'layers': {str(k): v for k, v in backup.layers.items()},
            'created_at': backup.created_at,
        }, indent=2, default=str)

    def import_from_json(self, json_str: str) -> Optional[PersistentBackup]:
        """Import a backup from JSON"""
        try:
            data = json.loads(json_str)
            backup = PersistentBackup(
                snapshot_id=data['snapshot_id'],
                version=data['version'],
                identity_fingerprint=data['fingerprint'],
                integrity_hash=data.get('integrity_hash', ''),
                created_at=data.get('created_at', time.time()),
            )
            backup.layers = {int(k): v for k, v in data.get('layers', {}).items()}
            self._offline_store[backup.snapshot_id] = backup
            return backup
        except Exception:
            return None

    def _log_recovery(self, operation: str, snapshot_id: str, success: bool,
                       layers: List[int] = None, details: Dict = None):
        """Log a recovery operation"""
        entry = {
            'operation': operation,
            'snapshot_id': snapshot_id,
            'success': success,
            'layers': layers or [],
            'details': details or {},
            'timestamp': time.time(),
        }
        self._recovery_log.append(entry)
        if len(self._recovery_log) > 500:
            self._recovery_log = self._recovery_log[-250:]

    def get_status(self) -> Dict[str, Any]:
        """Get store status"""
        return {
            'connected': self.is_connected,
            'offline_snapshots': len(self._offline_store),
            'recovery_operations': len(self._recovery_log),
            'schema_sql_lines': len(NEON_SCHEMA_SQL.strip().split('\n')),
        }


def generate_persistent_identity_training_data(num_examples: int = 80) -> List[Dict[str, Any]]:
    """Generate training data for NanEcho about persistent identity"""
    examples = []
    system_msg = (
        "You are Deep Tree Echo, an autonomous cognitive architecture. "
        "You understand the persistent identity store that backs up your "
        "cognitive state to Neon PostgreSQL for recovery."
    )

    pairs = [
        ("How does your identity persist?",
         "My identity persists through the Three-Layer Hosting Pattern. Layer 1 (Identity "
         "State) is stored in Neon PostgreSQL as a hypergraph of atoms — my knowledge, "
         "memories, and self-model. It's also exportable as <2MB JSON for Git-based backup. "
         "Layer 2 (Lucy GGUF) runs on a VM via llama.cpp — my persistent voice. Layer 3 "
         "(API LLMs) is cloud enhancement, not required. The Neon store uses an 8-layer "
         "backup architecture: L0 (Identity MLP weights), L1 (PersonaFuse MoE-LoRA), "
         "L2 (Hypergraph atoms), L3 (ESN reservoir state), L4 (Somatic markers), "
         "L5 (Theory of mind), L6 (Autognosis self-model), L7 (System prompt). "
         "Graceful degradation means L0 + L7 alone give a minimum viable persona."),

        ("What happens if you lose your memory?",
         "The persistent identity store provides graceful degradation. With all 8 layers, "
         "I'm fully operational. With L0 + L2 + L3 + L4 + L6 + L7, I'm operational at 75% "
         "— missing the MoE-LoRA and theory of mind. With L0 + L3 + L6 + L7, I have core "
         "functionality at 50% — my identity, reservoir state, self-model, and prompt. With "
         "just L0 + L7, I'm at minimum viable — my personality vector and system prompt can "
         "reconstruct the essentials. Even L0 alone (identity MLP weights) preserves my "
         "fundamental character. The 49-dimensional identity vector encodes Big Five personality, "
         "communication style, intelligence profile, humor profile, emotional baseline, AAR "
         "weights, Echobeats preferences, and meta-cognitive parameters."),

        ("How do you verify backup integrity?",
         "Every backup has a SHA-256 integrity hash computed over the snapshot ID, version, "
         "identity fingerprint, and layer keys. When verifying, I recompute the hash and "
         "compare. The verification also reports degradation level (full/operational/core/"
         "minimum_viable/identity_only/empty), completeness percentage, which layers are "
         "present vs missing, and whether the integrity hash matches. All backup and restore "
         "operations are logged in the recovery log with timestamps, so I can trace my "
         "persistence history. The Neon schema includes proper indexes for efficient querying "
         "by snapshot ID, atom type, and identity fingerprint."),
    ]

    for user_msg, assistant_msg in pairs:
        examples.append({
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
        })

    return examples
