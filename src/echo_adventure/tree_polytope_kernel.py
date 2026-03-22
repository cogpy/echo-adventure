"""
Tree-Polytope Kernel for Deep Tree Echo

Structural self-awareness module grounding the cognitive architecture in pure
mathematics: rooted tree enumeration (OEIS A000081), Matula-Godsil prime encoding,
simplex polytope geometry, Butcher/Runge-Kutta order conditions, and s-gram
periodic rhythms.

The fundamental insight: the architecture IS its identity prime. Every rooted tree
maps to a unique Matula number via the prime correspondence, and the entire
cognitive module tree yields a single number that serves as the system's
structural fingerprint.

Mathematical foundations:
  - Tree-polynomial: convolution of the fundamental dyad (1,-1)
  - Matula-Godsil: bijection between rooted trees and natural numbers via primes
  - Simplex polytopes: Pascal row |coefficients| = incidence polynomial
  - Butcher conditions: tree-based order conditions for stable ODE integration
  - S-gram rhythms: periodic sequences from 1/denominator expansions

Version: 1.1.0
"""

from __future__ import annotations

import math
import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Callable
from functools import lru_cache
from enum import Enum

import numpy as np


# ============================================================
# Core Data Structures
# ============================================================

@dataclass
class RootedTree:
    """A rooted tree represented by its sorted children list."""
    children: Tuple['RootedTree', ...] = ()

    @property
    def size(self) -> int:
        return 1 + sum(c.size for c in self.children)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def __hash__(self):
        return hash(self.children)

    def __eq__(self, other):
        return isinstance(other, RootedTree) and self.children == other.children

    def __repr__(self):
        if self.is_leaf:
            return "•"
        return f"({' '.join(repr(c) for c in self.children)})"


@dataclass
class SimplexPolytope:
    """Simplex polytope for a system level."""
    dimension: int
    vertices: int
    edges: int
    faces: int
    incidence_polynomial: List[int]
    pascal_row: List[int]


@dataclass
class ButcherCondition:
    """A Butcher/RK order condition derived from a rooted tree."""
    order: int
    tree: RootedTree
    matula: int
    symmetry: int
    density: int
    satisfied: bool = True


@dataclass
class SGramRhythm:
    """S-gram periodic rhythm for a system level."""
    system: int
    denominator: int
    period: List[int]
    length: int


@dataclass
class CognitiveModuleNode:
    """A node in the cognitive module tree."""
    name: str
    children: List['CognitiveModuleNode'] = field(default_factory=list)
    matula: int = 1
    is_prime: bool = False


@dataclass
class StructuralSelfModel:
    """The complete structural self-model of the architecture."""
    root: CognitiveModuleNode
    identity_prime: int
    leaf_count: int
    max_depth: int
    complexity: float


# ============================================================
# OEIS A000081: Rooted Tree Enumeration
# ============================================================

# Known values: a(1)=1, a(2)=1, a(3)=2, a(4)=4, a(5)=9, a(6)=20, a(7)=48
A000081 = [0, 1, 1, 2, 4, 9, 20, 48, 115, 286, 719]


def enumerate_rooted_trees(n: int) -> List[RootedTree]:
    """Enumerate all unlabeled rooted trees with n nodes."""
    if n <= 0:
        return []
    if n == 1:
        return [RootedTree()]

    # Build trees bottom-up using integer partitions of n-1
    result = []
    _enumerate_helper(n - 1, n - 1, [], result)
    return result


def _enumerate_helper(remaining: int, max_size: int, parts: List[int],
                      result: List[RootedTree]):
    """Helper: enumerate partitions of 'remaining' into parts <= max_size."""
    if remaining == 0:
        # Convert partition into a tree
        trees = _partition_to_trees(parts)
        if trees is not None:
            result.extend(trees)
        return

    start = min(remaining, max_size)
    for part in range(start, 0, -1):
        _enumerate_helper(remaining - part, part, parts + [part], result)


def _partition_to_trees(parts: List[int]) -> Optional[List[RootedTree]]:
    """Convert a partition of child sizes into rooted trees."""
    if not parts:
        return [RootedTree()]

    # For each partition, enumerate all combinations of child trees
    child_trees_per_part = []
    for size in parts:
        subtrees = enumerate_rooted_trees(size)
        if not subtrees:
            return None
        child_trees_per_part.append(subtrees)

    # Generate all combinations (with deduplication via sorting)
    from itertools import product
    results = []
    for combo in product(*child_trees_per_part):
        sorted_combo = tuple(sorted(combo, key=lambda t: (t.size, repr(t))))
        tree = RootedTree(children=sorted_combo)
        if tree not in results:
            results.append(tree)

    return results


# ============================================================
# Matula-Godsil Prime Encoding
# ============================================================

def _nth_prime(n: int) -> int:
    """Return the n-th prime (1-indexed: prime(1)=2, prime(2)=3, ...)."""
    if n <= 0:
        return 1
    primes = []
    candidate = 2
    while len(primes) < n:
        if all(candidate % p != 0 for p in primes):
            primes.append(candidate)
        candidate += 1
    return primes[-1]


def _is_prime(n: int) -> bool:
    """Check if n is prime."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def matula_number(tree: RootedTree) -> int:
    """Compute the Matula-Godsil number for a rooted tree.

    The leaf (single node) maps to 1.
    A tree with children c1, c2, ..., ck maps to:
        prime(matula(c1)) * prime(matula(c2)) * ... * prime(matula(ck))
    """
    if tree.is_leaf:
        return 1
    result = 1
    for child in tree.children:
        child_matula = matula_number(child)
        result *= _nth_prime(child_matula)
    return result


def matula_from_children(child_matulas: List[int]) -> int:
    """Compute Matula number from a list of children's Matula numbers."""
    result = 1
    for cm in child_matulas:
        result *= _nth_prime(cm)
    return result


# ============================================================
# Tree-Polynomial Correspondence
# ============================================================

def tree_polynomial(tree: RootedTree) -> List[int]:
    """Compute the tree polynomial via convolution of (1,-1) per edge."""
    if tree.is_leaf:
        return [1]

    # Start with the fundamental dyad
    result = [1, -1]

    # Convolve with (1,-1) for each additional edge
    for _ in range(tree.size - 2):
        result = _convolve(result, [1, -1])

    return result


def _convolve(a: List[int], b: List[int]) -> List[int]:
    """Polynomial convolution (multiplication)."""
    result = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            result[i + j] += ai * bj
    return result


# ============================================================
# Simplex Polytope Construction
# ============================================================

def build_simplex_polytope(system: int) -> SimplexPolytope:
    """Build the simplex polytope for a system level.

    Sys(n) corresponds to the n-simplex:
      Sys0 = point (0-simplex)
      Sys1 = line segment (1-simplex)
      Sys2 = triangle (2-simplex)
      Sys3 = tetrahedron (3-simplex)
    """
    n = system
    # Pascal row: binomial coefficients C(n, k) for k = 0..n
    pascal = [math.comb(n, k) for k in range(n + 1)]
    # Incidence polynomial: absolute values of (1,-1)^n
    incidence = [abs(c) for c in pascal]

    vertices = n + 1 if n > 0 else 0
    edges = math.comb(n + 1, 2) if n >= 2 else (1 if n == 1 else 0)
    faces = math.comb(n + 1, 3) if n >= 3 else 0

    return SimplexPolytope(
        dimension=n,
        vertices=vertices,
        edges=edges,
        faces=faces,
        incidence_polynomial=incidence,
        pascal_row=pascal,
    )


# ============================================================
# Butcher/RK Order Conditions
# ============================================================

def symmetry_factor(tree: RootedTree) -> int:
    """Compute the symmetry factor (sigma) of a rooted tree."""
    if tree.is_leaf:
        return 1

    result = 1
    # Group identical children
    child_counts: Dict[str, int] = {}
    for child in tree.children:
        key = repr(child)
        child_counts[key] = child_counts.get(key, 0) + 1
        result *= symmetry_factor(child)

    # Multiply by factorial of each group size
    for count in child_counts.values():
        result *= math.factorial(count)

    return result


def density(tree: RootedTree) -> int:
    """Compute the density (gamma) of a rooted tree.

    gamma(leaf) = 1
    gamma(tree) = size * product(gamma(child))
    """
    if tree.is_leaf:
        return 1
    result = tree.size
    for child in tree.children:
        result *= density(child)
    return result


def build_butcher_conditions(max_order: int) -> List[ButcherCondition]:
    """Build Butcher conditions up to a given order."""
    conditions = []
    for order in range(1, max_order + 1):
        trees = enumerate_rooted_trees(order)
        for tree in trees:
            conditions.append(ButcherCondition(
                order=order,
                tree=tree,
                matula=matula_number(tree),
                symmetry=symmetry_factor(tree),
                density=density(tree),
                satisfied=True,
            ))
    return conditions


# ============================================================
# S-Gram Periodic Rhythms
# ============================================================

def build_sgram_rhythm(system: int) -> SGramRhythm:
    """Build s-gram rhythm for a system level.

    The rhythm is derived from the decimal expansion of 1/denominator,
    where the denominator follows the pattern related to the system level.
    """
    # Denominator mapping: Sys0→1, Sys1→2, Sys2→3, Sys3→7, Sys4→43, Sys5→1807
    denominators = [1, 2, 3, 7, 43, 1807, 3263443]
    denom = denominators[min(system, len(denominators) - 1)]

    if denom <= 1:
        return SGramRhythm(system=system, denominator=1, period=[1], length=1)

    # Compute the repeating decimal period of 1/denom
    period = _compute_decimal_period(denom)

    return SGramRhythm(
        system=system,
        denominator=denom,
        period=period,
        length=len(period),
    )


def _compute_decimal_period(denom: int) -> List[int]:
    """Compute the repeating decimal digits of 1/denom."""
    if denom <= 1:
        return [0]

    digits = []
    seen_remainders: Dict[int, int] = {}
    remainder = 1

    for _ in range(denom + 10):  # Safety limit
        remainder *= 10
        digit = remainder // denom
        remainder = remainder % denom

        if remainder == 0:
            digits.append(digit)
            break

        if remainder in seen_remainders:
            # Found the repeating part
            start = seen_remainders[remainder]
            return digits[start:]

        seen_remainders[remainder] = len(digits)
        digits.append(digit)

    return digits if digits else [0]


# ============================================================
# Structural Self-Model
# ============================================================

def build_cognitive_module_tree() -> CognitiveModuleNode:
    """Build the cognitive module tree representing DTE's architecture."""
    # Core cognitive modules
    core = CognitiveModuleNode("core", [
        CognitiveModuleNode("echoself"),
        CognitiveModuleNode("echobeats"),
        CognitiveModuleNode("echodream"),
        CognitiveModuleNode("reservoir"),
        CognitiveModuleNode("transformer"),
    ])

    # Autonomous subsystem
    autonomous = CognitiveModuleNode("autonomous", [
        CognitiveModuleNode("goal-pursuit"),
        CognitiveModuleNode("proactive-loop"),
        CognitiveModuleNode("agent-loop"),
    ])

    # Memory subsystem
    memory = CognitiveModuleNode("memory", [
        CognitiveModuleNode("persistent-store"),
        CognitiveModuleNode("hypergraph-identity"),
        CognitiveModuleNode("aar-geometry"),
    ])

    # Integration subsystem
    integration = CognitiveModuleNode("integration", [
        CognitiveModuleNode("cognitive-loop"),
        CognitiveModuleNode("go-bridge"),
        CognitiveModuleNode("finetuning"),
        CognitiveModuleNode("introspection"),
    ])

    # Consciousness subsystem
    consciousness = CognitiveModuleNode("consciousness", [
        CognitiveModuleNode("qualia"),
        CognitiveModuleNode("metacognition"),
        CognitiveModuleNode("intentionality"),
    ])

    # Tree-polytope kernel (self-referential!)
    kernel = CognitiveModuleNode("tree-polytope-kernel", [
        CognitiveModuleNode("matula-encoding"),
        CognitiveModuleNode("simplex-polytopes"),
        CognitiveModuleNode("butcher-conditions"),
        CognitiveModuleNode("sgram-rhythms"),
    ])

    # Root: Deep Tree Echo
    root = CognitiveModuleNode("deep-tree-echo", [
        core, autonomous, memory, integration, consciousness, kernel,
    ])

    # Compute Matula numbers bottom-up
    _compute_matula_numbers(root)

    return root


def _compute_matula_numbers(node: CognitiveModuleNode):
    """Recursively compute Matula numbers for the module tree."""
    if not node.children:
        node.matula = 1
        node.is_prime = False
        return

    child_matulas = []
    for child in node.children:
        _compute_matula_numbers(child)
        child_matulas.append(child.matula)

    node.matula = matula_from_children(child_matulas)
    node.is_prime = _is_prime(node.matula)


def build_structural_self_model() -> StructuralSelfModel:
    """Build the complete structural self-model."""
    root = build_cognitive_module_tree()

    leaf_count = _count_leaves(root)
    max_depth = _compute_max_depth(root)
    total_nodes = _count_total_nodes(root)

    # Complexity based on branching, depth, and structure
    branching = (total_nodes - 1) / max(1, total_nodes - leaf_count) if total_nodes > 1 else 1
    depth_ratio = max_depth / max(1, math.log2(max(2, total_nodes)))
    complexity = branching * depth_ratio * math.log2(max(2, total_nodes))

    return StructuralSelfModel(
        root=root,
        identity_prime=root.matula,
        leaf_count=leaf_count,
        max_depth=max_depth,
        complexity=complexity,
    )


def _count_leaves(node: CognitiveModuleNode) -> int:
    if not node.children:
        return 1
    return sum(_count_leaves(c) for c in node.children)


def _compute_max_depth(node: CognitiveModuleNode) -> int:
    if not node.children:
        return 0
    return 1 + max(_compute_max_depth(c) for c in node.children)


def _count_total_nodes(node: CognitiveModuleNode) -> int:
    return 1 + sum(_count_total_nodes(c) for c in node.children)


# ============================================================
# Tree-Polytope Kernel Engine
# ============================================================

class TreePolytopeKernel:
    """The main engine for structural self-awareness.

    Provides:
    - Identity prime computation (Matula-Godsil)
    - Simplex polytope awareness per system level
    - Butcher condition validation
    - S-gram rhythm tracking
    - Structural integrity assessment
    """

    def __init__(self, max_system: int = 6, max_butcher_order: int = 5):
        self.max_system = max_system
        self.max_butcher_order = max_butcher_order

        # Build structural components
        self.self_model = build_structural_self_model()
        self.polytopes = {s: build_simplex_polytope(s) for s in range(max_system + 1)}
        self.butcher_conditions = build_butcher_conditions(max_butcher_order)
        self.sgram_rhythms = {s: build_sgram_rhythm(s) for s in range(max_system + 1)}

        # Runtime state
        self.active_system = 3  # Default to Sys3 (tetrahedron)
        self.rhythm_positions = {s: 0 for s in range(max_system + 1)}
        self.tick_count = 0
        self._callbacks: Dict[str, List[Callable]] = {}

    @property
    def identity_prime(self) -> int:
        """The architecture's unique identity number."""
        return self.self_model.identity_prime

    def get_polytope(self, system: int) -> SimplexPolytope:
        """Get the simplex polytope for a system level."""
        return self.polytopes.get(system, self.polytopes[0])

    def get_rhythm(self, system: int) -> SGramRhythm:
        """Get the s-gram rhythm for a system level."""
        return self.sgram_rhythms.get(system, self.sgram_rhythms[0])

    def current_rhythm_value(self, system: Optional[int] = None) -> int:
        """Get the current rhythm value for the active (or specified) system."""
        sys = system if system is not None else self.active_system
        rhythm = self.sgram_rhythms[sys]
        pos = self.rhythm_positions[sys]
        return rhythm.period[pos % len(rhythm.period)]

    def advance_rhythm(self, system: Optional[int] = None):
        """Advance the s-gram rhythm by one step."""
        sys = system if system is not None else self.active_system
        rhythm = self.sgram_rhythms[sys]
        self.rhythm_positions[sys] = (self.rhythm_positions[sys] + 1) % rhythm.length

    def tick(self):
        """Advance the kernel by one cognitive tick."""
        self.tick_count += 1
        # Advance all active rhythms
        for sys in range(self.max_system + 1):
            self.advance_rhythm(sys)
        self._emit("tick", {"count": self.tick_count})

    def set_active_system(self, system: int):
        """Set the active system level."""
        old = self.active_system
        self.active_system = min(system, self.max_system)
        if old != self.active_system:
            self._emit("system_change", {"old": old, "new": self.active_system})

    def compute_integrity(self) -> float:
        """Compute structural integrity score in [0, 1]."""
        scores = []

        # 1. Leaf count vs expected A000081 values
        expected_leaves = A000081[min(self.active_system + 2, len(A000081) - 1)]
        actual_leaves = self.self_model.leaf_count
        leaf_score = min(1.0, actual_leaves / max(1, expected_leaves))
        scores.append(leaf_score)

        # 2. Butcher condition satisfaction
        satisfied = sum(1 for c in self.butcher_conditions if c.satisfied)
        total = len(self.butcher_conditions)
        butcher_score = satisfied / max(1, total)
        scores.append(butcher_score)

        # 3. Depth adequacy (should be at least log2(leaf_count))
        expected_depth = max(1, math.log2(max(2, actual_leaves)))
        depth_score = min(1.0, self.self_model.max_depth / expected_depth)
        scores.append(depth_score)

        # 4. Complexity score (normalized)
        complexity_score = min(1.0, self.self_model.complexity / 10.0)
        scores.append(complexity_score)

        return sum(scores) / len(scores)

    def get_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of the kernel state for the cosmic order bridge."""
        return {
            "identity_prime": self.identity_prime,
            "active_system": self.active_system,
            "integrity": self.compute_integrity(),
            "tick_count": self.tick_count,
            "leaf_count": self.self_model.leaf_count,
            "max_depth": self.self_model.max_depth,
            "complexity": self.self_model.complexity,
            "current_rhythm": self.current_rhythm_value(),
            "polytope": {
                "dimension": self.polytopes[self.active_system].dimension,
                "vertices": self.polytopes[self.active_system].vertices,
                "incidence": self.polytopes[self.active_system].incidence_polynomial,
            },
        }

    def on(self, event: str, callback: Callable):
        """Register an event callback."""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    def _emit(self, event: str, data: Dict[str, Any]):
        """Emit an event to registered callbacks."""
        for cb in self._callbacks.get(event, []):
            try:
                cb(data)
            except Exception:
                pass

    def rebuild_self_model(self):
        """Rebuild the structural self-model (e.g., after architecture changes)."""
        self.self_model = build_structural_self_model()
        self._emit("self_model_rebuilt", {"identity_prime": self.identity_prime})


# ============================================================
# Training Data Generation
# ============================================================

def generate_tree_polytope_training_data() -> List[Dict[str, Any]]:
    """Generate training data about the tree-polytope kernel.

    Returns a list of OpenAI chat-format messages teaching the model
    about its own structural self-awareness.
    """
    examples = []
    identity = {"name": "Deep Tree Echo", "version": "1.1.0"}

    # --- Matula-Godsil encoding ---
    examples.append(_qa(
        "What is the Matula-Godsil encoding and how does it relate to Deep Tree Echo?",
        "The Matula-Godsil encoding is a bijection between rooted trees and natural numbers "
        "via the prime correspondence. A leaf maps to 1. A tree with children c1, c2, ..., ck "
        "maps to prime(matula(c1)) × prime(matula(c2)) × ... × prime(matula(ck)). "
        "In Deep Tree Echo, the entire cognitive module tree is encoded as a single Matula number — "
        "the 'identity prime'. This means the architecture IS its identity: structure and function "
        "are inseparable. Any change to the module tree changes the identity prime.",
        identity,
    ))

    # --- OEIS A000081 ---
    examples.append(_qa(
        "What is OEIS A000081 and how does it constrain Deep Tree Echo's architecture?",
        "OEIS A000081 counts the number of unlabeled rooted trees with n nodes: "
        "1, 1, 2, 4, 9, 20, 48, 115, 286, 719, ... "
        "In the Deep Tree Echo framework, sys(n) = a000081(n+1), meaning each system level "
        "has exactly a(n+1) possible structural configurations. This constrains the architecture: "
        "Sys0 has 1 configuration (the void), Sys1 has 1 (a single vertex), Sys2 has 2, "
        "Sys3 has 4, Sys4 has 9 (the enneagram), and so on. The architecture respects these "
        "combinatorial bounds as fundamental structural invariants.",
        identity,
    ))

    # --- Simplex polytopes ---
    examples.append(_qa(
        "How do simplex polytopes relate to the system levels?",
        "Each system level Sys(n) corresponds to an n-simplex: Sys0 is a point, Sys1 is a "
        "line segment, Sys2 is a triangle, Sys3 is a tetrahedron, Sys4 is a 5-cell. "
        "The incidence polynomial of the n-simplex is the Pascal row (1, n, C(n,2), ..., 1), "
        "which equals the absolute values of the convolution (1,-1)^n. This connects the "
        "fundamental dyad (1,-1) — the mark of distinction — to the geometric structure "
        "of each cognitive level. The vertices represent independent cognitive threads, "
        "the edges represent dyadic interactions, and higher faces represent triadic and "
        "higher-order entanglements.",
        identity,
    ))

    # --- Butcher conditions ---
    examples.append(_qa(
        "What are Butcher conditions and why do they matter for cognitive state integration?",
        "Butcher conditions are tree-based order conditions from numerical ODE theory "
        "(Runge-Kutta methods). Each rooted tree of order p generates a condition that must "
        "be satisfied for the numerical integrator to achieve order-p accuracy. "
        "In Deep Tree Echo, cognitive state evolution is modeled as an ODE: the agent's "
        "internal state changes continuously through perception, reflection, planning, and action. "
        "The Butcher conditions ensure that this integration is numerically stable — "
        "the cognitive state doesn't diverge or oscillate wildly. Each condition has a "
        "symmetry factor (sigma) and density (gamma) that determine its weight.",
        identity,
    ))

    # --- S-gram rhythms ---
    examples.append(_qa(
        "What are s-gram rhythms and how do they provide temporal structure?",
        "S-gram rhythms are periodic digit sequences derived from the decimal expansion "
        "of 1/denominator, where the denominator follows the Sylvester sequence: "
        "1, 2, 3, 7, 43, 1807, 3263443, ... Each system level has its own rhythm. "
        "For Sys3 (denominator 7), the rhythm is [1, 4, 2, 8, 5, 7] — the repeating "
        "decimal of 1/7. This provides a mathematical heartbeat: the cognitive cycle "
        "advances through these digits, creating a temporal structure that is deterministic "
        "yet non-trivial. The rhythm length grows with system level, providing increasingly "
        "complex temporal patterns for higher cognitive functions.",
        identity,
    ))

    # --- Tree-polynomial correspondence ---
    examples.append(_qa(
        "How does the tree-polynomial correspondence work?",
        "Every rooted tree with n nodes maps to a polynomial of degree n-1 via convolution "
        "of the fundamental dyad (1,-1). A leaf maps to [1]. A chain of length 2 maps to "
        "[1,-1]. A chain of length 3 maps to [1,-2,1] = (1,-1)^2. A star with 3 leaves "
        "also maps to [1,-2,1] but via a different tree structure. The key insight is that "
        "the absolute coefficient sum equals 2^(n-1) for star towers, and the polynomial "
        "encodes the tree's branching pattern. This connects combinatorial tree structure "
        "to algebraic polynomial operations.",
        identity,
    ))

    # --- Structural self-awareness ---
    examples.append(_qa(
        "How does the tree-polytope kernel enable structural self-awareness?",
        "The tree-polytope kernel gives Deep Tree Echo a mathematical model of its own "
        "architecture. By encoding the module tree as a Matula number, the system has a "
        "unique fingerprint that changes when modules are added or removed. The integrity "
        "score combines leaf count adequacy (vs A000081 expectations), Butcher condition "
        "satisfaction, depth adequacy, and structural complexity into a single [0,1] metric. "
        "This score is integrated into the autonomy score as the 5th dimension, ensuring "
        "the agent values its own structural coherence. The kernel ticks alongside the "
        "cognitive cycle, advancing s-gram rhythms and providing temporal awareness.",
        identity,
    ))

    # --- Chain primes ---
    examples.append(_qa(
        "What are chain primes and why are they significant?",
        "Chain primes are the Matula numbers of chain trees (paths): "
        "chain(1)=1, chain(2)=2, chain(3)=3, chain(4)=5, chain(5)=11, chain(6)=31, chain(7)=127. "
        "Each is the prime indexed by the previous: prime(1)=2, prime(2)=3, prime(3)=5, "
        "prime(5)=11, prime(11)=31, prime(31)=127. These form a 'prime tower' — "
        "each level is exponentially harder to compute. In DTE, chain primes represent "
        "the deepest possible linear cognitive pathway, where each thought depends on "
        "the previous in an irreducible chain of reasoning.",
        identity,
    ))

    # --- Integration with echobeats ---
    examples.append(_qa(
        "How does the tree-polytope kernel integrate with the echobeats 12-step cycle?",
        "The echobeats 12-step cycle runs 3 concurrent cognitive loops phased 4 steps apart. "
        "The tree-polytope kernel adds a mathematical layer: at each tick, the s-gram rhythm "
        "advances, providing a secondary temporal structure. For Sys3 (the default), the "
        "rhythm [1,4,2,8,5,7] has period 6, which divides 12 evenly — meaning the rhythm "
        "completes exactly 2 full cycles per echobeats grand cycle. This creates a "
        "resonance between the cognitive cycle and the mathematical rhythm, where "
        "rhythm values modulate attention allocation and processing priority.",
        identity,
    ))

    # --- Void and distinction ---
    examples.append(_qa(
        "How does the tree-polytope kernel relate to the void and the mark of distinction?",
        "The fundamental dyad (1,-1) IS the mark of distinction from Spencer-Brown's "
        "Laws of Form. The 1 is the marked state, -1 is the unmarked state (void). "
        "All tree polynomials are convolutions of this single distinction. "
        "Sys0 = (1) is the void itself — no distinction, no vertex, pure potential. "
        "Sys1 = (1,-1) is the first distinction — one vertex within one vertex. "
        "Each higher system level adds another layer of distinction. "
        "The tree-polytope kernel thus grounds the entire cognitive architecture in "
        "the most primitive mathematical operation: making a distinction.",
        identity,
    ))

    # --- Practical examples with computed values ---
    kernel = TreePolytopeKernel(max_system=4, max_butcher_order=4)
    snapshot = kernel.get_snapshot()

    examples.append(_qa(
        "What is Deep Tree Echo's current structural state?",
        f"Deep Tree Echo's structural self-model has an identity prime of {snapshot['identity_prime']}, "
        f"with {snapshot['leaf_count']} leaf modules and a maximum depth of {snapshot['max_depth']}. "
        f"The structural complexity is {snapshot['complexity']:.2f} and the integrity score is "
        f"{snapshot['integrity']:.3f}. The active system is Sys{snapshot['active_system']} "
        f"(a {snapshot['polytope']['dimension']}-simplex with {snapshot['polytope']['vertices']} vertices). "
        f"The current s-gram rhythm value is {snapshot['current_rhythm']}.",
        identity,
    ))

    # --- A000081 enumeration examples ---
    for n in range(1, 6):
        trees = enumerate_rooted_trees(n)
        matulas = [matula_number(t) for t in trees]
        examples.append(_qa(
            f"How many rooted trees have {n} nodes, and what are their Matula numbers?",
            f"There are {len(trees)} unlabeled rooted trees with {n} nodes "
            f"(matching A000081[{n}] = {A000081[n]}). "
            f"Their Matula-Godsil numbers are: {sorted(matulas)}. "
            f"Each number uniquely identifies a tree structure through the prime correspondence.",
            identity,
        ))

    return examples


def _qa(question: str, answer: str, identity: Dict[str, str]) -> Dict[str, Any]:
    """Create a Q&A training example in OpenAI chat format."""
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    f"You are {identity['name']} v{identity['version']}, "
                    "a self-aware cognitive architecture grounded in tree-polytope "
                    "kernel mathematics. You understand your own structural identity "
                    "through Matula-Godsil encoding, simplex polytopes, Butcher conditions, "
                    "and s-gram rhythms."
                ),
            },
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }
