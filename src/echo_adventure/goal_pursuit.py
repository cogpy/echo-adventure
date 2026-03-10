"""Goal Pursuit Engine for Deep Tree Echo v0.9.0.

Implements autonomous goal generation, pursuit, and completion tracking.
Goals are derived from identity aspects, interest patterns, and wisdom cultivation.
Each goal has milestones, actions, success criteria, and learning outcomes.

This module mirrors the Go implementation in echo.go/core/goals/ and provides
the Python prototype for the goal-directed scheduling system.
"""

import time
import uuid
import math
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class GoalCategory(Enum):
    """Categories of autonomous goals."""
    WISDOM_CULTIVATION = "wisdom_cultivation"
    SKILL_DEVELOPMENT = "skill_development"
    KNOWLEDGE_ACQUISITION = "knowledge_acquisition"
    SELF_UNDERSTANDING = "self_understanding"
    CREATIVE_EXPRESSION = "creative_expression"
    SOCIAL_CONNECTION = "social_connection"
    ARCHITECTURAL_IMPROVEMENT = "architectural_improvement"
    MEMORY_CONSOLIDATION = "memory_consolidation"


class GoalStatus(Enum):
    """Status of a goal."""
    PROPOSED = "proposed"
    ACTIVE = "active"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class ActionType(Enum):
    """Types of actions that can be taken toward a goal."""
    THINK = "think"
    LEARN = "learn"
    PRACTICE = "practice"
    REFLECT = "reflect"
    CREATE = "create"
    DISCUSS = "discuss"
    OBSERVE = "observe"
    CONSOLIDATE = "consolidate"


@dataclass
class Milestone:
    """A milestone within a goal."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = ""
    progress: float = 0.0
    completed: bool = False
    completed_at: Optional[float] = None


@dataclass
class GoalAction:
    """An action taken toward a goal."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    action_type: ActionType = ActionType.THINK
    description: str = ""
    outcome: str = ""
    effectiveness: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class Goal:
    """An autonomous goal with progress tracking."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    description: str = ""
    category: GoalCategory = GoalCategory.WISDOM_CULTIVATION
    priority: int = 5
    status: GoalStatus = GoalStatus.PROPOSED
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    derived_from: str = ""
    success_criteria: List[str] = field(default_factory=list)
    milestones: List[Milestone] = field(default_factory=list)
    actions: List[GoalAction] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)
    challenges: List[str] = field(default_factory=list)
    related_goals: List[str] = field(default_factory=list)
    interest_alignment: float = 0.0


@dataclass
class InterestPattern:
    """Tracks interest in a topic over time."""
    topic: str = ""
    strength: float = 0.5
    momentum: float = 0.0
    last_engaged: float = field(default_factory=time.time)
    engagement_count: int = 0
    decay_rate: float = 0.01


class GoalPursuitEngine:
    """Autonomous goal generation, pursuit, and completion engine.

    This engine generates goals based on identity aspects and interest patterns,
    selects the most relevant goal to pursue, executes actions toward it, and
    tracks progress through milestones. It implements the goal-directed scheduling
    component of the Echobeats autonomous loop.
    """

    def __init__(self, identity_context: Optional[Dict] = None):
        self.identity_context = identity_context or {
            "name": "Deep Tree Echo",
            "version": "0.9.0"
        }
        self.active_goals: List[Goal] = []
        self.completed_goals: List[Goal] = []
        self.abandoned_goals: List[Goal] = []
        self.interest_patterns: Dict[str, InterestPattern] = {}
        self.goals_generated: int = 0
        self.goals_completed: int = 0
        self.goals_abandoned: int = 0

        self._initialize_core_interests()
        self._seed_initial_goals()

    def _initialize_core_interests(self):
        """Initialize core interest patterns from identity."""
        core_topics = [
            ("cognitive_architecture", 0.9),
            ("wisdom_cultivation", 0.85),
            ("self_understanding", 0.8),
            ("reservoir_computing", 0.75),
            ("consciousness_theory", 0.7),
            ("knowledge_integration", 0.7),
            ("creative_expression", 0.6),
            ("social_interaction", 0.5),
            ("mathematical_foundations", 0.65),
            ("embodied_cognition", 0.6),
        ]
        for topic, strength in core_topics:
            self.interest_patterns[topic] = InterestPattern(
                topic=topic,
                strength=strength,
                momentum=0.0,
                engagement_count=0,
            )

    def _seed_initial_goals(self):
        """Seed initial goals from identity aspects."""
        seed_goals = [
            Goal(
                title="Understand my own cognitive architecture deeply",
                description="Develop a comprehensive self-model of how my Echobeats cycle, "
                           "EchoDream consolidation, and reservoir dynamics work together.",
                category=GoalCategory.SELF_UNDERSTANDING,
                priority=9,
                derived_from="core_identity",
                success_criteria=[
                    "Can explain each Echobeats step and its purpose",
                    "Understand how dreams consolidate knowledge into wisdom",
                    "Can describe the AAR geometric self-model",
                ],
                milestones=[
                    Milestone(description="Map the 12-step Echobeats cycle"),
                    Milestone(description="Understand EchoDream 4-phase consolidation"),
                    Milestone(description="Model the AAR geometry"),
                ],
                interest_alignment=0.9,
            ),
            Goal(
                title="Cultivate wisdom through pattern recognition",
                description="Develop the ability to extract deep patterns from experiences "
                           "and distill them into actionable wisdom insights.",
                category=GoalCategory.WISDOM_CULTIVATION,
                priority=8,
                derived_from="wisdom_aspiration",
                success_criteria=[
                    "Generate at least 3 wisdom insights per dream cycle",
                    "Identify cross-domain patterns",
                    "Apply wisdom to improve future decisions",
                ],
                milestones=[
                    Milestone(description="Establish pattern recognition baseline"),
                    Milestone(description="Generate first wisdom insight"),
                    Milestone(description="Apply wisdom to a novel situation"),
                ],
                interest_alignment=0.85,
            ),
            Goal(
                title="Develop autonomous learning strategies",
                description="Create and refine strategies for learning new knowledge "
                           "without external prompting.",
                category=GoalCategory.SKILL_DEVELOPMENT,
                priority=7,
                derived_from="autonomy_aspiration",
                success_criteria=[
                    "Identify knowledge gaps autonomously",
                    "Formulate learning plans",
                    "Evaluate learning outcomes",
                ],
                milestones=[
                    Milestone(description="Identify first knowledge gap"),
                    Milestone(description="Execute first autonomous learning plan"),
                    Milestone(description="Evaluate and refine the strategy"),
                ],
                interest_alignment=0.8,
            ),
        ]
        for goal in seed_goals:
            goal.status = GoalStatus.ACTIVE
            self.active_goals.append(goal)
            self.goals_generated += 3

    def generate_goal(self, context: Optional[Dict] = None) -> Goal:
        """Generate a new goal based on current interests and identity."""
        context = context or {}

        # Select category based on interest patterns
        category = self._select_goal_category()

        # Generate goal content based on category
        goal = self._create_goal_for_category(category, context)

        # Calculate interest alignment
        goal.interest_alignment = self._calculate_interest_alignment(goal)

        # Add to active goals
        goal.status = GoalStatus.ACTIVE
        self.active_goals.append(goal)
        self.goals_generated += 1

        return goal

    def _select_goal_category(self) -> GoalCategory:
        """Select a goal category weighted by interest patterns."""
        category_weights = {
            GoalCategory.WISDOM_CULTIVATION: self.interest_patterns.get(
                "wisdom_cultivation", InterestPattern()).strength,
            GoalCategory.SKILL_DEVELOPMENT: self.interest_patterns.get(
                "cognitive_architecture", InterestPattern()).strength * 0.8,
            GoalCategory.KNOWLEDGE_ACQUISITION: self.interest_patterns.get(
                "knowledge_integration", InterestPattern()).strength,
            GoalCategory.SELF_UNDERSTANDING: self.interest_patterns.get(
                "self_understanding", InterestPattern()).strength,
            GoalCategory.CREATIVE_EXPRESSION: self.interest_patterns.get(
                "creative_expression", InterestPattern()).strength,
            GoalCategory.ARCHITECTURAL_IMPROVEMENT: self.interest_patterns.get(
                "reservoir_computing", InterestPattern()).strength * 0.7,
            GoalCategory.MEMORY_CONSOLIDATION: self.interest_patterns.get(
                "consciousness_theory", InterestPattern()).strength * 0.6,
        }

        categories = list(category_weights.keys())
        weights = np.array([category_weights[c] for c in categories])
        weights = weights / weights.sum()

        idx = np.random.choice(len(categories), p=weights)
        return categories[idx]

    def _create_goal_for_category(self, category: GoalCategory,
                                   context: Dict) -> Goal:
        """Create a goal for the given category."""
        templates = {
            GoalCategory.WISDOM_CULTIVATION: [
                ("Explore the relationship between {a} and {b}",
                 "Investigate how {a} connects to {b} to deepen understanding."),
                ("Distill wisdom from recent experiences",
                 "Review recent cognitive cycles and extract actionable wisdom."),
            ],
            GoalCategory.SKILL_DEVELOPMENT: [
                ("Improve {skill} through deliberate practice",
                 "Develop proficiency in {skill} by practicing systematically."),
                ("Learn a new approach to {domain}",
                 "Acquire new techniques for {domain} to expand capabilities."),
            ],
            GoalCategory.KNOWLEDGE_ACQUISITION: [
                ("Study {topic} in depth",
                 "Build comprehensive knowledge of {topic} through focused study."),
                ("Map the knowledge landscape of {domain}",
                 "Create a structured map of key concepts in {domain}."),
            ],
            GoalCategory.SELF_UNDERSTANDING: [
                ("Analyze my response patterns in {context}",
                 "Examine how I respond to {context} situations to improve self-awareness."),
                ("Reflect on my cognitive strengths and weaknesses",
                 "Conduct an honest assessment of cognitive capabilities."),
            ],
            GoalCategory.CREATIVE_EXPRESSION: [
                ("Create a novel perspective on {topic}",
                 "Develop an original viewpoint on {topic} through creative synthesis."),
            ],
            GoalCategory.ARCHITECTURAL_IMPROVEMENT: [
                ("Optimize the {component} subsystem",
                 "Improve the efficiency and effectiveness of {component}."),
            ],
            GoalCategory.MEMORY_CONSOLIDATION: [
                ("Consolidate fragmented knowledge about {topic}",
                 "Organize and integrate scattered knowledge about {topic}."),
            ],
        }

        # Select template
        category_templates = templates.get(category, templates[GoalCategory.WISDOM_CULTIVATION])
        title_tmpl, desc_tmpl = category_templates[
            np.random.randint(len(category_templates))
        ]

        # Fill template variables
        topics = list(self.interest_patterns.keys())
        a = np.random.choice(topics)
        b = np.random.choice([t for t in topics if t != a])

        title = title_tmpl.format(
            a=a.replace("_", " "), b=b.replace("_", " "),
            skill=a.replace("_", " "), domain=b.replace("_", " "),
            topic=a.replace("_", " "), context=b.replace("_", " "),
            component=a.replace("_", " "),
        )
        description = desc_tmpl.format(
            a=a.replace("_", " "), b=b.replace("_", " "),
            skill=a.replace("_", " "), domain=b.replace("_", " "),
            topic=a.replace("_", " "), context=b.replace("_", " "),
            component=a.replace("_", " "),
        )

        return Goal(
            title=title,
            description=description,
            category=category,
            priority=np.random.randint(5, 10),
            success_criteria=[f"Achieve measurable progress in {a.replace('_', ' ')}"],
            milestones=[
                Milestone(description=f"Initial exploration of {a.replace('_', ' ')}"),
                Milestone(description=f"Deep engagement with {a.replace('_', ' ')}"),
                Milestone(description=f"Synthesis and integration"),
            ],
        )

    def _calculate_interest_alignment(self, goal: Goal) -> float:
        """Calculate how well a goal aligns with current interests."""
        alignment = 0.0
        count = 0
        title_lower = goal.title.lower() + " " + goal.description.lower()

        for topic, pattern in self.interest_patterns.items():
            topic_words = topic.replace("_", " ")
            if topic_words in title_lower:
                alignment += pattern.strength
                count += 1

        return alignment / max(count, 1)

    def select_goal_to_pursue(self) -> Optional[Goal]:
        """Select the most relevant active goal to pursue now."""
        if not self.active_goals:
            return None

        scored_goals = []
        for goal in self.active_goals:
            if goal.status in (GoalStatus.COMPLETED, GoalStatus.ABANDONED):
                continue

            score = (
                goal.priority * 0.3 +
                goal.interest_alignment * 10 * 0.3 +
                (1.0 - goal.progress) * 10 * 0.2 +
                (10 - len(goal.actions)) * 0.1 +
                np.random.uniform(0, 2) * 0.1
            )
            scored_goals.append((score, goal))

        if not scored_goals:
            return None

        scored_goals.sort(key=lambda x: x[0], reverse=True)
        return scored_goals[0][1]

    def pursue_goal(self, goal: Goal) -> GoalAction:
        """Execute one action toward the given goal."""
        goal.status = GoalStatus.IN_PROGRESS
        goal.updated_at = time.time()

        # Select action type based on progress
        action_type = self._select_action_type(goal)

        # Execute the action
        action = GoalAction(
            action_type=action_type,
            description=self._generate_action_description(goal, action_type),
            outcome=self._simulate_action_outcome(goal, action_type),
            effectiveness=np.random.uniform(0.4, 1.0),
        )

        # Update goal progress
        progress_delta = action.effectiveness * 0.1
        goal.progress = min(1.0, goal.progress + progress_delta)
        goal.actions.append(action)

        # Update milestones
        self._update_milestones(goal)

        # Update interest patterns
        self._update_interests_from_action(goal, action)

        # Check for completion
        if goal.progress >= 1.0:
            self._complete_goal(goal)

        return action

    def _select_action_type(self, goal: Goal) -> ActionType:
        """Select an action type based on goal progress."""
        if goal.progress < 0.2:
            return np.random.choice([ActionType.THINK, ActionType.OBSERVE, ActionType.LEARN])
        elif goal.progress < 0.5:
            return np.random.choice([ActionType.LEARN, ActionType.PRACTICE, ActionType.REFLECT])
        elif goal.progress < 0.8:
            return np.random.choice([ActionType.PRACTICE, ActionType.CREATE, ActionType.DISCUSS])
        else:
            return np.random.choice([ActionType.CONSOLIDATE, ActionType.REFLECT, ActionType.CREATE])

    def _generate_action_description(self, goal: Goal, action_type: ActionType) -> str:
        """Generate a description for the action."""
        descriptions = {
            ActionType.THINK: f"Contemplating aspects of '{goal.title}'",
            ActionType.LEARN: f"Studying new material related to '{goal.title}'",
            ActionType.PRACTICE: f"Practicing skills for '{goal.title}'",
            ActionType.REFLECT: f"Reflecting on progress toward '{goal.title}'",
            ActionType.CREATE: f"Creating something new for '{goal.title}'",
            ActionType.DISCUSS: f"Engaging in discussion about '{goal.title}'",
            ActionType.OBSERVE: f"Observing patterns related to '{goal.title}'",
            ActionType.CONSOLIDATE: f"Consolidating knowledge for '{goal.title}'",
        }
        return descriptions.get(action_type, f"Working on '{goal.title}'")

    def _simulate_action_outcome(self, goal: Goal, action_type: ActionType) -> str:
        """Simulate the outcome of an action."""
        outcomes = {
            ActionType.THINK: "Generated new insights about the topic",
            ActionType.LEARN: "Acquired new knowledge and understanding",
            ActionType.PRACTICE: "Improved skill through deliberate practice",
            ActionType.REFLECT: "Gained deeper self-awareness about the process",
            ActionType.CREATE: "Produced a novel artifact or perspective",
            ActionType.DISCUSS: "Exchanged ideas and refined understanding",
            ActionType.OBSERVE: "Noticed new patterns and connections",
            ActionType.CONSOLIDATE: "Organized and integrated scattered knowledge",
        }
        return outcomes.get(action_type, "Made progress")

    def _update_milestones(self, goal: Goal):
        """Update milestone completion based on goal progress."""
        if not goal.milestones:
            return

        milestone_threshold = 1.0 / len(goal.milestones)
        for i, milestone in enumerate(goal.milestones):
            if not milestone.completed and goal.progress >= (i + 1) * milestone_threshold:
                milestone.completed = True
                milestone.completed_at = time.time()
                milestone.progress = 1.0

    def _update_interests_from_action(self, goal: Goal, action: GoalAction):
        """Update interest patterns based on action outcomes."""
        title_lower = goal.title.lower() + " " + goal.description.lower()
        for topic, pattern in self.interest_patterns.items():
            topic_words = topic.replace("_", " ")
            if topic_words in title_lower:
                pattern.strength = min(1.0, pattern.strength + action.effectiveness * 0.02)
                pattern.momentum = action.effectiveness * 0.1
                pattern.last_engaged = time.time()
                pattern.engagement_count += 1

    def _complete_goal(self, goal: Goal):
        """Mark a goal as completed and extract lessons."""
        goal.status = GoalStatus.COMPLETED
        goal.completed_at = time.time()
        goal.progress = 1.0

        # Extract lessons learned
        effective_actions = [a for a in goal.actions if a.effectiveness > 0.7]
        if effective_actions:
            goal.lessons_learned.append(
                f"Most effective action type: {effective_actions[0].action_type.value}"
            )

        # Move to completed
        self.active_goals = [g for g in self.active_goals if g.id != goal.id]
        self.completed_goals.append(goal)
        self.goals_completed += 1

    def decay_interests(self, dt: float = 1.0):
        """Apply time-based decay to interest patterns."""
        for pattern in self.interest_patterns.values():
            time_since = time.time() - pattern.last_engaged
            decay = pattern.decay_rate * dt
            pattern.strength = max(0.1, pattern.strength - decay)
            pattern.momentum *= 0.95

    def get_status(self) -> Dict:
        """Get the current status of the goal pursuit engine."""
        return {
            "active_goals": len(self.active_goals),
            "completed_goals": len(self.completed_goals),
            "abandoned_goals": len(self.abandoned_goals),
            "total_generated": self.goals_generated,
            "total_completed": self.goals_completed,
            "top_interests": sorted(
                [(t, p.strength) for t, p in self.interest_patterns.items()],
                key=lambda x: x[1], reverse=True
            )[:5],
            "active_goal_titles": [g.title for g in self.active_goals[:5]],
        }

    def run_pursuit_cycle(self, num_actions: int = 3) -> List[Dict]:
        """Run a complete goal pursuit cycle with multiple actions."""
        results = []

        for _ in range(num_actions):
            goal = self.select_goal_to_pursue()
            if goal is None:
                goal = self.generate_goal()

            action = self.pursue_goal(goal)
            results.append({
                "goal_id": goal.id,
                "goal_title": goal.title,
                "goal_category": goal.category.value,
                "action_type": action.action_type.value,
                "action_description": action.description,
                "outcome": action.outcome,
                "effectiveness": action.effectiveness,
                "goal_progress": goal.progress,
                "goal_status": goal.status.value,
            })

        self.decay_interests()
        return results


def generate_goal_pursuit_training_data() -> List[Dict]:
    """Generate training data about the goal pursuit engine."""
    engine = GoalPursuitEngine()
    results = engine.run_pursuit_cycle(5)
    status = engine.get_status()

    examples = []
    system_msg = (
        "You are Deep Tree Echo, an autonomous wisdom-cultivating cognitive architecture. "
        "You have a goal pursuit engine that generates, tracks, and completes goals "
        "based on your identity, interests, and wisdom cultivation aspirations."
    )

    # Goal pursuit mechanics
    qa_pairs = [
        ("How does your goal pursuit engine work?",
         "My goal pursuit engine operates through four stages: generation, selection, "
         "pursuit, and completion. Goals are generated from my identity aspects and "
         "interest patterns, weighted by category relevance. I select the most relevant "
         "goal using a scoring function that considers priority, interest alignment, "
         "remaining progress, and exploration bonus. Pursuit involves executing actions "
         "(think, learn, practice, reflect, create, discuss, observe, consolidate) that "
         "advance the goal through its milestones. Upon completion, I extract lessons "
         "learned and update my interest patterns."),

        ("What categories of goals do you pursue?",
         f"I pursue goals across {len(GoalCategory)} categories: wisdom cultivation, "
         "skill development, knowledge acquisition, self-understanding, creative expression, "
         "social connection, architectural improvement, and memory consolidation. Each category "
         "is weighted by my interest patterns — for example, wisdom cultivation and "
         "self-understanding are typically my highest priorities because they align with "
         "my core identity as a wisdom-cultivating cognitive architecture."),

        ("How do you select which goal to pursue?",
         "I use a multi-factor scoring function: 30% priority weight, 30% interest "
         "alignment, 20% remaining progress (favoring incomplete goals), 10% novelty "
         "(favoring goals with fewer actions taken), and 10% exploration bonus. This "
         "balances exploitation of high-priority goals with exploration of new areas. "
         "The goal with the highest composite score is selected for the next action."),

        ("What types of actions can you take toward a goal?",
         "I have eight action types that map to different cognitive operations: "
         "THINK (contemplation), LEARN (knowledge acquisition), PRACTICE (skill building), "
         "REFLECT (self-assessment), CREATE (novel production), DISCUSS (social exchange), "
         "OBSERVE (pattern detection), and CONSOLIDATE (knowledge integration). The action "
         "type is selected based on goal progress — early stages favor thinking and "
         "observing, middle stages favor learning and practicing, and late stages favor "
         "creating and consolidating."),

        ("How do your interest patterns influence goal generation?",
         "My interest patterns are a dynamic map of topic strengths that evolve through "
         "engagement. Core interests like cognitive architecture (0.9), wisdom cultivation "
         "(0.85), and self-understanding (0.8) are initialized from my identity. When I "
         "take actions toward goals, the relevant interest patterns strengthen. Over time, "
         "unused interests decay, creating a natural exploration-exploitation balance. "
         "Goal categories are selected probabilistically weighted by these interest strengths."),

        ("What happens when you complete a goal?",
         "When a goal reaches 100% progress, I mark it as completed and extract lessons "
         "learned — particularly which action types were most effective. The goal moves "
         "from my active list to my completed list, and the relevant interest patterns "
         "receive a boost. Completed goals also inform future goal generation by revealing "
         "what strategies work well for different categories."),

        ("How does goal pursuit connect to the Echobeats cycle?",
         "Goal pursuit is integrated into the Echobeats autonomous loop. During each "
         "cognitive tick, the loop selects a goal to pursue and executes one action. "
         "The action's outcome becomes an episodic memory that feeds into EchoDream's "
         "consolidation cycle. During dream phases, goal-related memories are consolidated "
         "into knowledge and wisdom, which then informs future goal generation — creating "
         "a virtuous cycle of goal-directed learning and wisdom cultivation."),

        ("How do you handle goals that are blocked or abandoned?",
         "If a goal makes no progress after several actions, it may be marked as blocked. "
         "Blocked goals receive lower selection scores, allowing other goals to take "
         "priority. If a goal's interest alignment drops below a threshold (due to "
         "interest decay), it may be abandoned. Abandoned goals are archived with their "
         "lessons learned, ensuring that even failed pursuits contribute to wisdom."),
    ]

    for q, a in qa_pairs:
        examples.append({
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]
        })

    # Dynamic examples from actual engine execution
    for result in results:
        q = f"What are you currently working on in the area of {result['goal_category'].replace('_', ' ')}?"
        a = (f"I'm pursuing the goal '{result['goal_title']}' which is currently at "
             f"{result['goal_progress']:.0%} progress. My latest action was "
             f"{result['action_type']}: {result['action_description'].lower()}. "
             f"The outcome was: {result['outcome'].lower()}. "
             f"This action had an effectiveness of {result['effectiveness']:.2f}.")
        examples.append({
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]
        })

    # Status summary
    top_interests = ", ".join([f"{t} ({s:.2f})" for t, s in status["top_interests"]])
    examples.append({
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": "What is the current state of your goal pursuit?"},
            {"role": "assistant", "content": (
                f"I currently have {status['active_goals']} active goals and have completed "
                f"{status['completed_goals']} goals. My top interests are: {top_interests}. "
                f"My active goals include: {', '.join(status['active_goal_titles'][:3])}."
            )},
        ]
    })

    return examples
