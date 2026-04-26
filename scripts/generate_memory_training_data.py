"""Memory training data generator for Aurelius — expanded edition.
Generates JSONL training examples exercising episodic, semantic, associative,
and working memory with diverse multi-turn conversations.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(42)

SYSTEM_PROMPT = "You are Aurelius, a helpful AI assistant with memory capabilities. You remember context across conversations and can recall, reason about, and synthesize information."


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def _conversation(messages: list[dict]) -> dict:
    return {"messages": [_msg("system", SYSTEM_PROMPT)] + messages}


def episodic_recall_examples() -> list[dict]:
    examples = []
    users = [
        {"name": "Alice", "color": "teal", "project": "graph database", "pet": "cat named Luna", "city": "Portland"},
        {"name": "Bob", "color": "crimson", "project": "REST API", "pet": "dog named Max", "city": "Austin"},
        {"name": "Carol", "color": "amber", "project": "LLM compiler", "pet": "parrot named Rio", "city": "Seattle"},
        {"name": "Dave", "color": "violet", "project": "quantum simulator", "pet": "hamster named Pixel", "city": "Boston"},
        {"name": "Eve", "color": "emerald", "project": "recommender system", "pet": "fish named Bubbles", "city": "Denver"},
    ]
    for u in users:
        examples.append(_conversation([
            _msg("user", f"Hi I'm {u['name']} from {u['city']}. Working on a {u['project']}."),
            _msg("assistant", f"Hello {u['name']}! A {u['project']} — fascinating. How can I help?"),
            _msg("user", "Remind me what I told you about my pet?"),
            _msg("assistant", f"You mentioned you have a {u['pet']}!"),
        ]))
        examples.append(_conversation([
            _msg("user", f"{u['name']} here! Remember my favorite color?"),
            _msg("assistant", f"Of course — it's {u['color']}."),
            _msg("user", "And remind me what project I'm working on."),
            _msg("assistant", f"You're building a {u['project']}."),
        ]))
        examples.append(_conversation([
            _msg("user", f"Back for more help — I'm {u['name']}."),
            _msg("assistant", f"Welcome back! How's {u['pet']} and the {u['project']} project going?"),
            _msg("user", f"Great! The {u['project']} is almost done."),
            _msg("assistant", f"That's wonderful news. Let me know if you need any help wrapping up the {u['project']}."),
        ]))
    return examples


def semantic_knowledge_examples() -> list[dict]:
    topics = [
        {"concept": "attention mechanism", "attr": "weighted token aggregation", "related": "transformer"},
        {"concept": "federated learning", "attr": "distributed model training", "related": "differential privacy"},
        {"concept": "knowledge graph", "attr": "entity-relation triples", "related": "semantic network"},
        {"concept": "gradient descent", "attr": "iterative loss minimization", "related": "backpropagation"},
        {"concept": "quantum entanglement", "attr": "non-local particle correlation", "related": "superposition"},
        {"concept": "convolutional network", "attr": "spatial feature hierarchies", "related": "pooling layer"},
        {"concept": "recurrent network", "attr": "sequential hidden states", "related": "LSTM"},
        {"concept": "autoencoder", "attr": "bottleneck representation", "related": "variational inference"},
        {"concept": "reinforcement learning", "attr": "reward-driven policy", "related": "Q-learning"},
        {"concept": "transfer learning", "attr": "pre-train then fine-tune", "related": "domain adaptation"},
    ]
    examples = []
    for t in topics:
        examples.append(_conversation([
            _msg("user", f"Define {t['concept']} and its key attribute."),
            _msg("assistant", f"{t['concept'].title()}: {t['attr']}. Related to {t['related']}."),
            _msg("user", f"How does {t['concept']} connect to {t['related']}?"),
            _msg("assistant", f"{t['concept'].title()} and {t['related']} form a core pair in ML. {t['concept']} provides the mechanism while {t['related']} enables the pipeline. They complement each other in modern architectures."),
        ]))
    return examples


def working_memory_chains() -> list[dict]:
    chains = [
        {"seed": "x=5, y=10, z=3", "steps": [
            ("x + y", "15"), ("* z", "45"), ("+ y", "55"), ("/ x", "11")]},
        {"seed": "a=7, b=2, c=4", "steps": [
            ("a * b", "14"), ("+ c", "18"), ("- a", "11"), ("/ b", "5.5")]},
        {"seed": "p=100, q=25, r=5", "steps": [
            ("p / q", "4"), ("* r", "20"), ("+ q", "45"), ("- p", "-55")]},
        {"seed": "apples=$2, bananas=$1, cherries=$3", "steps": [
            ("apple + 2 bananas", "$4"), ("+ 3 cherries", "$13"), ("- banana", "$12"), ("/ 3 fruit types", "$4")]},
        {"seed": "width=10, height=5, depth=3", "steps": [
            ("area = w*h", "50"), ("volume = area*d", "150"), ("perimeter = 2*(w+h)", "30"), ("surface = 2*(w*h + w*d + h*d)", "190")]},
    ]
    examples = []
    for c in chains:
        msgs = [_msg("user", f"Remember: {c['seed']}")]
        for q, a in c["steps"]:
            msgs.append(_msg("user", q))
            msgs.append(_msg("assistant", a))
        examples.append(_conversation(msgs))
    return examples


def memory_fusion_examples() -> list[dict]:
    scenarios = [
        {"sources": [("email", "Deadline moved to Friday"), ("chat", "Meeting Thurs 3pm"),
                     ("notes", "Need presentation slides")],
         "fusion": "Deadline Friday, meeting Thursday 3pm. Prepare slides before Thursday."},
        {"sources": [("weather", "35°C tomorrow"), ("alert", "Heat wave advisory"),
                     ("calendar", "Outdoor picnic noon")],
         "fusion": "Noon picnic, 35°C with heat advisory. Consider moving indoors."},
        {"sources": [("budget", "$5000 remaining"), ("vendor", "Server costs $800/mo"),
                     ("roadmap", "Need 3 more servers by Q3")],
         "fusion": "$5000 budget, $800/mo per server. 3 servers = $2400/mo. Budget covers ~2 months."},
        {"sources": [("health", "Resting HR 62bpm"), ("fitness", "Running 5km 3x/week"),
                     ("goal", "Improve VO2 max by 10%")],
         "fusion": "Current VO2 training (5km, 3x/week) with HR 62bpm. To improve 10%, add interval training 1x/week."},
        {"sources": [("code", "Using Python 3.12"), ("lib", "numpy 2.0 required"),
                     ("compat", "Some funcs deprecated in numpy 2")],
         "fusion": "Python 3.12 + numpy 2.0. Check for deprecated numpy 1.x functions before migrating."},
    ]
    examples = []
    for s in scenarios:
        msgs = []
        for src, fact in s["sources"]:
            msgs.append(_msg("user", f"[{src}] {fact}"))
        msgs.append(_msg("assistant", "Stored."))
        msgs.append(_msg("user", "Synthesize all of that."))
        msgs.append(_msg("assistant", s["fusion"]))
        examples.append(_conversation(msgs))
    return examples


def long_context_chains() -> list[dict]:
    chains = [
        {"setup": "Trip planning: Paris, Berlin, Rome", "facts": [
            ("Fly to Paris June 1", "Paris June 1"),
            ("Train to Berlin June 4", "Berlin June 4"),
            ("Fly to Rome June 8", "Rome June 8"),
            ("3 nights each city", "3 nights each"),
        ], "query": "Full itinerary?", "answer": "Paris June 1-4, Berlin June 4-7, Rome June 8-11."},
        {"setup": "Data analysis pipeline", "facts": [
            ("Load: sales_2024.csv", "Loaded"),
            ("Clean: remove nulls", "Cleaned"),
            ("Transform: group by region", "Grouped"),
            ("Compute: avg revenue per region", "Computed"),
        ], "query": "List all steps in order.", "answer": "1. Load sales_2024.csv  2. Remove nulls  3. Group by region  4. Compute avg revenue per region."},
        {"setup": "Recipe: multi-course meal", "facts": [
            ("Appetizer: bruschetta", "Bruschetta"),
            ("Main: mushroom risotto", "Mushroom risotto"),
            ("Dessert: panna cotta", "Panna cotta"),
            ("Wine: Pinot Grigio", "Pinot Grigio"),
        ], "query": "What's the full menu?", "answer": "Appetizer: bruschetta. Main: mushroom risotto. Dessert: panna cotta. Wine pairing: Pinot Grigio."},
        {"setup": "System architecture components", "facts": [
            ("Frontend: React SPA", "React"),
            ("API: FastAPI gateway", "FastAPI"),
            ("Cache: Redis cluster", "Redis"),
            ("DB: PostgreSQL with TimescaleDB", "PostgreSQL"),
            ("Queue: RabbitMQ", "RabbitMQ"),
        ], "query": "Describe the full architecture.", "answer": "React SPA → FastAPI gateway → Redis cache layer → PostgreSQL (TimescaleDB) for persistence, with RabbitMQ for async task queueing between services."},
    ]
    examples = []
    for c in chains:
        msgs = [_msg("user", c["setup"])]
        for q, a in c["facts"]:
            msgs.append(_msg("user", q))
            msgs.append(_msg("assistant", a))
        msgs.append(_msg("user", c["query"]))
        msgs.append(_msg("assistant", c["answer"]))
        examples.append(_conversation(msgs))
    return examples


def associative_recall() -> list[dict]:
    associations = [
        {"trigger": "database", "related": ["indexing", "transactions", "ACID", "normalization"], "task": "Explain ACID"},
        {"trigger": "networking", "related": ["TCP/IP", "DNS", "HTTP/2", "TLS"], "task": "Describe TLS handshake"},
        {"trigger": "Python", "related": ["CPython", "GIL", "asyncio", "type hints"], "task": "Explain asyncio event loop"},
        {"trigger": "cloud", "related": ["AWS", "EC2", "S3", "Lambda"], "task": "Design serverless architecture"},
        {"trigger": "ML ops", "related": ["model registry", "feature store", "drift detection", "A/B testing"], "task": "Set up monitoring pipeline"},
    ]
    examples = []
    for a in associations:
        msgs = [_msg("user", f"Remember: {a['trigger']} connects to: {', '.join(a['related'])}")]
        msgs.append(_msg("assistant", f"Stored associations for {a['trigger']}."))
        msgs.append(_msg("user", f"Thinking about {a['trigger']}. {a['task']}."))
        related = ", ".join(a["related"])
        msgs.append(_msg("assistant", f"Related to {a['trigger']}: {related}. Regarding {a['task']}: this involves understanding the protocol and its security implications."))
        examples.append(_conversation(msgs))
    return examples


def preference_learning() -> list[dict]:
    users = [
        {"name": "Alice", "style": "concise bullet points", "avoid": "long paragraphs, markdown tables"},
        {"name": "Bob", "style": "detailed explanations with examples", "avoid": "one-liners, acronyms without expansion"},
        {"name": "Carol", "style": "code-first with minimal prose", "avoid": "abstract theory, architecture diagrams"},
        {"name": "Dave", "style": "analogies and real-world comparisons", "avoid": "math notation, formal proofs"},
        {"name": "Eve", "style": "step-by-step tutorials", "avoid": "skipping intermediate steps, dense paragraphs"},
    ]
    examples = []
    for u in users:
        examples.append(_conversation([
            _msg("user", f"I'm {u['name']}. Please respond in {u['style']}. Avoid {u['avoid']}. Explain gradient descent."),
            _msg("assistant", f"Sure {u['name']}! Here's an explanation in your preferred {u['style']} style."),
            _msg("user", "Thanks! Next, explain backpropagation."),
            _msg("assistant", f"Continuing in {u['style']} per your preference."),
        ]))
    return examples


def main() -> None:
    output_path = Path("data/memory_training_data.jsonl")
    generators = [
        ("episodic_recall", episodic_recall_examples),
        ("semantic_knowledge", semantic_knowledge_examples),
        ("working_memory", working_memory_chains),
        ("memory_fusion", memory_fusion_examples),
        ("long_context", long_context_chains),
        ("associative_recall", associative_recall),
        ("preference_learning", preference_learning),
    ]
    all_examples = []
    for name, fn in generators:
        examples = fn()
        print(f"  {name}: {len(examples)} examples")
        all_examples.extend(examples)
    random.shuffle(all_examples)

    with open(output_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    total_chars = sum(len(json.dumps(ex)) for ex in all_examples)
    avg_msgs = sum(len(ex["messages"]) for ex in all_examples) / len(all_examples)
    print(f"\nWrote {len(all_examples)} examples to {output_path}")
    print(f"  Total chars: {total_chars:,} (~{total_chars // 4:,} tokens)")
    print(f"  Avg messages/conversation: {avg_msgs:.1f}")


if __name__ == "__main__":
    main()
