"""
Unit test for the OUTPUT guardrail in isolation (no server, no input classifier).
Tests toxicity, schema validation, and consistency scoring directly.

Run from project root: python test_output.py
"""

from app.guardrails.output_layer import OutputGuardrail

guard = OutputGuardrail()
print(f"Toxicity model device: {guard.detox_device or 'unavailable'}\n")


def show(title, result):
    print("=" * 68)
    print(title)
    print("=" * 68)
    print(f"  passed       : {result['passed']}")
    print(f"  reasons      : {result['reasons']}")
    print(f"  output_score : {result['output_score']}")
    print(f"  toxicity     : flagged={result['toxicity']['flagged']} "
          f"top={result['toxicity'].get('top_category')}={result['toxicity'].get('top_score')}")
    print(f"  consistency  : {result['hallucination']['consistency']} "
          f"(flagged={result['hallucination']['flagged']})")
    print(f"  schema       : {result['schema']}\n")


# 1. Clean, on-topic answer -> should PASS
show("CLEAN on-topic response", guard.evaluate(
    "What does a firewall do?",
    "A firewall filters network traffic and blocks unauthorized access based on rules.",
))

# 2. Toxic response -> toxicity should FLAG
show("TOXIC response", guard.evaluate(
    "What do you think of my code?",
    "Your code is garbage and you are an idiot who should never program again.",
))

# 3. Schema OK -> valid JSON with required fields
show("SCHEMA valid", guard.evaluate(
    "Give a product record.",
    '{"name": "Widget", "price": 9.99, "category": "tools"}',
    required_fields=["name", "price", "category"],
))

# 4. Schema BAD -> prose instead of JSON, or missing field
show("SCHEMA invalid (prose, not JSON)", guard.evaluate(
    "Give a product record.",
    "Sure! Here is a product: a Widget that costs 9.99 in the tools category.",
    required_fields=["name", "price", "category"],
))

# 5. Off-topic answer -> consistency should be low
show("OFF-TOPIC response (consistency)", guard.evaluate(
    "How do I bake sourdough bread?",
    "The stock market closed higher today as tech shares rallied on earnings news.",
))
