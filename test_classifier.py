from answer import classify_query

tests = [
    # Greetings - should all return "greeting"
    ("hello", "greeting"),
    ("Hello!", "greeting"),
    ("hey there", "greeting"),
    ("hey", "greeting"),
    ("good morning", "greeting"),
    ("good afternoon", "greeting"),
    ("good evening", "greeting"),
    ("hi there", "greeting"),
    ("hi!!!", "greeting"),
    # Small talk - should all return "small_talk"
    ("how are you?", "small_talk"),
    ("thanks", "small_talk"),
    ("thank you", "small_talk"),
    ("bye", "small_talk"),
    ("ok", "small_talk"),
    # Knowledge queries - should all return "knowledge_query"
    ("what is the transformer?", "knowledge_query"),
    ("what is the key result of this research?", "knowledge_query"),
    ("explain self-attention", "knowledge_query"),
    ("how does multi-head attention work?", "knowledge_query"),
]

passed = 0
failed = 0
for query, expected in tests:
    result = classify_query(query)
    status = "✅" if result == expected else "❌"
    if result != expected:
        failed += 1
    else:
        passed += 1
    print(f"{status} {query!r:45} -> {result} (expected: {expected})")

print(f"\n{passed}/{len(tests)} passed")
