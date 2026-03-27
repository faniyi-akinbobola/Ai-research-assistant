import json
import os
from datetime import datetime

METRICS_FILE = "./data/metrics.json"

def load_metrics() -> dict:
    """Load existing metrics from file."""
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            return json.load(f)
    return {
        "total_queries": 0,
        "query_types": {"greeting": 0, "small_talk": 0, "knowledge_query": 0},
        "errors": 0,
        "sources_shown": 0,
        "sources_hidden": 0,
        "queries": []
    }

def log_query(
    question: str,
    query_type: str,
    response_time_ms: float,
    sources_count: int,
    sources_shown: bool,
    error: bool = False
):
    """Log a single query's metrics."""
    metrics = load_metrics()

    metrics["total_queries"] += 1
    metrics["query_types"][query_type] = metrics["query_types"].get(query_type, 0) + 1

    if error:
        metrics["errors"] += 1
    if sources_shown:
        metrics["sources_shown"] += 1
    else:
        metrics["sources_hidden"] += 1

    metrics["queries"].append({
        "timestamp": datetime.now().isoformat(),
        "question": question[:100],
        "query_type": query_type,
        "response_time_ms": round(response_time_ms, 2),
        "sources_count": sources_count,
        "sources_shown": sources_shown,
        "error": error
    })

    os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

def print_summary():
    """Print a summary of all metrics."""
    metrics = load_metrics()

    print("=" * 50)
    print("METRICS SUMMARY")
    print("=" * 50)
    print(f"Total queries     : {metrics['total_queries']}")
    print(f"Errors            : {metrics['errors']}")
    print(f"Sources shown     : {metrics['sources_shown']}")
    print(f"Sources hidden    : {metrics['sources_hidden']}")
    print("\nQuery type breakdown:")
    for query_type, count in metrics["query_types"].items():
        print(f"  {query_type:<20}: {count}")

    if metrics["queries"]:
        response_times = [q["response_time_ms"] for q in metrics["queries"] if not q["error"]]
        if response_times:
            print(f"\nAvg response time : {sum(response_times) / len(response_times):.0f}ms")
            print(f"Fastest           : {min(response_times):.0f}ms")
            print(f"Slowest           : {max(response_times):.0f}ms")
    print("=" * 50)

if __name__ == "__main__":
    print_summary()
