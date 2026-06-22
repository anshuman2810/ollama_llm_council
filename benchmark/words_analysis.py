import json

with open(
    "outputs/benchmark_results_latest.json",
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)

responses = data["responses"]

by_model = {}

for r in responses:

    model = r["model"]

    by_model.setdefault(
        model,
        {
            "total": 0,
            "truncated": 0
        }
    )

    by_model[model]["total"] += 1

    if r.get("response_tokens") == 1024:
        by_model[model]["truncated"] += 1

for model, stats in by_model.items():

    pct = (
        stats["truncated"]
        / stats["total"]
        * 100
    )

    print(
        f"{model}: "
        f"{stats['truncated']}/"
        f"{stats['total']} "
        f"({pct:.1f}%)"
    )