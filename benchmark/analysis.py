import json
import csv
import statistics
from pathlib import Path
from collections import defaultdict

# ==================================================
# PATHS
# ==================================================

BASE_DIR = Path(__file__).parent

INPUT_FILE = (
    BASE_DIR /
    "evaluations/llama evaluator" /
    "evaluation_results_latest.json"
)

ANALYSIS_DIR = BASE_DIR / "analysis"
ANALYSIS_DIR.mkdir(exist_ok=True)

# ==================================================
# LOAD DATA
# ==================================================

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

evaluations = data["evaluations"]

# ==================================================
# STORAGE
# ==================================================

model_stats = defaultdict(list)
category_stats = defaultdict(lambda: defaultdict(list))

metric_names = [
    "accuracy",
    "completeness",
    "structure",
    "reasoning",
    "critical_thinking",
    "practicality"
]

metric_scores = defaultdict(
    lambda: defaultdict(list)
)

wins = defaultdict(int)

podiums = defaultdict(
    lambda: {
        "first": 0,
        "second": 0,
        "third": 0,
        "fourth": 0
    }
)

response_times = defaultdict(list)
tps_values = defaultdict(list)

head_to_head = defaultdict(int)

# ==================================================
# PARSE EVALUATIONS
# ==================================================

for evaluation in evaluations:

    category = evaluation["question_category"]

    scores = evaluation["scores"]

    ranking = evaluation["ranking"]

    answer_key = evaluation["answer_key"]

    # --------------------------
    # Rankings
    # --------------------------

    if ranking:
        winner_model = (
            answer_key[ranking[0]]["model"]
        )
        wins[winner_model] += 1

    rank_names = [
        "first",
        "second",
        "third",
        "fourth"
    ]

    for idx, label in enumerate(ranking):

        model = answer_key[label]["model"]

        podiums[model][
            rank_names[idx]
        ] += 1

    # --------------------------
    # Head to Head
    # --------------------------

    ranked_models = [
        answer_key[x]["model"]
        for x in ranking
    ]

    for i in range(len(ranked_models)):
        for j in range(i + 1,
                       len(ranked_models)):

            winner = ranked_models[i]
            loser = ranked_models[j]

            head_to_head[
                (winner, loser)
            ] += 1

    # --------------------------
    # Score Collection
    # --------------------------

    for label, score_obj in scores.items():

        model = answer_key[label]["model"]

        total_score = score_obj["total"]

        model_stats[model].append(
            total_score
        )

        category_stats[
            category
        ][model].append(
            total_score
        )

        for metric in metric_names:

            metric_scores[
                model
            ][metric].append(
                score_obj[metric]
            )

        rt = answer_key[label].get(
            "response_time"
        )

        if rt is not None:
            response_times[
                model
            ].append(rt)

        tps = answer_key[label].get(
            "tokens_per_second"
        )

        if tps is not None:
            tps_values[
                model
            ].append(tps)

# ==================================================
# OVERALL SCORES
# ==================================================

overall_rows = []

for model, values in model_stats.items():

    avg_score = (
        sum(values) / len(values)
    )

    overall_rows.append({
        "model": model,
        "avg_total_score":
            round(avg_score, 3),
        "wins":
            wins[model],
        "questions":
            len(values)
    })

overall_rows.sort(
    key=lambda x:
    x["avg_total_score"],
    reverse=True
)

with open(
    ANALYSIS_DIR /
    "overall_scores.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model",
            "avg_total_score",
            "wins",
            "questions"
        ]
    )

    writer.writeheader()
    writer.writerows(
        overall_rows
    )

# ==================================================
# CATEGORY SCORES
# ==================================================

with open(
    ANALYSIS_DIR /
    "category_scores.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "category",
        "model",
        "avg_score"
    ])

    for category,models in category_stats.items():

        for model,scores_list in models.items():

            writer.writerow([
                category,
                model,
                round(
                    sum(scores_list)
                    /
                    len(scores_list),
                    3
                )
            ])

# ==================================================
# METRIC SCORES
# ==================================================

with open(
    ANALYSIS_DIR /
    "metric_scores.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:

    header = (
        ["model"] +
        metric_names
    )

    writer = csv.writer(f)

    writer.writerow(header)

    for model,metrics in metric_scores.items():

        row = [model]

        for metric in metric_names:

            row.append(
                round(
                    sum(
                        metrics[
                            metric
                        ]
                    )
                    /
                    len(
                        metrics[
                            metric
                        ]
                    ),
                    3
                )
            )

        writer.writerow(row)

# ==================================================
# FIRST PLACE COUNTS
# ==================================================

with open(
    ANALYSIS_DIR /
    "first_place_counts.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "model",
        "wins"
    ])

    for model,count in sorted(
            wins.items(),
            key=lambda x: x[1],
            reverse=True
        ):

        writer.writerow([
            model,
            count
        ])

# ==================================================
# PODIUM COUNTS
# ==================================================

with open(
    ANALYSIS_DIR /
    "podium_counts.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "model",
        "first",
        "second",
        "third",
        "fourth"
    ])

    for model,values in podiums.items():

        writer.writerow([
            model,
            values["first"],
            values["second"],
            values["third"],
            values["fourth"]
        ])

# ==================================================
# EFFICIENCY
# ==================================================

with open(
    ANALYSIS_DIR /
    "efficiency_scores.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "model",
        "avg_score",
        "avg_response_time",
        "efficiency"
    ])

    for model in model_stats:

        avg_score = (
            sum(model_stats[model])
            /
            len(model_stats[model])
        )

        avg_rt = (
            sum(
                response_times[
                    model
                ]
            )
            /
            len(
                response_times[
                    model
                ]
            )
        )

        efficiency = (
            avg_score / avg_rt
        )

        writer.writerow([
            model,
            round(avg_score, 3),
            round(avg_rt, 3),
            round(
                efficiency,
                5
            )
        ])

# ==================================================
# HEAD TO HEAD
# ==================================================

with open(
    ANALYSIS_DIR /
    "head_to_head.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "winner",
        "loser",
        "wins"
    ])

    for (
        winner,
        loser
    ), count in sorted(
        head_to_head.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        writer.writerow([
            winner,
            loser,
            count
        ])

# ==================================================
# SCORE VARIANCE
# ==================================================

with open(
    ANALYSIS_DIR /
    "score_variance.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "model",
        "metric_stddev"
    ])

    for model,metrics in metric_scores.items():

        all_std = []

        for metric in metric_names:

            values = metrics[
                metric
            ]

            if len(values) > 1:

                all_std.append(
                    statistics.stdev(
                        values
                    )
                )

        writer.writerow([
            model,
            round(
                sum(all_std)
                /
                len(all_std),
                5
            )
        ])

# ==================================================
# MODEL SUMMARY JSON
# ==================================================

summary = {}

for model in model_stats:

    summary[model] = {

        "avg_score":
            round(
                sum(
                    model_stats[
                        model
                    ]
                )
                /
                len(
                    model_stats[
                        model
                    ]
                ),
                3
            ),

        "wins":
            wins[model],

        "avg_response_time":
            round(
                sum(
                    response_times[
                        model
                    ]
                )
                /
                len(
                    response_times[
                        model
                    ]
                ),
                3
            ),

        "avg_tps":
            round(
                sum(
                    tps_values[
                        model
                    ]
                )
                /
                len(
                    tps_values[
                        model
                    ]
                ),
                3
            )
    }

with open(
    ANALYSIS_DIR /
    "model_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )

# ==================================================
# FINAL REPORT
# ==================================================

report = []

report.append(
    "=" * 60
)

report.append(
    "LLM COUNCIL BENCHMARK REPORT"
)

report.append(
    "=" * 60
)

report.append("")

report.append(
    "OVERALL RANKING"
)

for idx,row in enumerate(
        overall_rows,
        start=1
    ):

    report.append(
        f"{idx}. "
        f"{row['model']} "
        f"(Score="
        f"{row['avg_total_score']})"
    )

report.append("")

report.append(
    "CATEGORY WINNERS"
)

for category,models in category_stats.items():

    best_model = max(
        models.items(),
        key=lambda x:
        sum(x[1])
        /
        len(x[1])
    )[0]

    report.append(
        f"{category}: "
        f"{best_model}"
    )

report.append("")

report.append(
    "FIRST PLACE COUNTS"
)

for model,count in sorted(
        wins.items(),
        key=lambda x: x[1],
        reverse=True
    ):

    report.append(
        f"{model}: {count}"
    )

with open(
    ANALYSIS_DIR /
    "final_report.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(report)
    )

print(
    "\nAnalysis complete."
)

print(
    f"Output folder: "
    f"{ANALYSIS_DIR}"
)