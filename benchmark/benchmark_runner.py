import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
QUESTIONS_DIR = BASE_DIR / "questions"
PROMPTS_DIR = BASE_DIR / "prompts"
OUTPUTS_DIR = BASE_DIR / "outputs"
REPORTS_DIR = BASE_DIR / "reports"

OLLAMA_URL = "http://127.0.0.1:11434"
RESULTS_LATEST = OUTPUTS_DIR / "benchmark_results_latest.json"
SUMMARY_LATEST = REPORTS_DIR / "benchmark_summary_latest.json"


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=True)


def sanitize_model_name(model_name):
    return model_name.replace(":", "_").replace("/", "_").replace("\\", "_")


def ollama_get(path, timeout=10):
    with urlopen(f"{OLLAMA_URL}{path}", timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ollama_post(path, payload, timeout=600):
    request = Request(
        f"{OLLAMA_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def load_config(question_file_name):
    models_config = load_json(CONFIG_DIR / "models.json")
    settings = load_json(CONFIG_DIR / "benchmark_settings.json")
    questions_path = QUESTIONS_DIR / question_file_name
    questions = load_json(questions_path)
    prompt_template = (PROMPTS_DIR / "benchmark_prompt.txt").read_text(encoding="utf-8")

    models = [model["name"] for model in models_config["models"] if model.get("enabled", True)]
    if not models:
        raise RuntimeError("No enabled models found in benchmark/config/models.json")
    if not questions:
        raise RuntimeError("No questions found in benchmark/questions/benchmark_questions.json")

    return models, settings, questions, prompt_template, questions_path


def start_or_verify_ollama():
    try:
        ollama_get("/api/tags")
        print("Ollama is already running.")
        return
    except Exception:
        print("Ollama is not responding. Trying to start ollama serve...")

    creationflags = 0
    if sys.platform.startswith("win"):
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Ollama CLI was not found. Install Ollama or add it to PATH.") from exc

    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            ollama_get("/api/tags")
            print("Ollama started successfully.")
            return
        except Exception:
            time.sleep(1)

    raise RuntimeError("Ollama did not start within 30 seconds.")


def list_installed_models():
    tags = ollama_get("/api/tags")
    return {model["name"] for model in tags.get("models", [])}


def verify_and_pull_models(models):
    installed = list_installed_models()
    missing = [model for model in models if model not in installed]

    if not missing:
        print("All configured models are already installed.")
        return

    print("Missing models found:")
    for model in missing:
        print(f"  - {model}")

    for model in missing:
        print(f"Pulling {model}...")
        result = subprocess.run(["ollama", "pull", model], text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to pull missing model: {model}")

    installed_after_pull = list_installed_models()
    still_missing = [model for model in models if model not in installed_after_pull]
    if still_missing:
        raise RuntimeError(f"Models still missing after pull: {', '.join(still_missing)}")


def warmup_models(models, settings):
    options = build_options(settings)
    options["num_predict"] = min(int(settings.get("num_predict", 128)), 64)

    for model in models:
        print(f"Warming up {model}...")
        payload = {
            "model": model,
            "prompt": "Warmup only. Reply with: OK",
            "stream": False,
            "options": options,
        }
        try:
            ollama_post("/api/generate", payload, timeout=300)
        except URLError as exc:
            raise RuntimeError(f"Warmup failed for {model}") from exc


def build_options(settings):
    options = {}
    for key in ("temperature", "top_p", "num_predict"):
        if key in settings:
            options[key] = settings[key]
    return options


def print_benchmark_summary(models, questions, settings, questions_path, question_range=None):
    categories = {}
    for question in questions:
        categories[question["category"]] = categories.get(question["category"], 0) + 1

    print("\nBenchmark Summary")
    print("=================")
    print(f"Question file: {questions_path}")
    print(f"Models: {len(models)}")
    for model in models:
        print(f"  - {model}")
    if question_range:
        print(f"Question range: {question_range}")
    print(f"Questions: {len(questions)}")
    for category, count in sorted(categories.items()):
        print(f"  - {category}: {count}")
    print(f"Total generations: {len(models) * len(questions)}")
    print(f"Temperature: {settings.get('temperature')}")
    print(f"Top-p: {settings.get('top_p')}")
    print(f"Max response tokens: {settings.get('num_predict')}")
    print(f"Resume enabled: {settings.get('resume_enabled', True)}")
    print(f"Aggregate output: {RESULTS_LATEST}")
    print(f"Summary output: {SUMMARY_LATEST}\n")


def ask_user_confirmation(auto_yes):
    if auto_yes:
        print("User confirmation skipped because --yes was provided.")
        return

    answer = input("Run benchmark now? This will generate model responses. [y/N]: ").strip().lower()
    if answer not in {"y", "yes"}:
        raise SystemExit("Benchmark cancelled by user.")


def load_existing_results(settings):
    if not settings.get("resume_enabled", True):
        return []
    if not RESULTS_LATEST.exists():
        return []

    data = load_json(RESULTS_LATEST)
    return data.get("responses", [])


def completed_pairs(existing_results):
    return {(item["model"], item["question_id"]) for item in existing_results}


def build_prompt(prompt_template, question):
    return prompt_template.replace("{question}", question["question"])


def generate_response(model, question, prompt, settings):
    options = build_options(settings)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }

    started = time.perf_counter()
    response = ollama_post("/api/generate", payload, timeout=900)
    response_time = time.perf_counter() - started

    prompt_tokens = response.get("prompt_eval_count")
    response_tokens = response.get("eval_count")
    eval_duration_ns = response.get("eval_duration") or 0

    tokens_per_second = None
    if response_tokens and eval_duration_ns:
        tokens_per_second = response_tokens / (eval_duration_ns / 1_000_000_000)
    elif response_tokens and response_time:
        tokens_per_second = response_tokens / response_time

    return {
        "model": model,
        "question_id": question["id"],
        "question_category": question["category"],
        "response": response.get("response", ""),
        "response_time": round(response_time, 3),
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "tokens_per_second": round(tokens_per_second, 3) if tokens_per_second else None,
    }


def save_response_files(result):
    model_dir = OUTPUTS_DIR / sanitize_model_name(result["model"])
    response_path = model_dir / f"{result['question_id']}.json"
    save_json(response_path, result)


def save_aggregate_results(run_data):
    save_json(RESULTS_LATEST, run_data)


def generate_summary(run_data, models, questions, started_at, finished_at):
    responses = run_data["responses"]
    by_model = {}
    by_category = {}

    for response in responses:
        model_stats = by_model.setdefault(
            response["model"],
            {"responses": 0, "avg_response_time": 0.0, "avg_tokens_per_second": 0.0},
        )
        model_stats["responses"] += 1
        model_stats["avg_response_time"] += response.get("response_time") or 0
        model_stats["avg_tokens_per_second"] += response.get("tokens_per_second") or 0

        category_stats = by_category.setdefault(response["question_category"], {"responses": 0})
        category_stats["responses"] += 1

    for stats in by_model.values():
        count = stats["responses"] or 1
        stats["avg_response_time"] = round(stats["avg_response_time"] / count, 3)
        stats["avg_tokens_per_second"] = round(stats["avg_tokens_per_second"] / count, 3)

    summary = {
        "started_at": started_at,
        "finished_at": finished_at,
        "models": models,
        "question_count": len(questions),
        "expected_responses": len(models) * len(questions),
        "stored_responses": len(responses),
        "by_model": by_model,
        "by_category": by_category,
        "results_file": str(RESULTS_LATEST),
    }
    save_json(SUMMARY_LATEST, summary)
    return summary

def apply_question_range(questions, question_range):
    if not question_range:
        return questions

    try:
        start_str, end_str = question_range.split(":", 1)

        start = int(start_str) if start_str else 1
        end = int(end_str) if end_str else len(questions)

    except ValueError:
        raise ValueError(
            "--question-range must be in format start:end "
            "(examples: 1:10, 11:20, 50:)"
        )

    if start < 1:
        raise ValueError("Question range start must be >= 1")

    if end < start:
        raise ValueError("Question range end must be >= start")

    selected = questions[start - 1 : end]

    if not selected:
        raise ValueError(
            f"No questions selected from range {question_range}"
        )

    return selected

def run_benchmark(models, settings, questions, prompt_template):
    started_at = datetime.now().isoformat(timespec="seconds")
    existing_results = load_existing_results(settings)
    done = completed_pairs(existing_results)

    run_data = {
        "started_at": started_at,
        "settings": settings,
        "models": models,
        "question_count": len(questions),
        "question_range": settings.get("question_range"),
        "responses": existing_results,
    }

    total = len(models) * len(questions)
    completed = len(done)

    for model in models:
        print(f"\nRunning all questions for model: {model}")
        for question in questions:
            prompt = build_prompt(prompt_template, question)
            key = (model, question["id"])
            if key in done:
                print(f"[{completed}/{total}] Skipping existing {model} {question['id']}")
                continue

            print(f"[{completed + 1}/{total}] Running {model} on {question['id']} ({question['category']})")
            result = generate_response(model, question, prompt, settings)
            run_data["responses"].append(result)
            done.add(key)
            completed += 1

            save_response_files(result)
            save_aggregate_results(run_data)

    finished_at = datetime.now().isoformat(timespec="seconds")
    run_data["finished_at"] = finished_at
    save_aggregate_results(run_data)
    return generate_summary(run_data, models, questions, started_at, finished_at)


def main():
    parser = argparse.ArgumentParser(description="Run the local Ollama LLM council role benchmark.")
    parser.add_argument("--yes", action="store_true", help="Skip interactive confirmation.")
    parser.add_argument(
    "--questions",
    default="sample_questions.json",
    help="Question file from benchmark/questions. Default: sample_questions.json",
    )
    parser.add_argument(
        "--question-range",
        type=str,
        default=None,
        help="Range of questions to run. Examples: 1:10, 11:20, 50:",
    )
    args = parser.parse_args()
    

    print("1. Load Config")
    models, settings, questions, prompt_template, questions_path = load_config(args.questions)
    questions = apply_question_range(
        questions,
        args.question_range,
    )

    print("2. Start/Verify Ollama")
    start_or_verify_ollama()

    print("3. Verify Models")
    print("4. Auto Pull Missing Models")
    verify_and_pull_models(models)

    print("5. Warmup Models")
    warmup_models(models, settings)

    print("6. Benchmark Summary")
    print_benchmark_summary(models, questions, settings, questions_path, args.question_range)

    print("7. User Confirmation")
    ask_user_confirmation(args.yes)

    print("8. Run Benchmark")
    summary = run_benchmark(models, settings, questions, prompt_template)

    print("9. Save Results")
    print(f"Results saved to {RESULTS_LATEST}")

    print("10. Generate Summary")
    print(f"Summary saved to {SUMMARY_LATEST}")
    print(f"Stored responses: {summary['stored_responses']} / {summary['expected_responses']}")


if __name__ == "__main__":
    main()
