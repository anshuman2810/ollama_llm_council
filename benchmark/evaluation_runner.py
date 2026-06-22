import argparse
import json
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen


BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
QUESTIONS_DIR = BASE_DIR / "questions"
PROMPTS_DIR = BASE_DIR / "prompts"
OUTPUTS_DIR = BASE_DIR / "outputs"
EVALUATIONS_DIR = BASE_DIR / "evaluations"

OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_EVALUATOR_MODEL = "llama3.2:3b"
EVALUATION_LATEST = EVALUATIONS_DIR / "evaluation_results_latest.json"

MODEL_FOLDERS = {
    "qwen3:4b": "qwen3_4b",
    "gemma3:4b": "gemma3_4b",
    "deepseek-r1:7b": "deepseek-r1_7b",
    "llama3.2:3b": "llama3.2_3b",
}

ANSWER_LABELS = ["A", "B", "C", "D"]
METRICS = ["accuracy", "completeness", "structure", "reasoning", "critical_thinking", "practicality"]

# Maximum words per answer sent to the evaluator.
# mistral:7b has a 32K context window so 800 words per answer is comfortable:
# Template ~700 tokens + 4 answers * 800 words (~1,067 tokens each) = ~5,000 tokens in.
# Leaves ~3,000 tokens for the JSON response output.
MAX_ANSWER_WORDS = 350


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=True)


def ollama_get(path, timeout=10):
    with urlopen(f"{OLLAMA_URL}{path}", timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ollama_post(path, payload, timeout=900):
    request = Request(
        f"{OLLAMA_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


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


def ensure_evaluator_model(model_name):
    installed = list_installed_models()
    if model_name in installed:
        print(f"Evaluator model is installed: {model_name}")
        return

    print(f"Evaluator model is missing. Pulling {model_name}...")
    result = subprocess.run(["ollama", "pull", model_name], text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to pull evaluator model: {model_name}")


def load_models():
    models_config = load_json(CONFIG_DIR / "models.json")
    return [model["name"] for model in models_config["models"] if model.get("enabled", True)]


def load_questions(question_file_name):
    return load_json(QUESTIONS_DIR / question_file_name)


def load_prompt_template():
    return (PROMPTS_DIR / "evaluator_prompt.txt").read_text(encoding="utf-8")


def load_response(model, question_id):
    folder = MODEL_FOLDERS.get(model, model.replace(":", "_").replace("/", "_").replace("\\", "_"))
    path = OUTPUTS_DIR / folder / f"{question_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing response file: {path}")
    return load_json(path)


def clean_response(text):
    """Strip chain-of-thought blocks and truncate to a word budget.

    Models like qwen3 and deepseek-r1 emit <think>...</think> blocks
    containing internal reasoning that should not be evaluated.
    """
    # Remove <think>...</think> blocks (handles multiline, non-greedy)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()

    # Hard word cap so all four answers fit inside the evaluator's context window
    words = text.split()
    if len(words) > MAX_ANSWER_WORDS:
        text = " ".join(words[:MAX_ANSWER_WORDS]) + "\n[truncated]"

    return text


def build_blind_answers(models, question_id, seed):
    responses = []
    for model in models:
        response_data = load_response(model, question_id)
        responses.append(
            {
                "model": model,
                "response": clean_response(response_data.get("response", "")),
                "response_time": response_data.get("response_time"),
                "prompt_tokens": response_data.get("prompt_tokens"),
                "response_tokens": response_data.get("response_tokens"),
                "tokens_per_second": response_data.get("tokens_per_second"),
            }
        )

    shuffled = responses[:]
    random.Random(f"{seed}:{question_id}").shuffle(shuffled)

    blind_answers = {}
    answer_key = {}
    for label, item in zip(ANSWER_LABELS, shuffled):
        blind_answers[label] = item["response"]
        answer_key[label] = {
            "model": item["model"],
            "response_time": item["response_time"],
            "prompt_tokens": item["prompt_tokens"],
            "response_tokens": item["response_tokens"],
            "tokens_per_second": item["tokens_per_second"],
        }

    return blind_answers, answer_key


def build_prompt(template, question, blind_answers):
    values = {
        "question_id": question["id"],
        "question_category": question["category"],
        "question": question["question"],
        "answer_a": blind_answers["A"],
        "answer_b": blind_answers["B"],
        "answer_c": blind_answers["C"],
        "answer_d": blind_answers["D"],
    }
    prompt = template
    for key, value in values.items():
        prompt = prompt.replace("{" + key + "}", str(value))
    return prompt


def evaluate_question(evaluator_model, prompt):
    payload = {
        "model": evaluator_model,
        "prompt": prompt,
        "system": (
            "You are a JSON compiler. "
            "Output exactly one valid JSON object. "
            "No markdown. "
            "No explanations. "
            "No prose. "
            "No reasoning. "
            "No <think> tags. "
            "Do not invent fields. "
            "The first character must be '{' "
            "and the last character must be '}'."
        ),
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "top_p": 0.8,
            "num_predict": 512,
        },
    }
    started = time.perf_counter()
    response = ollama_post("/api/generate", payload, timeout=1200)
    elapsed = time.perf_counter() - started
    raw = response.get("response", "")
    # qwen3 may still emit <think>...</think> before the JSON
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return raw, round(elapsed, 3)


def parse_json_response(raw_text):
    """Parse the evaluator's JSON response with progressive fallback strategies."""
    text = raw_text.strip()

    # 1. Clean parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Extract outermost {...} block (handles leading/trailing non-JSON text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in evaluator response. First 300 chars: {text[:300]}")

    extracted = text[start:end + 1]

    try:
        return json.loads(extracted)
    except json.JSONDecodeError:
        pass

    # 3. Truncated JSON recovery: parse as much as raw_decode can handle.
    # This recovers partial objects where the last field was cut mid-token,
    # which validate_evaluation() will then fill with safe defaults.
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(extracted)
        return obj
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Could not parse evaluator response as JSON.\n"
            f"Response length: {len(text)} chars\n"
            f"First 300 chars: {text[:300]}\n"
            f"Last 200 chars: {text[-200:]}"
        ) from exc


def validate_evaluation(evaluation, question_id):
    if evaluation.get("question_id") != question_id:
        evaluation["question_id"] = question_id

    scores = evaluation.setdefault("scores", {})
    for label in ANSWER_LABELS:
        item = scores.setdefault(label, {})
        total = 0
        for metric in METRICS:
            value = item.get(metric)
            if not isinstance(value, int):
                value = int(value) if str(value).isdigit() else 1
            value = max(1, min(10, value))
            item[metric] = value
            total += value
        item["total"] = total

    # ranking = evaluation.get("ranking")
    ranking = sorted(
    ANSWER_LABELS,
    key=lambda label:
        (
            scores[label]["total"],
            scores[label]["reasoning"],
            scores[label]["critical_thinking"],
            scores[label]["practicality"]
        ),
    reverse=True
    )

    evaluation["ranking"] = ranking
    # if not isinstance(ranking, list) or sorted(ranking) != ANSWER_LABELS:
    #     ranking = sorted(ANSWER_LABELS, key=lambda label: scores[label]["total"], reverse=True)
    #     evaluation["ranking"] = ranking

    evaluation["winner"] = ranking[0]
    return evaluation


def load_existing_evaluations():
    if not EVALUATION_LATEST.exists():
        return []
    return load_json(EVALUATION_LATEST).get("evaluations", [])


def save_question_evaluation(result):
    save_json(EVALUATIONS_DIR / f"{result['question_id']}.json", result)


def save_aggregate(run_data):
    save_json(EVALUATION_LATEST, run_data)


def print_summary(evaluator_model, models, questions):
    print("\nEvaluation Summary")
    print("==================")
    print(f"Evaluator model: {evaluator_model}")
    print(f"Questions: {len(questions)}")
    print(f"Answers per question: {len(models)}")
    print(f"Total answer evaluations: {len(questions) * len(models)}")
    print(f"Max words per answer (sent to evaluator): {MAX_ANSWER_WORDS}")
    print(f"Output folder: {EVALUATIONS_DIR}")
    print(f"Aggregate file: {EVALUATION_LATEST}\n")


def ask_user_confirmation(auto_yes):
    if auto_yes:
        print("User confirmation skipped because --yes was provided.")
        return

    answer = input("Run blind evaluation now? [y/N]: ").strip().lower()
    if answer not in {"y", "yes"}:
        raise SystemExit("Evaluation cancelled by user.")


def run_evaluation(evaluator_model, models, questions, prompt_template, seed):
    existing = load_existing_evaluations()
    completed = {item["question_id"] for item in existing}

    run_data = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "evaluator_model": evaluator_model,
        "blind_evaluation": True,
        "metrics": METRICS,
        "models_evaluated": models,
        "question_count": len(questions),
        "evaluations": existing,
    }

    total = len(questions)
    for index, question in enumerate(questions, start=1):
        question_id = question["id"]
        if question_id in completed:
            print(f"[{index}/{total}] Skipping existing evaluation for {question_id}")
            continue

        print(f"[{index}/{total}] Evaluating {question_id} blindly")
        blind_answers, answer_key = build_blind_answers(models, question_id, seed)
        prompt = build_prompt(prompt_template, question, blind_answers)
        evaluation = None
        raw_evaluator_response = ""

        for attempt in range(3):

            try:

                raw_evaluator_response, evaluation_time = (
                    evaluate_question(
                        evaluator_model,
                        prompt
                    )
                )

                parsed = parse_json_response(
                    raw_evaluator_response
                )

                evaluation = validate_evaluation(
                    parsed,
                    question_id
                )

                break

            except Exception as exc:

                print(
                    f"  Retry {attempt+1}/3 failed: {exc}"
                )

                if attempt == 2:

                    save_json(
                        EVALUATIONS_DIR /
                        f"{question_id}_failed.json",
                        {
                            "question_id": question_id,
                            "error": str(exc),
                            "raw_response": raw_evaluator_response
                        }
                    )

                    print(
                        f"  FAILED: {question_id}"
                    )

                    evaluation = None
                if evaluation is None:
                    continue

        result = {
            "question_id": question_id,
            "question_category": question["category"],
            "question": question["question"],
            "evaluator_model": evaluator_model,
            "evaluation_time": evaluation_time,
            "scores": evaluation["scores"],
            "ranking": evaluation["ranking"],
            "answer_key": answer_key,
        }

        run_data["evaluations"].append(result)
        completed.add(question_id)
        save_question_evaluation(result)
        save_aggregate(run_data)

    run_data["finished_at"] = datetime.now().isoformat(timespec="seconds")
    save_aggregate(run_data)
    return run_data


def main():
    parser = argparse.ArgumentParser(description="Run blind model-response evaluation with Ollama.")
    parser.add_argument("--yes", action="store_true", help="Skip interactive confirmation.")
    parser.add_argument("--evaluator-model", default=DEFAULT_EVALUATOR_MODEL)
    parser.add_argument(
        "--questions",
        default="sample_questions.json",
        help="Question file from benchmark/questions. Default: sample_questions.json",
    )
    parser.add_argument("--seed", default="llm-council-eval-v1", help="Stable answer shuffle seed.")
    parser.add_argument("--dry-run", action="store_true", help="Print the first blind evaluation prompt and exit.")
    args = parser.parse_args()

    print("1. Load Evaluation Config")
    models = load_models()
    questions = load_questions(args.questions)
    prompt_template = load_prompt_template()

    print("2. Start/Verify Ollama")
    start_or_verify_ollama()

    print("3. Verify Evaluator Model")
    ensure_evaluator_model(args.evaluator_model)

    print("4. Evaluation Summary")
    print_summary(args.evaluator_model, models, questions)

    if args.dry_run:
        question = questions[0]
        blind_answers, answer_key = build_blind_answers(models, question["id"], args.seed)
        prompt = build_prompt(prompt_template, question, blind_answers)
        print("Dry run prompt preview:")
        print(prompt)
        print("\nHidden answer key, not sent to evaluator:")
        print(json.dumps(answer_key, indent=2))
        return

    print("5. User Confirmation")
    ask_user_confirmation(args.yes)

    print("6. Run Blind Evaluation")
    results = run_evaluation(args.evaluator_model, models, questions, prompt_template, args.seed)

    print("7. Save Results")
    print(f"Saved {len(results['evaluations'])} question evaluations to {EVALUATION_LATEST}")


if __name__ == "__main__":
    main()