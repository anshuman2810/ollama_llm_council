import json
import ollama

from pathlib import Path
from datetime import datetime
import subprocess
import requests
import time
import re

# ==================================================
# PATHS
# ==================================================

BASE_DIR = Path(__file__).parent

CONFIG_DIR = BASE_DIR / "config"
PROMPTS_DIR = BASE_DIR / "prompts"

OUTPUT_DIR = (
    BASE_DIR /
    "outputs" /
    "council_runs"
)

LOG_DIR = BASE_DIR / "logs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LOG_DIR.mkdir(
    exist_ok=True
)

# ==================================================
# HELPERS
# ==================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def load_prompt(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return f.read()


def log(message):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    line = f"[{timestamp}] {message}"

    print(line)

    with open(
        LOG_DIR / "council.log",
        "a",
        encoding="utf-8"
    ) as f:

        f.write(line + "\n")


# ==================================================
# OLLAMA CALL
# ==================================================

def is_ollama_running():

    try:

        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=2
        )

        return response.status_code == 200

    except Exception:

        return False
    
def start_ollama():

    print("\nChecking Ollama...")

    if is_ollama_running():

        print("Ollama already running.")
        return

    print(
        "Ollama not running."
    )

    print(
        "Starting Ollama..."
    )

    try:

        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    except Exception as e:

        raise RuntimeError(
            f"Failed to start Ollama: {e}"
        )

    for _ in range(15):

        if is_ollama_running():

            print(
                "Ollama started successfully."
            )

            return

        time.sleep(1)

    raise RuntimeError(
        "Ollama failed to start."
    )

def get_installed_models():

    response = requests.get(
        "http://localhost:11434/api/tags"
    )

    data = response.json()

    return {
        item["name"]
        for item in data["models"]
    }

def verify_model(model_name):

    installed = get_installed_models()

    if model_name in installed:

        print(
            f"Model installed: {model_name}"
        )

        return

    print(
        f"\nModel missing: "
        f"{model_name}"
    )

    answer = input(
        f"Pull {model_name}? "
        f"(y/n): "
    )

    if answer.lower() != "y":

        raise RuntimeError(
            f"Required model missing: "
            f"{model_name}"
        )

    subprocess.run(
        ["ollama", "pull", model_name],
        check=True
    )




def ask_model(
    model_name,
    prompt,
    settings
):

    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature":
                settings["temperature"],

            "top_p":
                settings["top_p"],

            "num_predict":
                settings["num_predict"]
        }
    )

    return {
        "text":
            response["message"]["content"],

        "prompt_tokens":
            response.get(
                "prompt_eval_count",
                0
            ),

        "response_tokens":
            response.get(
                "eval_count",
                0
            )
    }


# ==================================================
# LOAD CONFIG
# ==================================================

roles = load_json(
    CONFIG_DIR / "roles.json"
)

settings = load_json(
    CONFIG_DIR / "settings.json"
)

analyst_prompt_template = load_prompt(
    PROMPTS_DIR / "analyst.txt"
)

critic_prompt_template = load_prompt(
    PROMPTS_DIR / "critic.txt"
)

judge_prompt_template = load_prompt(
    PROMPTS_DIR / "judge.txt"
)

# ==================================================
# QUESTION
# ==================================================

print(
    "\n1. Verify Ollama"
)

start_ollama()

print(
    "\n2. Verify Models"
)

verify_model(
    roles["analyst"]
)

verify_model(
    roles["critic"]
)

verify_model(
    roles["judge"]
)

SESSION_ID = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

SESSION_DIR = (
    OUTPUT_DIR /
    SESSION_ID
)

SESSION_DIR.mkdir(
    parents=True,
    exist_ok=True
)

question_counter = 1

while True:

    question = input(
        "\nEnter Question (type 'exit' to quit):\n\n> "
    ).strip()

    if not question:
        continue

    if question.lower() in [
        "exit",
        "quit",
        "q"
    ]:

        print(
            "\nExiting council..."
        )

        break

    log("Question received")

    # ==================================================
    # ANALYST
    # ==================================================

    analyst_prompt = (
        analyst_prompt_template
        .replace(
            "{question}",
            question
        )
    )

    log(
        f"Running Analyst "
        f"({roles['analyst']})"
    )

    analyst = ask_model(
        roles["analyst"],
        analyst_prompt,
        settings
    )

    analyst_response = analyst["text"]

    print(
        "\n" +
        "=" * 80
    )
    print("ANALYST")
    print("=" * 80)
    print(analyst_response)

    # ==================================================
    # CRITIC
    # ==================================================

    critic_prompt = (
        critic_prompt_template
        .replace(
            "{question}",
            question
        )
        .replace(
            "{analyst_response}",
            analyst_response
        )
    )

    log(
        f"Running Critic "
        f"({roles['critic']})"
    )

    critic = ask_model(
        roles["critic"],
        critic_prompt,
        settings
    )

    critic_response = critic["text"]
    critic_response = re.sub(
        r"<think>.*?</think>",
        "",
        critic_response,
        flags=re.DOTALL
    ).strip()

    print(
        "\n" +
        "=" * 80
    )
    print("CRITIC")
    print("=" * 80)
    print(critic_response)

    # ==================================================
    # JUDGE
    # ==================================================

    judge_prompt = (
        judge_prompt_template
        .replace(
            "{question}",
            question
        )
        .replace(
            "{analyst_response}",
            analyst_response
        )
        .replace(
            "{critic_response}",
            critic_response
        )
    )

    log(
        f"Running Judge "
        f"({roles['judge']})"
    )

    judge = ask_model(
        roles["judge"],
        judge_prompt,
        settings
    )

    final_response = judge["text"]

    print(
        "\n" +
        "=" * 80
    )
    print("FINAL COUNCIL ANSWER")
    print("=" * 80)
    print(final_response)

    # ==================================================
    # SAVE RUN
    # ==================================================

    run_data = {

        "created_at":
            datetime.now()
            .isoformat(),

        "question":
            question,

        "analyst_model":
            roles["analyst"],

        "critic_model":
            roles["critic"],

        "judge_model":
            roles["judge"],

        "analyst_response":
            analyst_response,

        "critic_response":
            critic_response,

        "final_response":
            final_response,

        "token_usage":
        {
            "analyst":
            {
                "prompt_tokens":
                    analyst[
                        "prompt_tokens"
                    ],

                "response_tokens":
                    analyst[
                        "response_tokens"
                    ]
            },

            "critic":
            {
                "prompt_tokens":
                    critic[
                        "prompt_tokens"
                    ],

                "response_tokens":
                    critic[
                        "response_tokens"
                    ]
            },

            "judge":
            {
                "prompt_tokens":
                    judge[
                        "prompt_tokens"
                    ],

                "response_tokens":
                    judge[
                        "response_tokens"
                    ]
            }
        }
    }

    output_file = (
    SESSION_DIR /
    f"q{question_counter}.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            run_data,
            f,
            indent=4,
            ensure_ascii=False
        )

    log(
        f"Council run saved: "
        f"{output_file.name}"
    )

    print(
        f"\nSaved:\n{output_file}"
    )
    question_counter += 1