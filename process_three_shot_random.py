import os
import json
import random
from tqdm import tqdm

# ------------------------
# CONFIG
# ------------------------
SEED = 42
BASE_DIR = "./data/wmt14"

random.seed(SEED)

# ------------------------
# IO UTILS
# ------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ------------------------
# PROMPT CONSTRUCTION (three-SHOT)
# ------------------------
def build_prompt_with_demos(demo_exs, instruction, src):
    prompt = "You are a translation assistant.\n\n"

    for i, demo in enumerate(demo_exs, start=1):
        prompt += (
            f"Example {i}:\n"
            f"Instruction: {demo['instruction']}\n"
            f"Input: {demo['input']}\n"
            f"Output: {demo['output']}\n\n"
        )

    prompt += (
        "Now translate the following sentence.\n"
        f"Instruction: {instruction}\n"
    )

    return prompt

# ------------------------
# CREATE TEST SET WITH RANDOM three-SHOT DEMOS
# ------------------------
def create_test_with_random_demos(
    test_data,
    valid_data,
    base_instruction,
    num_shots=3
):
    processed = []

    for ex in tqdm(test_data):
        src = ex["input"].strip()
        tgt = ex["output"].strip()

        if not src or not tgt:
            continue
        if len(src.split()) < 5 or len(tgt.split()) < 5:
            continue

        # Random three distinct validation examples per test instance
        demo_exs = random.sample(valid_data, num_shots)

        prompt = build_prompt_with_demos(
            demo_exs=demo_exs,
            instruction=base_instruction,
            src=src
        )

        processed.append({
            "instruction": prompt,
            "input": src,
            "output": tgt
        })

    return processed

# ------------------------
# LOAD DATA (JSON ONLY)
# ------------------------
print("📥 Loading existing JSON data...")

en_de_train = load_json(f"{BASE_DIR}/en-de/train.json")
en_de_valid = load_json(f"{BASE_DIR}/en-de/validation.json")
en_de_test  = load_json(f"{BASE_DIR}/en-de/test.json")

de_en_train = load_json(f"{BASE_DIR}/de-en/train.json")
de_en_valid = load_json(f"{BASE_DIR}/de-en/validation.json")
de_en_test  = load_json(f"{BASE_DIR}/de-en/test.json")

# ------------------------
# BASE INSTRUCTIONS
# ------------------------
INSTR_EN_DE = "Translate the following sentence from English to German."
INSTR_DE_EN = "Translate the following sentence from German to English."

# ------------------------
# BUILD TEST SETS WITH three-SHOT DEMOS
# ------------------------
print("🧪 Creating test sets with random three-shot in-context examples...")

test_en_de_with_demo = create_test_with_random_demos(
    test_data=en_de_test,
    valid_data=en_de_valid,
    base_instruction=INSTR_EN_DE,
    num_shots=3
)

test_de_en_with_demo = create_test_with_random_demos(
    test_data=de_en_test,
    valid_data=de_en_valid,
    base_instruction=INSTR_DE_EN,
    num_shots=3
)

# ------------------------
# SAVE
# ------------------------
save_json(
    test_en_de_with_demo,
    f"{BASE_DIR}/three-shot/random/en-de/test.json"
)

save_json(
    test_de_en_with_demo,
    f"{BASE_DIR}/three-shot/random/de-en/test.json"
)

print("✅ Done.")
print(f"📁 Files written to: {BASE_DIR}")
