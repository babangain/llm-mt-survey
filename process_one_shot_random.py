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
# PROMPT CONSTRUCTION
# ------------------------
def build_prompt_with_demo(demo_ex, instruction, src):
    return (
        "You are a translation assistant.\n\n"
        "Example:\n"
        f"Instruction: {demo_ex['instruction']}\n"
        f"Input: {demo_ex['input']}\n"
        f"Output: {demo_ex['output']}\n\n"
        "Now translate the following sentence.\n"
        f"Instruction: {instruction}\n"
    )

# ------------------------
# CREATE TEST SET WITH RANDOM DEMO
# ------------------------
def create_test_with_random_demo(
    test_data,
    valid_data,
    base_instruction
):
    processed = []

    for ex in tqdm(test_data):
        src = ex["input"].strip()
        tgt = ex["output"].strip()

        if not src or not tgt:
            continue
        if len(src.split()) < 5 or len(tgt.split()) < 5:
            continue

        # Random validation example PER test instance
        demo_ex = random.choice(valid_data)

        prompt = build_prompt_with_demo(
            demo_ex=demo_ex,
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
# BUILD TEST SETS WITH DEMOS
# ------------------------
print("🧪 Creating test sets with random in-context examples...")

test_en_de_with_demo = create_test_with_random_demo(
    test_data=en_de_test,
    valid_data=en_de_valid,
    base_instruction=INSTR_EN_DE
)

test_de_en_with_demo = create_test_with_random_demo(
    test_data=de_en_test,
    valid_data=de_en_valid,
    base_instruction=INSTR_DE_EN
)

# ------------------------
# SAVE
# ------------------------
save_json(
    test_en_de_with_demo,
    f"{BASE_DIR}/one-shot/random/en-de/test.json"
)

save_json(
    test_de_en_with_demo,
    f"{BASE_DIR}/one-shot/random/de-en/test.json"
)

print("✅ Done.")
print(f"📁 Files written to: {BASE_DIR}")
