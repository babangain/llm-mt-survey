import os
import json
import random
from datasets import load_dataset
from tqdm import tqdm

SEED = 42
TRAIN_SAMPLES = 100_000
BASE_DIR = "./data/wmt14"

random.seed(SEED)

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def process_split(dataset, src_lang, tgt_lang, instruction):
    processed = []
    for ex in tqdm(dataset):
        trans = ex["translation"]
        src = trans.get(src_lang, "").strip()
        tgt = trans.get(tgt_lang, "").strip()

        if not src or not tgt:
            continue

        # light quality filter (recommended)
        if len(src.split()) < 5 or len(tgt.split()) < 5:
            continue

        processed.append({
            "instruction": instruction,
            "input": src,
            "output": tgt
        })
    return processed

# Load WMT14 de-en
dataset = load_dataset("wmt/wmt14", "de-en")

print(dataset)

# ------------------------
# TRAIN (sample 100k)
# ------------------------
train_ds = dataset["train"].shuffle(seed=SEED)
train_ds = train_ds.select(range(min(TRAIN_SAMPLES, len(train_ds))))

# EN → DE
train_en_de = process_split(
    train_ds,
    src_lang="en",
    tgt_lang="de",
    instruction="Translate the following sentence from English to German."
)

# DE → EN
train_de_en = process_split(
    train_ds,
    src_lang="de",
    tgt_lang="en",
    instruction="Translate the following sentence from German to English."
)

# ------------------------
# VALIDATION (full)
# ------------------------
valid_ds = dataset["validation"]

valid_en_de = process_split(
    valid_ds,
    "en", "de",
    "Translate the following sentence from English to German."
)

valid_de_en = process_split(
    valid_ds,
    "de", "en",
    "Translate the following sentence from German to English."
)

# ------------------------
# TEST (full)
# ------------------------
test_ds = dataset["test"]

test_en_de = process_split(
    test_ds,
    "en", "de",
    "Translate the following sentence from English to German."
)

test_de_en = process_split(
    test_ds,
    "de", "en",
    "Translate the following sentence from German to English."
)

# ------------------------
# SAVE FILES
# ------------------------
save_json(train_en_de, f"{BASE_DIR}/en-de/train.json")
save_json(valid_en_de, f"{BASE_DIR}/en-de/validation.json")
save_json(test_en_de,  f"{BASE_DIR}/en-de/test.json")

save_json(train_de_en, f"{BASE_DIR}/de-en/train.json")
save_json(valid_de_en, f"{BASE_DIR}/de-en/validation.json")
save_json(test_de_en,  f"{BASE_DIR}/de-en/test.json")

print("✅ Done. Files written to:", BASE_DIR)
