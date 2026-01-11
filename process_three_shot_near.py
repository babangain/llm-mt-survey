import os
import json
import random
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------
# CONFIG
# ------------------------
SEED = 42
BASE_DIR = "./data/wmt14"
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64
NUM_SHOTS = 3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
def build_prompt_with_demos(demo_exs, instruction):
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
# EMBEDDING UTILS
# ------------------------
print("🔌 Loading embedding model...")
embedder = SentenceTransformer(EMB_MODEL_NAME, device=DEVICE)

def compute_embeddings(texts):
    """
    Compute normalized sentence embeddings in batches.
    """
    return embedder.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

def build_validation_index(valid_data):
    """
    Build embedding index for validation set.
    """
    valid_srcs = [ex["input"].strip() for ex in valid_data]
    valid_embs = compute_embeddings(valid_srcs)
    return valid_embs

def get_topk_similar_examples(
    src_sentence,
    valid_data,
    valid_embeddings,
    k=3
):
    """
    Retrieve top-k most similar validation examples.
    """
    src_emb = compute_embeddings([src_sentence])  # (1, D)
    sims = cosine_similarity(src_emb, valid_embeddings)[0]
    topk_idx = np.argsort(sims)[-k:]# [::-1]
    return [valid_data[i] for i in topk_idx]

# ------------------------
# CREATE TEST SET WITH SIMILARITY-BASED DEMOS
# ------------------------
def create_test_with_similar_demos(
    test_data,
    valid_data,
    valid_embeddings,
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

        demo_exs = get_topk_similar_examples(
            src_sentence=src,
            valid_data=valid_data,
            valid_embeddings=valid_embeddings,
            k=num_shots
        )

        prompt = build_prompt_with_demos(
            demo_exs=demo_exs,
            instruction=base_instruction
        )

        processed.append({
            "instruction": prompt,
            "input": src,
            "output": tgt
        })

    return processed

# ------------------------
# LOAD DATA
# ------------------------
print("📥 Loading WMT14 JSON data...")

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
# BUILD VALIDATION INDICES (ONCE)
# ------------------------
print("🔢 Computing validation embeddings (EN→DE)...")
en_de_valid_embs = build_validation_index(en_de_valid)

print("🔢 Computing validation embeddings (DE→EN)...")
de_en_valid_embs = build_validation_index(de_en_valid)

# ------------------------
# BUILD TEST SETS
# ------------------------
print("🧪 Creating similarity-based three-shot test sets...")

test_en_de_with_demo = create_test_with_similar_demos(
    test_data=en_de_test,
    valid_data=en_de_valid,
    valid_embeddings=en_de_valid_embs,
    base_instruction=INSTR_EN_DE,
    num_shots=NUM_SHOTS
)

test_de_en_with_demo = create_test_with_similar_demos(
    test_data=de_en_test,
    valid_data=de_en_valid,
    valid_embeddings=de_en_valid_embs,
    base_instruction=INSTR_DE_EN,
    num_shots=NUM_SHOTS
)

# ------------------------
# SAVE
# ------------------------
save_json(
    test_en_de_with_demo,
    f"{BASE_DIR}/three-shot/similarity/en-de/test.json"
)

save_json(
    test_de_en_with_demo,
    f"{BASE_DIR}/three-shot/similarity/de-en/test.json"
)

print("✅ Done.")
print(f"📁 Files written to: {BASE_DIR}/three-shot/similarity/")
