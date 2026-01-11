import json
from pathlib import Path
from tqdm import tqdm
import sacrebleu
import pandas as pd
import torch

from comet import download_model, load_from_checkpoint

####################################
# Paths
####################################

ROOT = Path("outputs")

SRC_MAP = {
    "de-en": Path("/home/baban/scripts/survey/data/wmt14/de-en/test.json"),
    "en-de": Path("/home/baban/scripts/survey/data/wmt14/en-de/test.json"),
}

####################################
# Load source sentences
####################################

def load_sources(src_path):
    with src_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [ex["input"].strip() for ex in data]

SOURCE_CACHE = {
    k: load_sources(v) for k, v in SRC_MAP.items()
}

####################################
# Metric functions
####################################

def compute_bleu_chrf(preds, refs):
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    chrf = sacrebleu.corpus_chrf(preds, [refs])
    return bleu.score, chrf.score

def compute_comet(model, srcs, preds, refs, batch_size=16):
    data = [
        {"src": s, "mt": p, "ref": r}
        for s, p, r in zip(srcs, preds, refs)
    ]

    with torch.no_grad():
        out = model.predict(
            data,
            batch_size=batch_size,
            gpus=1 if torch.cuda.is_available() else 0,
        )

    return sum(out["scores"]) / len(out["scores"])

####################################
# Load COMET-22 once
####################################

print("Loading COMET-22 model...")
model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(model_path)
comet_model.eval()

####################################
# Main evaluation loop
####################################

results = []
jsonl_files = list(ROOT.rglob("generated_predictions.jsonl"))

print(f"Found {len(jsonl_files)} prediction files\n")

for jsonl_path in tqdm(jsonl_files, desc="Evaluating"):
    preds, refs = [], []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            obj = json.loads(line)
            preds.append(obj["predict"].strip())
            refs.append(obj["label"].strip())

    # Detect direction
    parts = jsonl_path.relative_to(ROOT).parts
    if "de-en" in parts:
        direction = "de-en"
    elif "en-de" in parts:
        direction = "en-de"
    else:
        raise RuntimeError(f"Cannot infer direction from path: {jsonl_path}")

    srcs = SOURCE_CACHE[direction]

    if len(srcs) != len(preds):
        raise ValueError(
            f"Source/pred mismatch for {jsonl_path}: "
            f"{len(srcs)} vs {len(preds)}"
        )

    # Metrics
    bleu, chrf = compute_bleu_chrf(preds, refs)
    comet = compute_comet(comet_model, srcs, preds, refs)

    result = {
        "path": str(jsonl_path.relative_to(ROOT)),
        "direction": direction,
        "bleu": round(bleu, 2),
        "chrf": round(chrf, 2),
        "comet22": round(float(comet), 4),
        "num_samples": len(preds),
    }

    # Metadata parsing
    if "oneshot" in parts:
        result["shots"] = "1"
    elif "twoshot" in parts:
        result["shots"] = "2"
    elif "threeshot" in parts:
        result["shots"] = "3"
    else:
        result["shots"] = "full"

    if "random" in parts:
        result["selection"] = "random"
    elif "similarity" in parts:
        result["selection"] = "similarity"
    else:
        result["selection"] = "n/a"

    if "lora" in parts:
        result["training"] = "lora"
    elif "full" in parts:
        result["training"] = "full"
    else:
        result["training"] = "n/a"

    results.append(result)

####################################
# Save report
####################################

df = pd.DataFrame(results).sort_values(
    by=["direction", "shots", "selection", "training"],
    na_position="last",
)

df.to_csv("bleu_chrf_comet22_report.csv", index=False)

print("\n=== BLEU + chrF + COMET-22 REPORT ===\n")
print(df.to_string(index=False))
print("\nSaved to: bleu_chrf_comet22_report.csv")
