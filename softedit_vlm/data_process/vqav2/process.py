"""Process VQA v2: 10-annotator answers → parquet."""
import json
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from data_process.common import validate_df, write_parquet

RAW_DIR = "data_raw/vqav2"
IMG_DIR = "data/images/vqav2"

SOURCES = [
    {
        "annotations": "v2_mscoco_train2014_annotations.json",
        "questions": "v2_OpenEnded_mscoco_train2014_questions.json",
        "img_subdir": "train2014",
        "img_prefix": "COCO_train2014_",
    },
    {
        "annotations": "v2_mscoco_val2014_annotations.json",
        "questions": "v2_OpenEnded_mscoco_val2014_questions.json",
        "img_subdir": "val2014",
        "img_prefix": "COCO_val2014_",
    },
]


def load_source(cfg: dict) -> list:
    with open(os.path.join(RAW_DIR, cfg["annotations"])) as f:
        annotations = json.load(f)["annotations"]
    with open(os.path.join(RAW_DIR, cfg["questions"])) as f:
        questions = {q["question_id"]: q for q in json.load(f)["questions"]}

    rows = []
    for ann in annotations:
        qid = ann["question_id"]
        q = questions.get(qid)
        if q is None:
            continue

        answer_list = [a["answer"].strip().lower() for a in ann["answers"]]
        answer_counts = {}
        for a in answer_list:
            answer_counts[a] = answer_counts.get(a, 0) + 1

        image_id = ann["image_id"]
        image_path = os.path.join(
            IMG_DIR, cfg["img_subdir"],
            f"{cfg['img_prefix']}{image_id:012d}.jpg"
        )

        rows.append({
            "uid": f"vqav2_{qid}",
            "image_path": image_path,
            "question": q["question"],
            "answer_counts": json.dumps(answer_counts),
        })
    return rows


def main():
    all_rows = []
    for cfg in SOURCES:
        print(f"Loading {cfg['annotations']}...")
        all_rows.extend(load_source(cfg))

    df = pd.DataFrame(all_rows)
    validate_df(df, "vqav2")
    path = write_parquet(df, "vqav2", suffix="_raw")
    print(f"Written to {path}")


if __name__ == "__main__":
    main()
