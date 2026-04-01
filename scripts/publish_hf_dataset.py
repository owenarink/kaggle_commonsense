import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_OUTPUT_DIR = ROOT / "hf_export" / "kaggle-commonsense-dataset"
DEFAULT_REPO_ID = "owenarink/kaggle-commonsense"
DATASET_FILES = [
    "train_data.csv",
    "train_answers.csv",
    "test_data.csv",
    "sample.csv",
]


DATASET_CARD = """---
pretty_name: Kaggle Commonsense
language:
- en
license: unknown
task_categories:
- text-classification
task_ids:
- multiple-choice-qa
size_categories:
- 10K<n<100K
---

# Kaggle Commonsense

This dataset contains the course-project data used in the `kaggle_commonsense` repository for commonsense reasoning with multiple-choice repairs.

## Dataset Description

Each example contains:

- an `id`
- one false sentence in `FalseSent`
- three candidate repairs in `OptionA`, `OptionB`, and `OptionC`

For training, the correct answer label is provided in `train_answers.csv`.

## Files

- `train_data.csv`: training inputs
- `train_answers.csv`: training labels
- `test_data.csv`: test inputs
- `sample.csv`: sample submission format

## Task

The task is to select the correct answer (`A`, `B`, or `C`) that best repairs the false sentence.

Example:

- False sentence: `The sun rises in the west.`
- A: `The sun rises in the east.`
- B: `The sun sets in the west.`
- C: `The sun shines at night.`

Correct answer: `A`

## Notes

This is a task-specific academic dataset from a course competition setup. It should not be treated as a broadly standardized benchmark.
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Publish the Kaggle Commonsense CSV files to a Hugging Face dataset repo.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--push", action="store_true")
    return parser.parse_args()


def build_dataset_folder(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in DATASET_FILES:
        shutil.copy2(DATA_DIR / name, output_dir / name)
    (output_dir / "README.md").write_text(DATASET_CARD)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    build_dataset_folder(output_dir)
    print(f"Prepared dataset export in: {output_dir}")

    if args.push:
        api = HfApi()
        api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=str(output_dir),
            commit_message="Upload Kaggle Commonsense dataset",
        )
        print(f"Pushed dataset to Hugging Face Hub: {args.repo_id}")
    else:
        print("Push skipped. Re-run with --push to upload the dataset repo.")


if __name__ == "__main__":
    main()
