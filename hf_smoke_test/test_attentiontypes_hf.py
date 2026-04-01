import os

from transformers import AutoModelForSequenceClassification, AutoTokenizer


REPO_ID = os.environ.get("HF_REPO_ID", "owenarink/attentiontypes-commonsense")


def main():
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(REPO_ID, trust_remote_code=True)

    false_sentence = "The sun rises in the west."
    options = [
        "The sun rises in the east.",
        "The sun sets in the west.",
        "The sun shines at night.",
    ]

    encoded = tokenizer(
        [false_sentence] * len(options),
        options,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    scores = model(**encoded).logits.squeeze(-1)

    assert scores.shape == (3,), f"Expected 3 option scores, got {tuple(scores.shape)}"
    print("repo:", REPO_ID)
    print("scores:", scores.tolist())
    print("best_option_index:", int(scores.argmax().item()))


if __name__ == "__main__":
    main()
