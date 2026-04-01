# Hugging Face Smoke Test

This folder contains a minimal smoke test for the published Hugging Face model.

It checks that:

- the tokenizer loads from the Hub
- the custom model loads with `trust_remote_code=True`
- a three-option commonsense example produces three scores

Run:

```bash
python hf_smoke_test/test_attentiontypes_hf.py
```

Override the repo if needed:

```bash
HF_REPO_ID=owenarink/attentiontypes-commonsense python hf_smoke_test/test_attentiontypes_hf.py
```
