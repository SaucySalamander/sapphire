# Ternary corpus manifests

These manifests are intended for `--convert-ternary` runs.

## Files

- `27b_high_signal_calib_manifest.csv`
  - 50% FineWeb-Edu prose
  - 25% The Stack v2 logic/code replay (C/C++)
  - 25% Proof-Pile-2 scientific/math replay
- `27b_high_signal_validation_manifest.csv`
  - GSM8K prompts
  - HumanEval prompts

## Important constraints

- The manifest file itself must stay local because Sapphire rejects remote manifest paths.
- Manifest entries point at local plain-text files generated under `./corpora/`.
- The runtime currently expects plain text, not Parquet or JSON dataset shards.

## Prepare corpora

Install the helper dependencies and export the dataset shards:

```bash
./.venv/bin/python -m pip install -U -r scripts/requirements-ternary-corpus.txt
./.venv/bin/python scripts/prepare_ternary_corpus.py
```

Note: `bigcode/the-stack-v2` is gated on Hugging Face. Accept the dataset terms and
authenticate first (for example with `huggingface-cli login` or `HF_TOKEN`) before
running the export helper.

Also note: `HF_TOKEN` only unlocks The Stack v2 metadata. The helper now tries
public unsigned Software Heritage object access first and falls back to the
public HTTPS content endpoint. Explicit AWS credentials are optional.

To avoid rescanning The Stack metadata stream on every export, build the local
metadata cache once and then run the normal export:

```bash
./.venv/bin/python scripts/prepare_ternary_corpus.py --stack-cache-only
./.venv/bin/python scripts/prepare_ternary_corpus.py
```

The helper also caches sampled text for every corpus source under `./.cache/corpus-sources`
so repeated runs can reuse the already-built source samples.

## Run conversion

```bash
./out/sapphire \
  -m gemma-3-27b-it \
  --convert-ternary \
  --output ./out/gemma-3-27b-it-ternary \
  --calib-manifest ./configs/corpus/27b_high_signal_calib_manifest.csv \
  --validation-manifest ./configs/corpus/27b_high_signal_validation_manifest.csv \
  --calibration-samples 8 \
  --validation-samples 64 \
  --validate-every 32 \
  --kl-weight 0.05
```
