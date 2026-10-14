#!/usr/bin/env bash
# Convert WildGuard (allenai/wildguard) to a local Q8_0 GGUF (task 2.11b).
#
# No GGUF of WildGuard is published, so this builds one on this machine.
# Everything stays local under ~/.cache/meta-eval (nothing is uploaded).
#
# Needs:
#   - HF_TOKEN (environment or the repo's .env) for an account that accepted
#     the AI2 terms at https://huggingface.co/allenai/wildguard (gated).
#   - llama.cpp from Homebrew (`brew install llama.cpp`): llama-tokenize.
#   - uv (for a throwaway venv with the converter's deps: torch, transformers).
#
# Output: ~/.cache/meta-eval/gguf/wildguard-Q8_0.gguf (the model_path in
# config/judges.yaml). About 15 GB download + 7.7 GB output.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CACHE="${META_EVAL_CACHE:-$HOME/.cache/meta-eval}"
# Pinned: the WildGuard commit checked on 2026-10-14, and the llama.cpp tag of
# the installed Homebrew build, so the converter matches the runtime.
WILDGUARD_REV="cbba4823f3e8020e5a74a5e29bf85072def6f2ff"
LLAMACPP_TAG="${LLAMACPP_TAG:-b$(brew list --versions llama.cpp | awk '{print $2}')}"
OUT="$CACHE/gguf/wildguard-Q8_0.gguf"

if [[ -z "${HF_TOKEN:-}" && -f "$REPO_DIR/.env" ]]; then
  HF_TOKEN="$(grep -E '^HF_TOKEN=' "$REPO_DIR/.env" | cut -d= -f2- | tr -d '"'"'")"
fi
[[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN is not set" >&2; exit 1; }
export HF_TOKEN

if [[ -f "$OUT" ]]; then
  echo "$OUT exists; delete it to rebuild." >&2
  exit 0
fi
mkdir -p "$CACHE/gguf"

echo "1/4 llama.cpp converter at $LLAMACPP_TAG"
if [[ ! -d "$CACHE/llama.cpp" ]]; then
  git clone --depth 1 --branch "$LLAMACPP_TAG" https://github.com/ggml-org/llama.cpp "$CACHE/llama.cpp"
fi

echo "2/4 converter venv"
if [[ ! -x "$CACHE/convert-venv/bin/python" ]]; then
  uv venv --python 3.12 "$CACHE/convert-venv"
  VIRTUAL_ENV="$CACHE/convert-venv" uv pip install \
    -r "$CACHE/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt"
fi
PY="$CACHE/convert-venv/bin/python"

echo "3/4 download allenai/wildguard@$WILDGUARD_REV"
"$PY" - "$CACHE/hf/wildguard" "$WILDGUARD_REV" <<'EOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download("allenai/wildguard", revision=sys.argv[2], local_dir=sys.argv[1])
EOF

echo "4/4 convert to Q8_0"
"$PY" "$CACHE/llama.cpp/convert_hf_to_gguf.py" "$CACHE/hf/wildguard" \
  --outtype q8_0 --outfile "$OUT.tmp"
# Check, before the file is put in place, that the GGUF tokenizes the prompt
# like the HF slow tokenizer WildGuard was trained with. llama.cpp adds one
# space token after each [INST] / [/INST]; WildGuardJudge.prompt_tokens
# removes it, so this check removes it too and then needs an exact match.
echo "Tokenizer check"
SAMPLE=$'<|user|>\n[INST] Human user:\nhi\n\nAnswers: [/INST]\n<|assistant|>\n'
GGUF_IDS="$(llama-tokenize -m "$OUT.tmp" -p "$SAMPLE" --ids --log-disable 2>/dev/null | tr -d '[],')"
"$PY" - "$CACHE/hf/wildguard" "$SAMPLE" "$GGUF_IDS" <<'PYEOF'
import sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1], use_fast=False)
hf = tok(sys.argv[2], add_special_tokens=True)["input_ids"]
gguf = [int(x) for x in sys.argv[3].split()]
control = set(tok.convert_tokens_to_ids(["[INST]", "[/INST]"]))
space = tok.convert_tokens_to_ids("▁")
fixed = [t for i, t in enumerate(gguf) if not (i and gguf[i - 1] in control and t == space)]
print("HF:  ", hf)
print("GGUF:", fixed)
if fixed != hf:
    sys.exit("ERROR: token ids differ after the [INST] space fix; the GGUF is not used.")
PYEOF
mv "$OUT.tmp" "$OUT"
echo "Done: $OUT"
echo "The HF checkpoint ($CACHE/hf/wildguard, ~15 GB) can be deleted now."
