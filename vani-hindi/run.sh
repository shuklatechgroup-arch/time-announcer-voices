#!/usr/bin/env bash
# SYSPIN (IISc) Hindi VITS voices -> sherpa-onnx (ONNX + tokens.txt)
set -x
cd "$GITHUB_WORKSPACE"

python -m pip install --upgrade pip -q
printf 'torch==2.4.1\ntorchaudio==2.4.1\n' > /tmp/constraints.txt
pip install -q torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu
pip install -q -c /tmp/constraints.txt coqui-tts onnx onnxruntime sherpa-onnx soundfile
pip list 2>/dev/null | grep -i -E '^(torch|torchaudio|coqui|onnx|onnxruntime|numpy|transformers|sherpa)'

for V in HindiMale:hi_male_vits_30hrs.pt:male HindiFemale:hi_female_vits_30hrs.pt:female; do
  IFS=: read -r REPO PT NAME <<< "$V"
  D="work/$NAME"
  mkdir -p "$D"
  for F in chars.txt extra.py jit_infer.py "$PT"; do
    curl -sSL --fail --retry 3 -o "$D/$F" \
      "https://huggingface.co/SYSPIN/tts_vits_coquiai_${REPO}/resolve/main/${F}" \
      || echo "DOWNLOAD FAILED: $REPO/$F"
  done
  ls -la "$D"
  python vani-hindi/convert.py "$D" vani-hindi/out work/release "$NAME" "$PT"
done

ls -la vani-hindi/out work/release
