#!/usr/bin/env bash
set -x
cd "$GITHUB_WORKSPACE"
pip install -q sherpa-onnx onnx onnxruntime onnxsim soundfile
pip list 2>/dev/null | grep -i -E '^(onnx|onnxruntime|onnxsim|sherpa)'
W=work
mkdir -p $W/syspin $W/pratham
HF=https://huggingface.co/Shuklatechgroup/vani-tts-voices/resolve/main
curl -sSL --fail -o $W/syspin/vani-hi-syspin-male.onnx $HF/syspin-hindi/vani-hi-syspin-male.onnx
curl -sSL --fail -o $W/syspin/vani-hi-syspin-male-tokens.txt $HF/syspin-hindi/vani-hi-syspin-male-tokens.txt
curl -sSL --fail -o $W/pratham.zip $HF/hi_IN-pratham-medium.zip
curl -sSL --fail -o $W/espeak.zip $HF/espeak-ng-data.zip
(cd $W && unzip -q -o pratham.zip -d pratham_raw && unzip -q -o espeak.zip -d espeak_raw)
find $W/pratham_raw -maxdepth 3 | head; find $W/espeak_raw -maxdepth 2 | head -5
cp "$(find $W/pratham_raw -name '*.onnx' | head -1)" $W/pratham/hi_IN-pratham-medium.onnx
cp "$(find $W/pratham_raw -name 'tokens.txt' | head -1)" $W/pratham/tokens.txt
ED=$(dirname "$(find $W/espeak_raw -name 'phontab' | head -1)")
ln -sfn "$(realpath $ED)" $W/espeak-ng-data
ls -la $W/pratham $W/espeak-ng-data/ | head

# ek simplify kiya hua version bhi try karo (load jaldi hota hai ya nahi)
python - <<'EOF'
import onnx, time
from onnxsim import simplify
t=time.time()
m = onnx.load("work/syspin/vani-hi-syspin-male.onnx")
meta = list(m.metadata_props)
try:
    ms, ok = simplify(m, test_input_shapes=None, skip_shape_inference=False)
    print("onnxsim ok:", ok, "nodes", len(m.graph.node), "->", len(ms.graph.node), f"{time.time()-t:.1f}s")
    del ms.metadata_props[:]
    for p in meta:
        e = ms.metadata_props.add(); e.key = p.key; e.value = p.value
    onnx.save(ms, "work/syspin/male.opt.onnx")
except Exception as e:
    print("onnxsim failed:", type(e).__name__, str(e)[:300])
EOF

python vani-hindi/bench.py $W vani-hindi/out
