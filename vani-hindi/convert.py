"""
SYSPIN (IISc) ke Coqui VITS Hindi model ko sherpa-onnx ke liye ONNX + tokens.txt
mein badalna, aur sherpa-onnx se hi bulwa kar test karna.

Usage: convert.py <src_dir> <out_dir> <release_dir> <name> <pt_file>
"""
import collections
import os
import sys
import time
import traceback

import numpy as np
import soundfile as sf
import torch

src, out, rel, name, ptname = sys.argv[1:6]
os.makedirs(out, exist_ok=True)
os.makedirs(rel, exist_ok=True)
logf = open(os.path.join(out, f"{name}-convert-log.txt"), "w", encoding="utf-8")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    logf.write(s + "\n")
    logf.flush()


def section(t):
    P("\n" + "=" * 20, t, "=" * 20)


TEST = ("नमस्ते, मैं वाणी की नई हिंदी आवाज़ हूँ। आज मौसम बहुत अच्छा है, "
        "और अभी सुबह के दस बजकर पंद्रह मिनट हुए हैं।")
TEST2 = "आपका पचास रुपये का भुगतान सफलतापूर्वक प्राप्त हो गया है।"

# ---------------------------------------------------------------- 1. files
section("FILES")
for f in ("jit_infer.py", "extra.py"):
    p = os.path.join(src, f)
    if os.path.exists(p):
        P(f"----- {f} -----")
        P(open(p, encoding="utf-8", errors="replace").read())
cp = os.path.join(src, "chars.txt")
if os.path.exists(cp):
    raw = open(cp, encoding="utf-8").read()
    P("chars.txt repr:", repr(raw))

# ---------------------------------------------------------------- 2. jit
section("JIT MODEL")
jit = None
sd = {}
try:
    jit = torch.jit.load(os.path.join(src, ptname), map_location="cpu")
    jit.eval()
    P("type:", type(jit), "original_name:", getattr(jit, "original_name", None))
    try:
        P("forward schema:", jit.forward.schema)
    except Exception as e:
        P("schema err:", e)
    for n, c in jit.named_children():
        P("child:", n, getattr(c, "original_name", type(c)))
    try:
        cm = jit.code_with_constants[1].const_mapping
        P("CONSTANTS:", {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in cm.items()})
    except Exception as e:
        P("const err:", e)
    try:
        P("CODE:\n", jit.code[:3000])
    except Exception as e:
        P("code err:", e)
    sd = jit.state_dict()
    P("tensors:", len(sd), "numel:", sum(v.numel() for v in sd.values()))
    P("prefixes:", collections.Counter(k.split(".")[0] for k in sd))
    for k in list(sd)[:25]:
        P("  ", k, tuple(sd[k].shape))
    P("emb:", [(k, tuple(v.shape)) for k, v in sd.items() if "emb" in k][:10])
except Exception:
    P(traceback.format_exc())

# ---------------------------------------------------------------- 3. reference
section("REFERENCE (their jit_infer.py)")
ns = {"__name__": "__jit_infer__"}
cwd = os.getcwd()
sys.path.insert(0, os.path.abspath(src))
os.chdir(src)
try:
    code = open("jit_infer.py", encoding="utf-8").read()
    exec(compile(code, "jit_infer.py", "exec"), ns)
except Exception:
    P(traceback.format_exc())
os.chdir(cwd)
for w in os.listdir(src):
    if w.endswith(".wav"):
        os.replace(os.path.join(src, w), os.path.join(out, f"{name}-reference-{w}"))
        P("reference wav saved:", w)

tokenizer = None
config = None
for k, v in ns.items():
    tn = type(v).__name__
    if tn == "TTSTokenizer":
        tokenizer = v
    if tn.endswith("Config") and hasattr(v, "model_args"):
        config = v
P("tokenizer:", tokenizer, "config:", type(config))
if tokenizer is not None:
    ch = tokenizer.characters
    P("characters class:", type(ch))
    P("vocab size:", len(ch.vocab))
    P("vocab:", ch.vocab)
    # SYSPIN ka apna tokenizer: hamesha blank daalta hai, BOS/EOS nahi
    ADD_BLANK = bool(getattr(tokenizer, "add_blank", True))
    USE_EOS_BOS = bool(getattr(tokenizer, "use_eos_bos", False))
    P("add_blank:", ADD_BLANK, "use_eos_bos:", USE_EOS_BOS)
    for a in ("pad", "eos", "bos", "blank", "punctuations"):
        P(a, repr(getattr(ch, a, None)))
    for a in ("pad_id", "eos_id", "bos_id", "blank_id"):
        try:
            P(a, getattr(ch, a))
        except Exception as e:
            P(a, "err", e)
    try:
        P("cleaner:", tokenizer.text_cleaner)
        P("test ids:", tokenizer.text_to_ids(TEST))
    except Exception:
        P(traceback.format_exc())
if config is not None:
    P("config.add_blank:", getattr(config, "add_blank", None),
      "enable_eos_bos_chars:", getattr(config, "enable_eos_bos_chars", None),
      "text_cleaner:", getattr(config, "text_cleaner", None))
    try:
        P("sample_rate:", config.audio.sample_rate)
        P("model_args:", config.model_args)
    except Exception as e:
        P("cfg err", e)

# ---------------------------------------------------------------- 4. coqui Vits
section("COQUI VITS + WEIGHTS")
model = None
SR = 22050
try:
    from TTS.tts.configs.vits_config import VitsConfig
    from TTS.tts.models.vits import Vits

    cfg = VitsConfig()
    cfg.model_args.init_discriminator = False
    SR = 22050
    tsd = sd
    emb = tsd.get("text_encoder.emb.weight")
    cfg.model_args.num_chars = int(emb.shape[0])
    P("num_chars from weights:", emb.shape[0])
    model = Vits(config=cfg)
    res = model.load_state_dict(tsd, strict=False)
    P("missing keys:", len(res.missing_keys), res.missing_keys[:40])
    P("unexpected keys:", len(res.unexpected_keys), res.unexpected_keys[:40])
    model.eval()
    if tokenizer is not None:
        ids = torch.LongTensor(tokenizer.text_to_ids(TEST)).unsqueeze(0)
        with torch.no_grad():
            o = model.inference(ids, aux_input={"x_lengths": torch.LongTensor([ids.shape[1]])})
        wav = o["model_outputs"].squeeze().cpu().numpy()
        sf.write(os.path.join(out, f"{name}-coqui-torch.wav"), wav, SR)
        P("coqui torch wav seconds:", len(wav) / SR)
except Exception:
    P(traceback.format_exc())
    model = None

# ---------------------------------------------------------------- 5. onnx
section("ONNX EXPORT")
onnx_path = os.path.join(rel, f"vani-hi-syspin-{name}.onnx")
tokens_path = os.path.join(rel, f"vani-hi-syspin-{name}-tokens.txt")
exported = False
if model is not None:
    try:
        model.export_onnx(output_path=onnx_path, verbose=False)
        exported = True
        P("exported:", onnx_path, os.path.getsize(onnx_path) / 1e6, "MB")
    except Exception:
        P(traceback.format_exc())


def add_meta(path):
    import onnx
    ch = tokenizer.characters
    meta = {
        "model_type": "vits",
        "comment": "coqui",
        "language": "Hindi",
        "voice": f"syspin-hi-{name}",
        "add_blank": int(ADD_BLANK),
        "n_speakers": 1,
        "sample_rate": SR,
        "use_eos_bos": int(USE_EOS_BOS),
        "blank_id": int(ch.blank_id) if ch.blank else 0,
        "pad_id": int(ch.pad_id) if ch.pad else 0,
        "frontend": "characters",
        "punctuation": " ".join(list(ch.punctuations or "")),
        "license": "CC-BY-4.0 (SYSPIN, IISc Bangalore - as labelled on Hugging Face)",
        "url": f"https://huggingface.co/SYSPIN/tts_vits_coquiai_Hindi{name.capitalize()}",
    }
    m = onnx.load(path)
    del m.metadata_props[:]
    for k, v in meta.items():
        e = m.metadata_props.add()
        e.key = k
        e.value = str(v)
    onnx.save(m, path)
    P("metadata:", meta)


def write_tokens():
    ch = tokenizer.characters
    c2i = {}
    for i, c in enumerate(ch.vocab):
        c2i[c] = i  # coqui: baad wala jeetta hai
    n = 0
    with open(tokens_path, "w", encoding="utf-8") as f:
        for c, i in sorted(c2i.items(), key=lambda x: x[1]):
            if c in ("<PAD>", "<EOS>", "<BOS>", "<BLNK>"):
                f.write(f"{c} {i}\n")
            elif len(c) != 1 or (c.isspace() and c != " "):
                P("skip token", repr(c), i)
                continue
            else:
                f.write(f"{c} {i}\n")
            n += 1
    P("tokens written:", n)


def sherpa_test(path, tag):
    import sherpa_onnx
    c = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(model=path, tokens=tokens_path),
            num_threads=2, debug=False, provider="cpu"),
        max_num_sentences=1)
    tts = sherpa_onnx.OfflineTts(c)
    for i, t in enumerate((TEST, TEST2)):
        t0 = time.time()
        a = tts.generate(t, sid=0, speed=1.0)
        dt = time.time() - t0
        secs = len(a.samples) / a.sample_rate
        sf.write(os.path.join(out, f"{name}-{tag}-sample{i + 1}.wav"),
                 np.array(a.samples, dtype=np.float32), a.sample_rate)
        P(f"sherpa {tag} sample{i + 1}: {secs:.2f}s audio, {dt:.2f}s compute, RTF {dt / max(secs, 1e-6):.3f}")


if exported and tokenizer is not None:
    try:
        write_tokens()
        add_meta(onnx_path)
        section("SHERPA-ONNX TEST (fp32)")
        sherpa_test(onnx_path, "sherpa-fp32")
    except Exception:
        P(traceback.format_exc())

    section("INT8")
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        q = onnx_path.replace(".onnx", ".int8.onnx")
        quantize_dynamic(onnx_path, q, weight_type=QuantType.QUInt8)
        add_meta(q)
        P("int8 size MB:", os.path.getsize(q) / 1e6)
        sherpa_test(q, "sherpa-int8")
    except Exception:
        P(traceback.format_exc())

section("DONE")
for f in sorted(os.listdir(rel)):
    P("release file:", f, round(os.path.getsize(os.path.join(rel, f)) / 1e6, 2), "MB")
