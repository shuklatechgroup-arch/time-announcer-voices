"""Pratham (Piper) bannaam SYSPIN: load time aur bolne ka time napna."""
import os
import sys
import time

import numpy as np
import onnx
import sherpa_onnx

W = sys.argv[1]
OUT = sys.argv[2]
log = open(os.path.join(OUT, "bench-log.txt"), "w", encoding="utf-8")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.write(s + "\n")
    log.flush()


SHORT = "नमस्ते"
TALKBACK = "सेटिंग्स, बटन, दो बार टैप करें"
PARA = ("आज सुबह से ही आसमान में बादल छाए हुए थे। दोपहर होते होते तेज़ बारिश शुरू हो गई। "
        "बच्चे स्कूल से लौटते समय पूरी तरह भीग गए। माँ ने उन्हें गरम दूध दिया और कपड़े बदलवाए। "
        "शाम को मौसम साफ़ हो गया और सबने छत पर बैठकर चाय पी।")

VOICES = {
    "pratham": dict(model=f"{W}/pratham/hi_IN-pratham-medium.onnx",
                    tokens=f"{W}/pratham/tokens.txt", data_dir=f"{W}/espeak-ng-data"),
    "syspin-male": dict(model=f"{W}/syspin/vani-hi-syspin-male.onnx",
                        tokens=f"{W}/syspin/vani-hi-syspin-male-tokens.txt", data_dir=""),
}
if os.path.exists(f"{W}/syspin/male.opt.onnx"):
    VOICES["syspin-male-opt"] = dict(model=f"{W}/syspin/male.opt.onnx",
                                     tokens=f"{W}/syspin/vani-hi-syspin-male-tokens.txt", data_dir="")

for name, v in VOICES.items():
    if not os.path.exists(v["model"]):
        P("MISSING", name, v["model"])
        continue
    m = onnx.load(v["model"], load_external_data=False)
    ops = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    P(f"\n#### {name}: {os.path.getsize(v['model']) / 1e6:.1f} MB, nodes={len(m.graph.node)}, "
      f"top ops={sorted(ops.items(), key=lambda x: -x[1])[:8]}")
    for threads in (2, 4):
        t0 = time.time()
        tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=v["model"], tokens=v["tokens"], data_dir=v["data_dir"],
                    noise_scale_w=0.8),
                num_threads=threads, provider="cpu"),
            max_num_sentences=1))
        load = time.time() - t0
        P(f"[{name} threads={threads}] LOAD {load:.2f}s")
        for label, text in (("first-short", SHORT), ("short", SHORT), ("talkback", TALKBACK),
                            ("paragraph", PARA), ("paragraph-danda-to-dot", PARA.replace("।", "."))):
            t0 = time.time()
            a = tts.generate(text, sid=0, speed=1.0)
            dt = time.time() - t0
            secs = len(a.samples) / a.sample_rate
            P(f"   {label:24s} compute {dt:6.2f}s  audio {secs:6.2f}s")
        # pehla vakya alag se (agar app vakya-vakya bole to pehli awaaz kitni der mein)
        first = PARA.split("।")[0] + "।"
        t0 = time.time()
        tts.generate(first, sid=0, speed=1.0)
        P(f"   {'first-sentence-only':24s} compute {time.time() - t0:6.2f}s")
        del tts
