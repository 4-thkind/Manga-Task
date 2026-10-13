# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "torch",
#     "torchvision",
#     "transformers==4.57.1",
#     "peft==0.17.1",
#     "accelerate==1.10.1",
#     "gdown==5.2.0",
#     "pandas",
#     "pillow",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import json
    import random
    import re
    import shutil
    import subprocess
    import sys
    import time
    import zipfile
    import difflib
    from collections import Counter

    import numpy as np
    from scipy.cluster.hierarchy import fcluster, linkage
    import importlib.util
    from importlib import metadata
    from pathlib import Path

    import marimo as mo

    SPECIAL = {"NARRATION", "UNKNOWN"}
    # Ways to combine Qwen with Magi; each is scored on val and the best one makes the test predictions.
    VARIANTS = {
        "Qwen only": None,
        "Magi speakers": {},
        "Magi speakers + Magi order": {"reorder": True},
        "Magi speakers + missed text": {"insert": True},
        "Magi speakers + order + missed text": {"reorder": True, "insert": True},
        "Magi speakers + drop non-story text": {"drop": True},
    }
    PROMPT = (
        "The images above are consecutive pages of an English manga. Transcribe the story text "
        "of PAGE {k} ONLY: dialogue, thoughts, narration, clearly spoken screams or vocalisations, "
        "and punctuation-only balloons such as ... or ?!. Skip sound effects, titles, credits, "
        "page numbers, signs, writing on objects, and text inside letters, phones or documents. "
        "Read in manga order (right-to-left, top-to-bottom, following the panel layout). "
        "Write one balloon per line as `speaker: text`. Use character1, character2, ... and reuse "
        "the labels from the earlier transcript for the same characters. Use NARRATION for "
        "narrator boxes and UNKNOWN when the speaker cannot be told. Join wrapped lines, keep "
        "stutters and punctuation. If page {k} has no story text, write NONE.\n\n"
        "Transcript of the earlier pages:\n{context}"
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # BYTE Manga Task [Advanced] — v4: Qwen2.5-VL-7B text + Magi v2 speakers, order and recall

    **Text** comes from the v2 recipe: Qwen2.5-VL-7B + LoRA, one page per call with the earlier
    pages as context, 4 epochs (a 2-epoch run had lower val loss but text_order fell to 0.76).

    **Why Magi.** Error analysis of v2 on val: perfect line order alone would lift text_order
    from 0.89 to 0.93, removing extra lines +0.01, recovering the 41 missed lines (of 362) +0.02.
    Speakers were the weakest part (0.31). Magi v2, an open manga model, gives text boxes in
    reading order, a text classifier, its own OCR, character boxes, speaker links and character
    embeddings. Each Qwen line is matched to a Magi text box, and then:

    * speaker = the Magi character linked to that box, grouped across the 3 pages by clustering
      the character embeddings (threshold T and text match M tuned on the 64 train sequences)
    * optionally, lines are re-sorted into Magi's reading order
    * optionally, Magi boxes that Qwen missed are added back with Magi's OCR text

    Every combination is scored on the 16 val sequences and the best one makes the test
    predictions. If Magi fails to load, the notebook falls back to Qwen only.

    **Before running:** GPU on (notebook specs → GPUs), then *Run all*. The last cell gives the download.
    """)
    return


@app.cell
def _():
    DATA_URL = "https://drive.google.com/file/d/1bGZSQoiGDyyJ_L0tTDXEB4bwdSE0TwZA/view?usp=sharing"

    MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
    WORK = Path("work").resolve()
    OUT = WORK / "outputs"
    OUT.mkdir(parents=True, exist_ok=True)
    SEED = 0
    N_VAL = 16
    EPOCHS = 4  # 2 epochs had lower val loss but text_order fell from 0.89 to 0.76 (v3 first run)
    LR = 1e-4
    GRAD_ACCUM = 8
    LORA_R = 16
    MAX_PIXELS = 1800 * 28 * 28  # ~1.4M px: every page in this dataset stays at native resolution
    MAX_NEW_TOKENS = 700  # one page; the longest gold page is ~250 tokens
    mo.md(f"Model `{MODEL_ID}` · output folder `{OUT}`")
    return (
        DATA_URL,
        EPOCHS,
        GRAD_ACCUM,
        LORA_R,
        LR,
        MAX_NEW_TOKENS,
        MAX_PIXELS,
        MODEL_ID,
        N_VAL,
        OUT,
        SEED,
        WORK,
    )


@app.cell
def _():
    # Install pinned versions only if missing or different (molab preinstalls torch).
    _want = {
        "transformers": "4.57.1",
        "peft": "0.17.1",
        "accelerate": "1.10.1",
        "gdown": "5.2.0",
    }
    _todo = []
    for _name, _ver in _want.items():
        try:
            if metadata.version(_name) != _ver:
                _todo.append(f"{_name}=={_ver}")
        except metadata.PackageNotFoundError:
            _todo.append(f"{_name}=={_ver}")
    for _name in ("torch", "torchvision", "pandas", "pillow", "numpy", "scipy", "einops", "pulp", "shapely", "networkx", "matplotlib", "timm"):
        try:
            metadata.version(_name)
        except metadata.PackageNotFoundError:
            _todo.append(_name)
    if _todo:
        _r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *_todo], capture_output=True, text=True)
        if _r.returncode != 0:  # some sandboxes ship without pip
            _r = subprocess.run(["uv", "pip", "install", "--python", sys.executable, *_todo], capture_output=True, text=True)
        assert _r.returncode == 0, _r.stderr[-2000:]
    deps_ready = True
    mo.md(f"Installed: `{', '.join(_todo)}`" if _todo else "All packages already present.")
    return (deps_ready,)


@app.cell
def _(DATA_URL, WORK, deps_ready):
    import gdown

    assert deps_ready

    def _find_root(base):
        for _p in sorted(Path(base).rglob("sequences.json")):
            return _p.parent
        return None

    DATA = _find_root(WORK / "data") or _find_root(Path("."))
    if DATA is None:
        _local_zips = sorted(Path(".").glob("*.zip"))  # also works if the zip was uploaded next to the notebook
        if _local_zips:
            _zip = _local_zips[0]
        else:
            _zip = WORK / "dataset.zip"
            gdown.download(url=DATA_URL, output=str(_zip), fuzzy=True, quiet=False)
        with zipfile.ZipFile(_zip) as _z:
            _z.extractall(WORK / "data")
        DATA = _find_root(WORK / "data")
    assert DATA is not None, "sequences.json not found in the downloaded data"

    _spec = importlib.util.spec_from_file_location("byte_score", DATA / "score.py")
    scorer = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(scorer)
    mo.md(f"Dataset at `{DATA}`")
    return DATA, scorer


@app.cell
def _(DATA, N_VAL, SEED):
    import pandas as pd

    sequences = {s["sequence_id"]: s for s in json.loads((DATA / "sequences.json").read_text())}
    labels = {}
    for _line in (DATA / "development" / "labels.jsonl").read_text().splitlines():
        if _line.strip():
            _row = json.loads(_line)
            labels[_row["sequence_id"]] = _row["pages"]
    test_ids = [json.loads(_l)["sequence_id"] for _l in (DATA / "sample_submission.jsonl").read_text().splitlines() if _l.strip()]

    # Split by whole sequence, never by page. Same split as v1 so the numbers are comparable.
    _dev = sorted(labels)
    random.Random(SEED).shuffle(_dev)
    val_ids, train_ids = sorted(_dev[:N_VAL]), sorted(_dev[N_VAL:])

    _lines = [r for p in labels.values() for page in p for r in page]
    stats = pd.DataFrame(
        {
            "value": [
                len(labels),
                len(test_ids),
                len(train_ids),
                len(val_ids),
                len(_lines),
                sum(not page for p in labels.values() for page in p),
                sum(r["speaker"] == "NARRATION" for r in _lines),
                sum(r["speaker"] == "UNKNOWN" for r in _lines),
                round(sum(len({r["speaker"] for page in p for r in page} - SPECIAL) for p in labels.values()) / len(labels), 2),
            ]
        },
        index=[
            "dev sequences",
            "test sequences",
            "train split",
            "val split",
            "dev lines",
            "empty dev pages",
            "NARRATION lines",
            "UNKNOWN lines",
            "characters per sequence",
        ],
    )
    stats
    return labels, pd, sequences, test_ids, train_ids, val_ids


@app.cell
def _(DATA, labels, sequences, train_ids):
    # Look at one training sequence next to its labels.
    _sid = train_ids[0]
    mo.vstack(
        [
            mo.md(f"**{_sid}**"),
            mo.hstack([mo.image(str(DATA / p), width=260) for p in sequences[_sid]["images"]]),
            mo.md(f"```\n{to_target(labels[_sid])}\n```"),
        ]
    )
    return


@app.function
def page_target(page):
    """One gold page -> the text the model is trained to write for it."""
    return "\n".join(f"{r['speaker']}: {' '.join(r['text'].split())}" for r in page) or "NONE"


@app.function
def to_target(pages):
    """Several pages -> transcript with PAGE headers (used as context and for the format check)."""
    return "\n".join(f"PAGE {_i}\n{page_target(_p)}" for _i, _p in enumerate(pages, 1))


@app.function
def parse_output(text):
    """Model text -> page lists. Tolerant of markdown junk and repeated lines."""
    pages, cur = [[], [], []], 0
    for _raw in text.splitlines():
        _s = _raw.strip().strip("*#`").strip()
        _m = re.match(r"^PAGE\s*([123])\b\W*$", _s, re.I)
        if _m:
            cur = int(_m.group(1)) - 1
            continue
        if ":" not in _s:
            continue
        _spk, _txt = _s.split(":", 1)
        _spk = _spk.strip(" -*").strip()
        _txt = " ".join(_txt.split())
        if not _spk or not _txt or len(_spk) > 30:
            continue
        if _spk.upper() in ("NARRATION", "NARRATOR"):
            _spk = "NARRATION"
        elif _spk.upper() == "UNKNOWN":
            _spk = "UNKNOWN"
        _item = {"speaker": _spk, "text": _txt}
        # Generation loops repeat a line forever; gold never repeats one line more than 4 times in a row.
        if len(pages[cur]) >= 4 and all(_x == _item for _x in pages[cur][-4:]):
            continue
        pages[cur].append(_item)
    return [collapse_loops(_p) for _p in pages]


@app.function
def parse_page(text):
    """Output for a single page -> one page list (ignores any stray PAGE headers)."""
    return collapse_loops([_r for _p in parse_output(text) for _r in _p])


@app.function
def collapse_loops(page):
    """Cut multi-line generation loops. Gold never repeats a block of 3+ lines, and a 2-line block at most twice."""
    for k in range(2, 9):
        copies = 3 if k == 2 else 2
        i = 0
        while i + copies * k <= len(page):
            blk = page[i : i + k]
            if all(page[i + c * k : i + (c + 1) * k] == blk for c in range(1, copies)):
                j = i + k
                while page[j : j + k] == blk:
                    j += k
                m = 0  # also drop the half-finished copy where generation got cut off
                while j + m < len(page) and page[j + m] == blk[m % k]:
                    m += 1
                page = page[: i + k] + page[j + m :]
            i += 1
    return page


@app.function
def norm_text(t):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", t.lower()).split())


@app.function
def text_similarity(a, b):
    a, b = norm_text(a), norm_text(b)
    if not a or not b:
        return 0.0
    _r = difflib.SequenceMatcher(None, a, b).ratio()
    if min(len(a), len(b)) >= 4 and (a in b or b in a):  # Magi often splits or merges balloons
        _r = max(_r, 0.9)
    return _r


@app.function
def cluster_characters(magi_pages, T):
    """Character ids across the 3 pages: merge Magi's per-page clusters, then average-linkage on crop embeddings."""
    keys, vecs = [], []
    for _p, _pg in enumerate(magi_pages):
        for _local in sorted(set(_pg["local"])):
            _rows = [i for i, l in enumerate(_pg["local"]) if l == _local]
            _v = _pg["emb"][_rows].mean(axis=0)
            keys.append((_p, _local))
            vecs.append(_v / (np.linalg.norm(_v) + 1e-8))
    if len(keys) <= 1:
        _ids = [1] * len(keys)
    else:
        _ids = fcluster(linkage(np.stack(vecs), method="average", metric="cosine"), t=T, criterion="distance")
    _gid = dict(zip(keys, (int(i) for i in _ids)))
    return [[_gid[(_p, l)] for l in _pg["local"]] for _p, _pg in enumerate(magi_pages)]


@app.function
def merge_with_magi(qwen_pages, magi_pages, T, M, drop=False, reorder=False, insert=False):
    """Qwen text + Magi: speakers from matched Magi boxes, optionally Magi reading order and Magi text for missed boxes."""
    if not magi_pages:
        return qwen_pages
    gids = cluster_characters(magi_pages, T)

    def _gid(p, j):
        _c = magi_pages[p]["assoc"].get(j)
        return gids[p][_c] if _c is not None and _c < len(gids[p]) else None

    votes, matched = {}, []
    for _p, _page in enumerate(qwen_pages):
        _rows = []
        for _line in _page:
            _sims = [text_similarity(_line["text"], o) for o in magi_pages[_p]["ocr"]]
            _j = max(range(len(_sims)), key=_sims.__getitem__) if _sims else None
            if _j is not None and _sims[_j] < M:
                _j = None
            _g = _gid(_p, _j) if _j is not None else None
            if _g is not None:
                votes.setdefault(_line["speaker"], Counter())[_g] += 1
            _rows.append((_line, _j, _g))
        matched.append(_rows)

    out = []
    for _p, _rows in enumerate(matched):
        _mp, _items, _last = magi_pages[_p], [], -1.0
        _ess = lambda j: _mp["essential"][j] if j < len(_mp["essential"]) else True
        for _n, (_line, _j, _g) in enumerate(_rows):
            if drop and _j is not None and not _ess(_j) and _line["speaker"] != "NARRATION":
                continue
            if _line["speaker"] == "NARRATION":
                _spk = "NARRATION"
            elif _g is not None:
                _spk = f"char{_g}"
            elif _line["speaker"] in votes:  # unmatched line: follow where the rest of this Qwen label went
                _spk = f"char{votes[_line['speaker']].most_common(1)[0][0]}"
            else:
                _spk = _line["speaker"]
            # Magi order: matched lines sort by their box; unmatched lines stay right after their predecessor
            _key = (float(_j) if _j is not None else _last + 1e-3) if reorder else float(_n)
            _last = _key
            _items.append((_key, _n, {"speaker": _spk, "text": _line["text"]}, _j))
        if insert:
            _used = {j for _, j, _ in _rows if j is not None}
            for _j, _o in enumerate(_mp["ocr"]):
                _txt = " ".join(_o.split())
                if _j in _used or not _ess(_j) or len(norm_text(_txt)) < 2:
                    continue
                if reorder:
                    _key = float(_j)
                else:  # place after the last Qwen line whose Magi box comes before this one
                    _before = [k for k, _, _, j in _items if j is not None and j < _j]
                    _key = (max(_before) if _before else -1.0) + 0.5
                _g = _gid(_p, _j)
                _items.append((_key, 10**6 + _j, {"speaker": f"char{_g}" if _g is not None else None, "text": _txt}, _j))
        _items.sort(key=lambda x: (x[0], x[1]))
        _page_out = []
        for _, _, _r, _ in _items:
            if _r["speaker"] is None:  # inserted line with no speaker link: assume the previous speaker continues
                _r = {"speaker": _page_out[-1]["speaker"] if _page_out else "UNKNOWN", "text": _r["text"]}
            _page_out.append(_r)
        out.append(_page_out)
    return out


@app.cell
def _(OUT, labels, scorer):
    def write_jsonl(path, rows):
        Path(path).write_text("".join(json.dumps({"sequence_id": k, "pages": v}, ensure_ascii=False) + "\n" for k, v in rows.items()))
        return path

    def score(pred, ids, name):
        _ref = write_jsonl(OUT / f"_ref_{name}.jsonl", {k: labels[k] for k in ids})
        _pred = write_jsonl(OUT / f"{name}_predictions.jsonl", {k: pred[k] for k in ids})
        _report = scorer.evaluate(scorer.load_jsonl(_ref, reference=True), scorer.load_jsonl(_pred))
        (OUT / f"{name}_scores.json").write_text(json.dumps(_report, indent=2))
        return _report

    # Sanity check: gold -> model text format -> parser must score perfectly, both whole-sequence and page-by-page.
    _seq = score({k: parse_output(to_target(v)) for k, v in labels.items()}, sorted(labels), "roundtrip_seq")
    _pg = score({k: [parse_page(page_target(p)) for p in v] for k, v in labels.items()}, sorted(labels), "roundtrip_page")
    for _r in (_seq, _pg):
        assert _r["macro"]["text_order_score"] > 0.999 and _r["macro"]["balanced_joint_f1"] > 0.999, _r["macro"]
    mo.md(f"Format round-trip check passed: `{_pg['macro']}`")
    return score, write_jsonl


@app.cell
def _(MAX_PIXELS, MODEL_ID, SEED, deps_ready):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    assert deps_ready
    assert torch.cuda.is_available(), "No GPU. In molab open notebook specs and switch the GPU on, then rerun."
    torch.manual_seed(SEED)
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=256 * 28 * 28, max_pixels=MAX_PIXELS)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    base_model.eval()
    mo.md(f"Loaded `{MODEL_ID}` on `{torch.cuda.get_device_name(0)}`")
    return base_model, processor, torch


@app.cell
def _(DATA, MAX_NEW_TOKENS, processor, sequences, torch):
    from PIL import Image

    def load_pages(sid):
        return [Image.open(DATA / p).convert("RGB") for p in sequences[sid]["images"]]

    def encode(images, k, earlier_pages, answer=None):
        """Prompt for page k (0-based): images of pages 0..k + transcript of pages 0..k-1."""
        _content = []
        for _j in range(k + 1):
            _content += [{"type": "text", "text": f"PAGE {_j + 1}:"}, {"type": "image"}]
        _context = to_target(earlier_pages) if earlier_pages else "(none, this is the first page)"
        _content.append({"type": "text", "text": PROMPT.format(k=k + 1, context=_context)})
        _prompt = processor.apply_chat_template([{"role": "user", "content": _content}], tokenize=False, add_generation_prompt=True)
        _imgs = images[: k + 1]
        if answer is None:
            return processor(text=[_prompt], images=_imgs, return_tensors="pt")
        _enc = processor(text=[_prompt + answer + "<|im_end|>\n"], images=_imgs, return_tensors="pt")
        _n_prompt = processor(text=[_prompt], images=_imgs, return_tensors="pt")["input_ids"].shape[1]
        _labels = _enc["input_ids"].clone()
        _labels[:, :_n_prompt] = -100  # loss only on the answer
        _enc["labels"] = _labels
        return _enc

    @torch.no_grad()
    def predict(model, ids, tag):
        """Page by page; each page sees the transcript the model itself wrote for the earlier pages."""
        model.eval()
        _preds, _raw = {}, {}
        _t = time.time()
        for _n, _sid in enumerate(ids, 1):
            _imgs, _pages, _raw[_sid] = load_pages(_sid), [], []
            for _k in range(3):
                _enc = encode(_imgs, _k, _pages).to(model.device)
                _out = model.generate(**_enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
                _text = processor.batch_decode(_out[:, _enc["input_ids"].shape[1]:], skip_special_tokens=True)[0]
                _raw[_sid].append(_text)
                _pages.append(parse_page(_text))
            _preds[_sid] = _pages
            print(f"[{tag}] {_n}/{len(ids)} {_sid} {time.time() - _t:.0f}s", flush=True)
        return _preds, _raw

    mo.md("Encoding / generation helpers ready.")
    return encode, load_pages, predict


@app.cell
def _(EPOCHS, GRAD_ACCUM, LORA_R, LR, SEED, encode, labels, load_pages, pd, torch):
    from peft import LoraConfig, get_peft_model

    def train_lora(base, train_seq_ids, val_seq_ids, tag):
        """Fresh LoRA on the language model (vision tower frozen). One example = one page with its earlier-page context."""
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
        base.config.use_cache = False
        model = get_peft_model(
            base,
            LoraConfig(
                r=LORA_R,
                lora_alpha=2 * LORA_R,
                lora_dropout=0.05,
                target_modules=r".*\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)",
                task_type="CAUSAL_LM",
            ),
        )
        _examples = [(s, k) for s in train_seq_ids for k in range(3)]
        _val_examples = [(s, k) for s in val_seq_ids for k in range(3)]
        _params = [p for p in model.parameters() if p.requires_grad]
        print(f"[{tag}] trainable params {sum(p.numel() for p in _params) / 1e6:.1f}M, {len(_examples)} page examples", flush=True)
        _opt = torch.optim.AdamW(_params, lr=LR, weight_decay=0.0)
        _total = EPOCHS * ((len(_examples) + GRAD_ACCUM - 1) // GRAD_ACCUM)
        _warm = max(1, _total // 10)
        _sched = torch.optim.lr_scheduler.LambdaLR(_opt, lambda s: min((s + 1) / _warm, max(0.0, (_total - s) / (_total - _warm))))

        def _batch(sid, k):
            return encode(load_pages(sid), k, labels[sid][:k], page_target(labels[sid][k])).to(model.device)

        def _val_loss():
            if not _val_examples:
                return None
            model.eval()
            with torch.no_grad():
                _l = [model(**_batch(s, k)).loss.item() for s, k in _val_examples]
            model.train()
            return sum(_l) / len(_l)

        _log = [{"epoch": 0, "step": 0, "train_loss": None, "val_loss": _val_loss()}]
        _rng = random.Random(SEED)
        _step, _t = 0, time.time()
        model.train()
        for _epoch in range(1, EPOCHS + 1):
            _order = list(_examples)
            _rng.shuffle(_order)
            _running = []
            for _i, (_sid, _k) in enumerate(_order, 1):
                _loss = model(**_batch(_sid, _k)).loss
                (_loss / GRAD_ACCUM).backward()
                _running.append(_loss.item())
                if _i % GRAD_ACCUM == 0 or _i == len(_order):
                    torch.nn.utils.clip_grad_norm_(_params, 1.0)
                    _opt.step()
                    _sched.step()
                    _opt.zero_grad(set_to_none=True)
                    _step += 1
                    _log.append({"epoch": _epoch, "step": _step, "train_loss": sum(_running) / len(_running), "val_loss": None})
                    print(f"[{tag}] epoch {_epoch} step {_step}/{_total} loss {_log[-1]['train_loss']:.4f} {time.time() - _t:.0f}s", flush=True)
                    _running = []
            _log.append({"epoch": _epoch, "step": _step, "train_loss": None, "val_loss": _val_loss()})
            print(f"[{tag}] epoch {_epoch} val_loss {_log[-1]['val_loss']}", flush=True)

        model.eval()
        base.gradient_checkpointing_disable()
        base.config.use_cache = True
        return model, pd.DataFrame(_log)

    mo.md("LoRA training function ready.")
    return (train_lora,)


@app.cell
def _(OUT, base_model, train_ids, train_lora, val_ids):
    # Qwen LoRA on the 64 train sequences (zero-shot baseline is in v2, not repeated here).
    model, _log = train_lora(base_model, train_ids, val_ids, "train64")
    model.save_pretrained(OUT / "lora_adapter_train64")
    _log.to_csv(OUT / "train_log_train64.csv", index=False)
    _log
    return (model,)


@app.cell
def _(OUT, model, pd, predict, score, val_ids):
    qwen_val_preds, _raw = predict(model, val_ids, "lora")
    (OUT / "lora_val_raw.json").write_text(json.dumps(_raw, indent=2, ensure_ascii=False))
    qwen_val_report = score(qwen_val_preds, val_ids, "val_qwen_only")
    pd.DataFrame({"Qwen LoRA (val)": qwen_val_report["macro"]})
    return qwen_val_preds, qwen_val_report


@app.cell
def _(deps_ready, torch):
    # Magi v2 (Sachdeva et al. 2024): manga-specific detector for characters, text boxes and speaker links.
    from transformers import AutoModel

    assert deps_ready
    try:
        magi = AutoModel.from_pretrained("ragavsachdeva/magiv2", trust_remote_code=True).to("cuda").eval()
        magi_error = None
    except Exception as _e:  # keep the run alive: without Magi we fall back to Qwen-only speakers
        magi, magi_error = None, repr(_e)
    mo.md("Magi v2 loaded." if magi else f"**Magi failed to load, falling back to Qwen-only speakers:** `{magi_error}`")
    return (magi,)


@app.cell
def _(DATA, OUT, labels, magi, sequences, test_ids, torch):
    from PIL import Image as _Image

    def _run_magi(sid):
        _imgs = [np.array(_Image.open(DATA / p).convert("L").convert("RGB")) for p in sequences[sid]["images"]]
        with torch.no_grad():
            _dets = magi.predict_detections_and_associations(_imgs)
            _pages = []
            for _img, _d in zip(_imgs, _dets):
                _texts, _chars = list(_d["texts"]), list(_d["characters"])
                _ocr = magi.predict_ocr([_img], [_texts])[0] if _texts else []
                _emb = np.zeros((0, 1), dtype=np.float32)
                if _chars:
                    _emb = magi.predict_crop_embeddings([_img], [_chars])[0].float().cpu().numpy()
                    _emb = _emb / (np.linalg.norm(_emb, axis=1, keepdims=True) + 1e-8)
                _pages.append(
                    {
                        "ocr": [str(x) for x in _ocr],
                        "essential": [bool(x) for x in _d["is_essential_text"]],
                        "assoc": {int(t): int(c) for t, c in _d["text_character_associations"]},
                        "local": [int(x) for x in _d["character_cluster_labels"]],
                        "emb": _emb,
                    }
                )
        return _pages

    magi_cache = {}
    if magi is not None:
        _t = time.time()
        for _n, _sid in enumerate(sorted(labels) + list(test_ids), 1):
            try:
                magi_cache[_sid] = _run_magi(_sid)
            except Exception as _e:  # one bad page should not kill the run; that sequence keeps Qwen speakers
                print(f"[magi] {_sid} failed: {_e!r}", flush=True)
            if _n % 10 == 0:
                print(f"[magi] {_n} sequences {time.time() - _t:.0f}s", flush=True)
    (OUT / "magi_outputs.json").write_text(json.dumps({k: [{**pg, "emb": pg["emb"].tolist()} for pg in v] for k, v in magi_cache.items()}))
    mo.md(f"Magi ran on {len(magi_cache)} sequences.")
    return (magi_cache,)


@app.cell
def _(labels, magi_cache, pd, score, train_ids):
    # Tune the two thresholds on the 64 TRAIN sequences only, using gold text with every speaker collapsed
    # to one label, so the score measures what Magi alone adds to speaker identity.
    _flat = {
        sid: [[{"speaker": r["speaker"] if r["speaker"] == "NARRATION" else "character1", "text": r["text"]} for r in page] for page in labels[sid]]
        for sid in train_ids
    }
    _rows = [{"T": None, "M": None, "balanced_joint_f1": score(_flat, train_ids, "tune")["macro"]["balanced_joint_f1"]}]
    if magi_cache:
        for _T in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6):  # first run peaked at the low edge (0.3)
            for _M in (0.4, 0.5, 0.6):
                _pred = {sid: merge_with_magi(_flat[sid], magi_cache.get(sid), _T, _M) for sid in train_ids}
                _rows.append({"T": _T, "M": _M, "balanced_joint_f1": score(_pred, train_ids, "tune")["macro"]["balanced_joint_f1"]})
    tuning = pd.DataFrame(_rows)
    _best = tuning.dropna().sort_values("balanced_joint_f1", ascending=False).head(1)
    best_T, best_M = (float(_best["T"].iloc[0]), float(_best["M"].iloc[0])) if len(_best) else (0.5, 0.5)
    # Diagnostic: how good is Magi's reading order? Reorder the GOLD lines by Magi and score (gold order = 1.0).
    _magi_order = None
    if magi_cache:
        _ro = {sid: merge_with_magi(_flat[sid], magi_cache.get(sid), best_T, best_M, reorder=True) for sid in train_ids}
        _magi_order = score(_ro, train_ids, "tune_order")["macro"]["text_order_score"]
    mo.vstack([mo.md(f"Threshold search on train (first row = no Magi). Best T={best_T}, M={best_M}. "
                     f"Gold text in Magi reading order scores text_order={_magi_order}"), tuning])
    return best_M, best_T, tuning


@app.cell
def _(OUT, best_M, best_T, magi_cache, pd, qwen_val_preds, qwen_val_report, score, tuning, val_ids):
    tuning.to_csv(OUT / "magi_threshold_tuning_train.csv", index=False)
    _variants = {"Qwen only": (qwen_val_preds, qwen_val_report)}
    if magi_cache:
        for _name, _opts in VARIANTS.items():
            if _opts is None:
                continue
            _p = {sid: merge_with_magi(qwen_val_preds[sid], magi_cache.get(sid), best_T, best_M, **_opts) for sid in val_ids}
            _variants[_name] = (_p, score(_p, val_ids, "val_" + re.sub(r"\W+", "_", _name)))
    val_table = pd.DataFrame({k: v[1]["macro"] for k, v in _variants.items()})
    val_table.to_csv(OUT / "val_comparison.csv")
    # 16 val sequences are noisy and we pick among several options, so a more complex option must beat the
    # current choice by > 0.01 (text_order + balanced_joint_f1) to be used. Variants are listed simplest first.
    _total = lambda k: _variants[k][1]["macro"]["text_order_score"] + _variants[k][1]["macro"]["balanced_joint_f1"]
    chosen = "Qwen only"
    for _k in _variants:
        if _total(_k) > _total(chosen) + 0.01:
            chosen = _k
    mo.vstack([mo.md(f"### Validation (16 held-out dev sequences)\nChosen for test: **{chosen}**"), val_table])
    return (chosen,)


@app.cell
def _(OUT, chosen, labels, model, train_lora):
    # Final Qwen model: same recipe, retrained on all 80 dev sequences.
    assert chosen  # val evaluation must finish before the val-split adapter is removed
    _base = model.unload()
    final_model, _log = train_lora(_base, sorted(labels), [], "all80")
    final_model.save_pretrained(OUT / "lora_adapter_all80")
    _log.to_csv(OUT / "train_log_all80.csv", index=False)
    _log
    return (final_model,)


@app.cell
def _(OUT, best_M, best_T, chosen, final_model, magi_cache, predict, scorer, test_ids, write_jsonl):
    # Final test predictions, generated straight from the images.
    _qwen, _raw = predict(final_model, test_ids, "test")
    (OUT / "test_raw.json").write_text(json.dumps(_raw, indent=2, ensure_ascii=False))
    write_jsonl(OUT / "predictions_qwen_only.jsonl", {k: _qwen[k] for k in test_ids})
    _opts = VARIANTS[chosen]
    test_preds = _qwen if _opts is None else {k: merge_with_magi(_qwen[k], magi_cache.get(k), best_T, best_M, **_opts) for k in test_ids}
    _path = write_jsonl(OUT / "predictions.jsonl", {k: test_preds[k] for k in test_ids})
    _check = scorer.load_jsonl(_path)  # official format validation
    assert len(_check) == 15 and list(_check) == test_ids
    mo.md(f"Wrote `{_path}` ({chosen}) with {len(_check)} sequences, {sum(len(p) for v in test_preds.values() for p in v)} lines.")
    return (test_preds,)


@app.cell
def _(OUT, WORK, test_preds):
    assert test_preds
    _zip = shutil.make_archive(str(WORK / "manga_task_v4_results"), "zip", OUT)
    mo.vstack(
        [
            mo.download(data=(OUT / "predictions.jsonl").read_bytes(), filename="predictions.jsonl", label="predictions.jsonl"),
            mo.download(data=Path(_zip).read_bytes(), filename="manga_task_v4_results.zip", label="all results (adapters, logs, scores)"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
