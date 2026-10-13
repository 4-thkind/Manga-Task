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
    import importlib.util
    from importlib import metadata
    from pathlib import Path

    import marimo as mo

    SPECIAL = {"NARRATION", "UNKNOWN"}
    PROMPT = (
        "These are three consecutive pages of an English manga. Transcribe the story text: "
        "dialogue, thoughts, narration, clearly spoken screams or vocalisations, and "
        "punctuation-only balloons such as ... or ?!. Skip sound effects, titles, credits, "
        "page numbers, signs, writing on objects, and text inside letters, phones or documents. "
        "Read in manga order (right-to-left, top-to-bottom, following the panel layout). "
        "Write one balloon per line as `speaker: text`. Use character1, character2, ... and keep "
        "the same label for the same character on all three pages. Use NARRATION for narrator "
        "boxes and UNKNOWN when the speaker cannot be told. Join wrapped lines, keep stutters "
        "and punctuation. Start each page with a header line PAGE 1, PAGE 2, PAGE 3; leave a "
        "page empty under its header if it has no story text."
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # BYTE Manga Task [Advanced] — Qwen2.5-VL-7B + LoRA

    Pipeline: all **3 pages of a sequence go into one prompt** to an open-weight VLM
    (Qwen2.5-VL-7B-Instruct), which writes every line as `speaker: text` page by page.
    Seeing the whole triplet at once is what keeps speaker labels consistent across pages.

    Experiment: zero-shot model vs. the same model after **LoRA fine-tuning** on 64 dev
    sequences, both scored with the official `score.py` on 16 held-out dev sequences
    (split by whole sequence). The fine-tuned adapter then predicts the 15 test sequences.

    **Before running:** turn the GPU on (notebook specs → GPUs), paste your Drive link in the
    next cell, then *Run all*. The last cell gives the download.
    """)
    return


@app.cell
def _():
    # Upload the dataset zip to Google Drive, set sharing to "Anyone with the link", paste the link here.
    DATA_URL = "PASTE_GOOGLE_DRIVE_FILE_LINK_HERE"

    MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
    WORK = Path("work").resolve()
    OUT = WORK / "outputs"
    OUT.mkdir(parents=True, exist_ok=True)
    SEED = 0
    N_VAL = 16
    EPOCHS = 3
    LR = 1e-4
    GRAD_ACCUM = 4
    LORA_R = 16
    MAX_PIXELS = 1280 * 28 * 28  # ~1M px per page: near full resolution, keeps small text legible
    MAX_NEW_TOKENS = 1200
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
    for _name in ("torch", "torchvision", "pandas", "pillow"):
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
            assert "drive.google.com" in DATA_URL, "Paste your Google Drive link into DATA_URL (cell above)."
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

    # Split by whole sequence, never by page.
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
def to_target(pages):
    """Gold pages -> the text the model is trained to write."""
    _blocks = []
    for _i, _page in enumerate(pages, 1):
        _rows = [f"{r['speaker']}: {' '.join(r['text'].split())}" for r in _page]
        _blocks.append("\n".join([f"PAGE {_i}", *_rows]))
    return "\n".join(_blocks)


@app.function
def parse_output(text):
    """Model text -> 3 page lists. Tolerant of markdown junk and repeated lines."""
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

    # Sanity check: gold -> model text format -> parser must score perfectly.
    _roundtrip = {k: parse_output(to_target(v)) for k, v in labels.items()}
    _r = score(_roundtrip, sorted(labels), "roundtrip_check")
    assert _r["macro"]["text_order_score"] > 0.999 and _r["macro"]["balanced_joint_f1"] > 0.999, _r["macro"]
    mo.md(f"Format round-trip check passed: `{_r['macro']}`")
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

    def encode(images, answer=None):
        _msgs = [{"role": "user", "content": [{"type": "image"}] * 3 + [{"type": "text", "text": PROMPT}]}]
        _prompt = processor.apply_chat_template(_msgs, tokenize=False, add_generation_prompt=True)
        if answer is None:
            return processor(text=[_prompt], images=images, return_tensors="pt")
        _enc = processor(text=[_prompt + answer + "<|im_end|>\n"], images=images, return_tensors="pt")
        _n_prompt = processor(text=[_prompt], images=images, return_tensors="pt")["input_ids"].shape[1]
        _labels = _enc["input_ids"].clone()
        _labels[:, :_n_prompt] = -100  # loss only on the answer
        _enc["labels"] = _labels
        return _enc

    @torch.no_grad()
    def predict(model, ids, tag):
        model.eval()
        _preds, _raw = {}, {}
        _t = time.time()
        for _n, _sid in enumerate(ids, 1):
            _enc = encode(load_pages(_sid)).to(model.device)
            _out = model.generate(**_enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            _raw[_sid] = processor.batch_decode(_out[:, _enc["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            _preds[_sid] = parse_output(_raw[_sid])
            print(f"[{tag}] {_n}/{len(ids)} {_sid} {time.time() - _t:.0f}s", flush=True)
        return _preds, _raw

    mo.md("Encoding / generation helpers ready.")
    return encode, load_pages, predict


@app.cell
def _(OUT, base_model, pd, predict, score, val_ids):
    # Experiment 1: zero-shot baseline (no adaptation), same prompt, same parser.
    baseline_preds, _raw = predict(base_model, val_ids, "zero-shot")
    (OUT / "zero_shot_raw.json").write_text(json.dumps(_raw, indent=2, ensure_ascii=False))
    baseline_report = score(baseline_preds, val_ids, "val_zero_shot")
    pd.DataFrame({"zero-shot (val)": baseline_report["macro"]})
    return (baseline_report,)


@app.cell
def _(
    EPOCHS,
    GRAD_ACCUM,
    LORA_R,
    LR,
    OUT,
    SEED,
    base_model,
    baseline_report,
    encode,
    labels,
    load_pages,
    pd,
    torch,
    train_ids,
    val_ids,
):
    from peft import LoraConfig, get_peft_model

    assert baseline_report  # get_peft_model edits base_model in place, so the baseline must run first

    base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base_model.enable_input_require_grads()
    base_model.config.use_cache = False
    # LoRA only on the language model; the vision tower stays frozen.
    model = get_peft_model(
        base_model,
        LoraConfig(
            r=LORA_R,
            lora_alpha=2 * LORA_R,
            lora_dropout=0.05,
            target_modules=r".*\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)",
            task_type="CAUSAL_LM",
        ),
    )
    _trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {_trainable / 1e6:.1f}M")

    _params = [p for p in model.parameters() if p.requires_grad]
    _opt = torch.optim.AdamW(_params, lr=LR, weight_decay=0.0)
    _total = EPOCHS * len(train_ids) // GRAD_ACCUM
    _warm = max(1, _total // 10)
    _sched = torch.optim.lr_scheduler.LambdaLR(_opt, lambda s: min((s + 1) / _warm, max(0.0, (_total - s) / (_total - _warm))))

    def _val_loss():
        model.eval()
        _losses = []
        with torch.no_grad():
            for _sid in val_ids:
                _enc = encode(load_pages(_sid), to_target(labels[_sid])).to(model.device)
                _losses.append(model(**_enc).loss.item())
        model.train()
        return sum(_losses) / len(_losses)

    _log = [{"epoch": 0, "step": 0, "train_loss": None, "val_loss": _val_loss()}]
    _rng = random.Random(SEED)
    _step, _t = 0, time.time()
    model.train()
    for _epoch in range(1, EPOCHS + 1):
        _order = list(train_ids)
        _rng.shuffle(_order)
        _running = []
        for _i, _sid in enumerate(_order, 1):
            _enc = encode(load_pages(_sid), to_target(labels[_sid])).to(model.device)
            _loss = model(**_enc).loss
            (_loss / GRAD_ACCUM).backward()
            _running.append(_loss.item())
            if _i % GRAD_ACCUM == 0 or _i == len(_order):
                torch.nn.utils.clip_grad_norm_(_params, 1.0)
                _opt.step()
                _sched.step()
                _opt.zero_grad(set_to_none=True)
                _step += 1
                _log.append({"epoch": _epoch, "step": _step, "train_loss": sum(_running) / len(_running), "val_loss": None})
                print(f"epoch {_epoch} step {_step}/{_total} loss {_log[-1]['train_loss']:.4f} {time.time() - _t:.0f}s", flush=True)
                _running = []
        _log.append({"epoch": _epoch, "step": _step, "train_loss": None, "val_loss": _val_loss()})
        print(f"epoch {_epoch} val_loss {_log[-1]['val_loss']:.4f}", flush=True)

    model.eval()
    base_model.gradient_checkpointing_disable()
    base_model.config.use_cache = True
    model.save_pretrained(OUT / "lora_adapter")
    train_log = pd.DataFrame(_log)
    train_log.to_csv(OUT / "train_log.csv", index=False)
    train_log
    return (model,)


@app.cell
def _(OUT, baseline_report, model, pd, predict, score, val_ids):
    # Experiment 2: same model + LoRA adapter, same val sequences.
    finetuned_preds, _raw = predict(model, val_ids, "lora")
    (OUT / "lora_val_raw.json").write_text(json.dumps(_raw, indent=2, ensure_ascii=False))
    finetuned_report = score(finetuned_preds, val_ids, "val_lora")

    comparison = pd.DataFrame({"zero-shot": baseline_report["macro"], "LoRA": finetuned_report["macro"]})
    comparison.to_csv(OUT / "val_comparison.csv")
    per_sequence = pd.DataFrame(
        {
            sid: {
                "zs_text_order": baseline_report["sequences"][sid]["text_order_score"],
                "lora_text_order": finetuned_report["sequences"][sid]["text_order_score"],
                "zs_joint_f1": baseline_report["sequences"][sid]["balanced_joint_f1"],
                "lora_joint_f1": finetuned_report["sequences"][sid]["balanced_joint_f1"],
            }
            for sid in val_ids
        }
    ).T.round(3)
    per_sequence.to_csv(OUT / "val_per_sequence.csv")
    mo.vstack([mo.md("### Validation (16 held-out dev sequences)"), comparison, mo.md("Per sequence"), per_sequence])
    return


@app.cell
def _(OUT, model, predict, scorer, test_ids, write_jsonl):
    # Final test predictions, generated straight from the images.
    test_preds, _raw = predict(model, test_ids, "test")
    (OUT / "test_raw.json").write_text(json.dumps(_raw, indent=2, ensure_ascii=False))
    _path = write_jsonl(OUT / "predictions.jsonl", {k: test_preds[k] for k in test_ids})
    _check = scorer.load_jsonl(_path)  # official format validation
    assert len(_check) == 15 and list(_check) == test_ids
    mo.md(f"Wrote `{_path}` with {len(_check)} sequences, {sum(len(p) for v in test_preds.values() for p in v)} lines.\n\n"
          f"```\n{_path.read_text().splitlines()[0][:1500]}\n```")
    return (test_preds,)


@app.cell
def _(OUT, WORK, test_preds):
    assert test_preds
    _zip = shutil.make_archive(str(WORK / "manga_task_results"), "zip", OUT)
    mo.vstack(
        [
            mo.download(data=(OUT / "predictions.jsonl").read_bytes(), filename="predictions.jsonl", label="predictions.jsonl"),
            mo.download(data=Path(_zip).read_bytes(), filename="manga_task_results.zip", label="all results (adapter, logs, scores)"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
