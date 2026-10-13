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
    # BYTE Manga Task [Advanced] — v2: page-by-page Qwen2.5-VL-7B + LoRA

    **What changed from v1 and why.** v1 put all 3 pages in one prompt and asked for the whole
    transcript at once (val text_order 0.71). Error analysis showed most pages were already
    near-perfect; the lost score came from *page bookkeeping*: whole pages dropped (score 0),
    text written under the wrong page, an empty page filled with another page's text.

    v2 transcribes **one page per call**. The model sees pages 1..k as images plus the transcript
    it already produced for pages 1..k-1, and writes page k only. Page assignment can no longer
    go wrong, and the earlier transcript carries the speaker labels forward so they stay
    consistent across the three pages. Also: native resolution, 4 epochs, and the final test
    model is retrained on all 80 dev sequences.

    Experiments (all scored with the official `score.py` on 16 held-out dev sequences):
    zero-shot vs. LoRA fine-tuned, same page-by-page pipeline.

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
    EPOCHS = 4  # v1 val loss was still falling after 3
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
def _(OUT, base_model, pd, predict, score, val_ids):
    # Experiment 1: zero-shot baseline (no adaptation), same page-by-page pipeline and parser.
    baseline_preds, _raw = predict(base_model, val_ids, "zero-shot")
    (OUT / "zero_shot_raw.json").write_text(json.dumps(_raw, indent=2, ensure_ascii=False))
    baseline_report = score(baseline_preds, val_ids, "val_zero_shot")
    pd.DataFrame({"zero-shot (val)": baseline_report["macro"]})
    return (baseline_report,)


@app.cell
def _(OUT, base_model, baseline_report, train_ids, train_lora, val_ids):
    # Experiment 2: LoRA on the 64 train sequences. (Runs after the baseline: get_peft_model edits base_model in place.)
    assert baseline_report
    model, _log = train_lora(base_model, train_ids, val_ids, "train64")
    model.save_pretrained(OUT / "lora_adapter_train64")
    _log.to_csv(OUT / "train_log_train64.csv", index=False)
    _log
    return (model,)


@app.cell
def _(OUT, baseline_report, model, pd, predict, score, val_ids):
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
    return (finetuned_report,)


@app.cell
def _(OUT, finetuned_report, labels, model, train_lora):
    # Final model: same recipe, retrained on all 80 dev sequences (more data for the test run).
    assert finetuned_report  # val evaluation must finish before the val-split adapter is removed
    _base = model.unload()  # strip the val-split LoRA, back to the plain base model
    final_model, _log = train_lora(_base, sorted(labels), [], "all80")
    final_model.save_pretrained(OUT / "lora_adapter_all80")
    _log.to_csv(OUT / "train_log_all80.csv", index=False)
    _log
    return (final_model,)


@app.cell
def _(OUT, final_model, predict, scorer, test_ids, write_jsonl):
    # Final test predictions, generated straight from the images.
    test_preds, _raw = predict(final_model, test_ids, "test")
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
    _zip = shutil.make_archive(str(WORK / "manga_task_v2_results"), "zip", OUT)
    mo.vstack(
        [
            mo.download(data=(OUT / "predictions.jsonl").read_bytes(), filename="predictions.jsonl", label="predictions.jsonl"),
            mo.download(data=Path(_zip).read_bytes(), filename="manga_task_v2_results.zip", label="all results (adapters, logs, scores)"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
