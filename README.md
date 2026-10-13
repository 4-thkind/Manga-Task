# BYTE Manga Task (Advanced)

Reading three consecutive English manga pages, pulling out the story text in reading order, and labelling who said each line, with the same label for the same character across all three pages.

**Final validation score (16 held out dev sequences):** text order **0.89**, balanced joint F1 **0.31**

## Approach

I used one open weight vision language model, Qwen2.5 VL 7B Instruct (`Qwen/Qwen2.5-VL-7B-Instruct`), and adapted it with LoRA on the development data.

The final pipeline (v2) works page by page:

1. For page k, the model gets the images of pages 1 to k plus the transcript it already wrote for the earlier pages.
2. It writes only page k, one balloon per line as `speaker: text`, or `NONE` if the page has no story text.
3. A small parser turns that into the required JSON and cuts generation loops (a model sometimes repeats the same lines forever).
4. The transcript of the earlier pages carries the speaker labels forward, so `character1` on page 1 stays `character1` on page 3.

Training details:

* LoRA rank 16, alpha 32, on the attention and MLP layers of the language model only. The vision encoder stays frozen.
* One training example is one page plus its earlier pages as context, with gold labels as the context. Loss is only on the answer tokens.
* 4 epochs, learning rate 1e-4, gradient accumulation 8, bf16, images at native resolution.
* Split: 64 train and 16 validation sequences, split by whole sequence (seed 0), never by page.
* After validation, the same recipe is retrained on all 80 dev sequences, and that model writes the test predictions.

## Experiments

All numbers are from the official `score.py` on the same 16 validation sequences.

| Run | text_order_score | balanced_joint_f1 | matched_token_coverage |
|---|---|---|---|
| v1 zero shot (3 pages in one prompt) | 0.31 | 0.15 | 0.43 |
| v1 LoRA | 0.71 | 0.26 | 0.82 |
| v2 zero shot (page by page) | 0.45 | 0.16 | 0.59 |
| **v2 LoRA (final)** | **0.89** | **0.31** | **0.93** |

**v1** gave the model all three pages at once and asked for the full transcript in one go. My idea was that seeing everything together would help speaker consistency. Fine tuning took text order from 0.31 to 0.71.

**Why I moved to v2.** I looked at the v1 scores page by page. Most pages were already near perfect. The lost score came from page bookkeeping, not from reading:

* in 2 of the 16 sequences, page 3 came out completely empty (score 0)
* text from one page got written under another page
* one page with no story text got filled with text from a neighbouring page

Making the model write one page per call removes all three problems. Page by page helped even without training (zero shot went from 0.31 to 0.45), and with LoRA it went from 0.71 to 0.89.

The training logs are in `results/V2-FINAL_RESULTS/train_log_train64.csv`. Validation loss went 0.37, 0.20, 0.15, 0.18, 0.16 over the 4 epochs, so it was lowest after epoch 2 and the model started to overfit a little after that.

## What worked and what did not

Worked:

* LoRA fine tuning on only 64 sequences. It taught the model the output format, what to skip (sound effects, signs) and the reading order much better than prompting alone.
* Page by page generation. Biggest single jump in the project.
* A simple loop cleaner in the parser. In v1 one test sequence repeated the same two lines about 40 times.

Did not work well:

* **Speakers.** Balanced joint F1 is only 0.31. In v1 the model usually used 2 labels when there were really 3 to 7 characters. In v2 it sometimes goes the other way and gives almost every line a new label (one test sequence ended up with 20 labels).
* **Hard pages.** `seq_2032620aa4e4ac7f` still scores 0.50 and `seq_ac5793f8168efdd1` scores 0.75. Small floating text and dense pages are the usual problems.
* **Sound effects.** A few translated sound effects still slip in. I checked if lowercase text could be used as a filter, but the gold labels have 89 real lines in lowercase, so that rule would delete real dialogue.

## What I would try next

* Train for 2 or 3 epochs instead of 4, since validation loss was best at epoch 2.
* A bigger model, Qwen2.5 VL 32B, which should fit on the same 96 GB GPU.
* A separate speaker step: detect characters, embed the crops, cluster them across the three pages, and link each balloon to the nearest character by its tail.
* A small classifier that drops sound effects before they reach the transcript.

## How to run

Everything is one marimo notebook, run on molab with a GPU (RTX Pro 6000, 96 GB).

1. Open `notebooks/manga_task_v2-FINAL-NB.py` on molab and turn the GPU on.
2. Click Run all. The dataset downloads from the Google Drive link already in the second cell.
3. It installs packages, checks the output format, runs the zero shot baseline, trains on 64 sequences, scores validation, retrains on all 80, writes `predictions.jsonl`, and gives a download in the last cell.

A full run takes about an hour. Seeds are fixed and decoding is greedy. `notebooks/manga_task_v1-TRIAL-101.py` is the earlier version, kept for the comparison.

## Files

| Path | What it is |
|---|---|
| `predictions.jsonl` | Final test predictions (15 sequences) |
| `notebooks/manga_task_v2-FINAL-NB.py` | Final pipeline: data, baseline, training, evaluation, test predictions |
| `notebooks/manga_task_v1-TRIAL-101.py` | First version (3 pages in one prompt) |
| `results/V2-FINAL_RESULTS/` | Scores, per sequence results, training logs and raw model outputs for v2 |
| `results/V1-first-run/` | Same for v1 |
| `adapters/` | LoRA configs. Weights are on Drive (link below) |
| `requirements.txt` | Package versions |

**Trained LoRA weights:** https://drive.google.com/drive/folders/1I8l-8jSxBofbJ7Z1tDxfuX_i0D9SolJ7?usp=sharing

`lora_adapter_all80` made the test predictions. `lora_adapter_train64` is the one scored on validation. Each is about 155 MB, too big for GitHub.

## Credits

* Qwen2.5 VL by the Qwen team, Alibaba: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
* Hugging Face Transformers: https://github.com/huggingface/transformers
* PEFT for LoRA: https://github.com/huggingface/peft
* marimo and molab: https://marimo.io
* gdown for the Drive download: https://github.com/wkentaro/gdown
* The official BYTE scoring script, used for every number above.

No hosted AI API was used to make training targets or predictions. Everything was produced by the open weight model above. I used an AI assistant (Claude) to help write code and to draft this README, which the task brief allows if disclosed.
