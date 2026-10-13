# BYTE Manga Task (Advanced)

Reading three consecutive English manga pages, pulling out the story text in reading order, and labelling who said each line, with the same label for the same character across all three pages.

**Final validation score (16 held out dev sequences):** text order **0.92**, balanced joint F1 **0.43**

## Final pipeline (v4)

Two open weight models, each doing the part it is good at.

**Qwen2.5 VL 7B Instruct** (`Qwen/Qwen2.5-VL-7B-Instruct`), fine tuned with LoRA, reads the text:

1. For page k, the model gets the images of pages 1 to k plus the transcript it already wrote for the earlier pages.
2. It writes only page k, one balloon per line as `speaker: text`, or `NONE` if the page has no story text.
3. A small parser turns that into JSON and cuts generation loops.

**Magi v2** (`ragavsachdeva/magiv2`), an open manga model, is used without training. It finds text boxes in reading order, character boxes, which character each text box belongs to, and an embedding for every character crop. I wrote the glue that combines it with Qwen:

1. Every Qwen line is matched to a Magi text box by text similarity (Magi runs its own OCR on each box).
2. **Speaker:** the character Magi links to that box. To keep labels consistent across the three pages, I average the crop embeddings of each character group on a page and cluster those groups across all three pages (average linkage, cosine distance, threshold T). Lines with no matching box follow the cluster that the rest of their Qwen label went to. NARRATION stays from Qwen.
3. **Order:** lines are sorted by the reading order of their Magi box. A line with no matching box stays right after the line before it.

The cluster threshold T and the text match threshold M were tuned on the 64 train sequences only (T = 0.3, M = 0.5).

Qwen training details:

* LoRA rank 16, alpha 32, on the attention and MLP layers of the language model only. The vision encoder stays frozen.
* One training example is one page plus its earlier pages as context. Loss is only on the answer tokens.
* 4 epochs, learning rate 1e-4, gradient accumulation 8, bf16, images at native resolution.
* Split: 64 train and 16 validation sequences, split by whole sequence (seed 0), never by page. The same split is used in every version.
* After validation, the same recipe is retrained on all 80 dev sequences, and that model writes the test predictions.

## Experiments

All numbers are from the official `score.py` on the same 16 validation sequences.

| Version | What changed | text_order_score | balanced_joint_f1 |
|---|---|---|---|
| v1 zero shot | 3 pages in one prompt, no training | 0.31 | 0.15 |
| v1 LoRA | same, fine tuned | 0.71 | 0.26 |
| v2 zero shot | one page per call, no training | 0.45 | 0.16 |
| v2 LoRA | one page per call, fine tuned | 0.89 | 0.31 |
| v3 | v2 with 2 epochs, plus Magi speakers | 0.76 | 0.40 |
| **v4 (final)** | v2 recipe, plus Magi speakers and Magi reading order | **0.92** | **0.43** |

### v1: three pages at once

I gave the model all three pages and asked for the whole transcript in one go, thinking that seeing everything together would help with speakers. Fine tuning took text order from 0.31 to 0.71. Looking at the scores page by page, most pages were already near perfect. The lost score came from page bookkeeping: in 2 of 16 sequences page 3 came out empty, text got written under the wrong page, and an empty page got filled with text from another page.

### v2: one page per call

Making the model write one page per call removed all of that. Zero shot went from 0.31 to 0.45 and the fine tuned model from 0.71 to 0.89. Speakers stayed weak at 0.31 because Qwen only guesses labels from the text.

### v3: fewer epochs, Magi for speakers

v2's validation loss was lowest after epoch 2 (0.37, 0.20, 0.15, 0.18, 0.16), so I tried 2 epochs. That was a mistake. Text order fell from 0.89 to 0.76. Lower loss did not mean better transcripts, so I went back to 4 epochs.

The other change in v3 worked. Taking speakers from Magi raised balanced joint F1 from 0.33 to 0.40 on the same Qwen text. In the threshold search on train, using the gold text so only the speaker part changes, Magi took speaker F1 from 0.31 to 0.56.

### v4: Magi for reading order too

Before v4 I checked where v2 still lost text score on validation. If every line were in the right order, v2 would score 0.93 instead of 0.89. Removing extra lines would add about 0.01, and recovering the 41 lines it missed (out of 362) about 0.02 more. So order was the biggest loss.

To test Magi's order on its own, I put the gold text of the 64 train sequences into Magi's reading order and scored it: 0.99. That meant Magi's order was worth trusting. v4 tries every combination on validation and picks the best:

| v4 variant | text_order_score | balanced_joint_f1 |
|---|---|---|
| Qwen only | 0.889 | 0.343 |
| Magi speakers | 0.889 | 0.415 |
| **Magi speakers + Magi order** (chosen) | **0.923** | **0.427** |
| Magi speakers + Magi text for missed boxes | 0.854 | 0.404 |
| Magi speakers + order + missed text | 0.886 | 0.417 |
| Magi speakers + drop text Magi calls non story | 0.885 | 0.415 |

Magi order improved 10 of the 16 sequences, left 4 the same and made 2 worse. Adding Magi's own OCR for boxes Qwen skipped made things worse, because Magi's OCR is clearly weaker than Qwen at reading the text. Dropping lines Magi marks as non story text did not help either.

The Qwen part of v4 reproduced v2 exactly (0.889), which confirms the v3 drop came from the epoch change.

## What worked and what did not

Worked:

* LoRA fine tuning on 64 sequences. It taught Qwen the output format, what to skip (sound effects, signs) and how to read pages in manga order.
* One page per call. Biggest single jump in the project.
* Using Magi for the two things Qwen is bad at: who is speaking, and the order of balloons across panels.
* A loop cleaner in the parser. In v1 one test sequence repeated the same two lines about 40 times.

Did not work:

* Picking epochs by validation loss (v3).
* Magi's OCR for lines Qwen missed.
* Filtering lines with Magi's non story text classifier.

Still weak:

* **Speakers.** 0.43 is much better than 0.31 but still the weakest part. Off panel speakers, crowd lines and characters drawn very differently across panels get split or merged wrongly. Magi made a few sequences worse (one went from 0.79 to 0.46).
* **Order on some pages.** Magi order made 2 sequences worse: `seq_a3bd36201a0cd61b` went from 0.90 to 0.83 and `seq_06ec0acde37dffa4` from 0.90 to 0.82.
* **Hardest page set.** `seq_2032620aa4e4ac7f` is still the lowest at 0.56.

## What I would try next

* Use Magi order only when it disagrees with Qwen at the panel level, instead of always.
* Fine tune Magi's character embeddings on the dev labels, since the clustering threshold is the main thing driving speaker quality.
* A bigger reader, Qwen2.5 VL 32B, which should fit on the same 96 GB GPU.
* Use the speech balloon tails more directly when Magi's text to character link is missing.

## How to run

Everything is one marimo notebook, run on molab with a GPU (RTX Pro 6000, 96 GB).

1. Open `notebooks/manga_task_v4_final.py` on molab and turn the GPU on.
2. Click Run all. The dataset downloads from the Google Drive link already in the second cell.
3. It installs packages, checks the output format, trains Qwen on 64 sequences, runs Magi on all pages, tunes the Magi thresholds on train, scores every variant on validation, retrains Qwen on all 80, writes `predictions.jsonl`, and gives a download in the last cell.

A full run takes about an hour. Seeds are fixed and decoding is greedy. Magi's raw outputs are saved in `results/v4_final/magi_outputs.json.gz`, so the speaker and order step can be redone from `predictions_qwen_only.jsonl` without a GPU. I checked this and it gives exactly the same `predictions.jsonl`.

## Files

| Path | What it is |
|---|---|
| `predictions.jsonl` | Final test predictions (15 sequences), from v4 |
| `notebooks/manga_task_v4_final.py` | Final pipeline |
| `notebooks/manga_task_v3.py` | v3, 2 epochs and Magi speakers |
| `notebooks/manga_task_v2.py` | v2, one page per call |
| `notebooks/manga_task_v1.py` | v1, 3 pages in one prompt |
| `results/v4_final/` | Validation table, every variant's scores, threshold search, training logs, raw outputs, Magi outputs |
| `results/v1/`, `results/v2/`, `results/v3/` | Same for the earlier versions |
| `adapters/` | LoRA configs of the final model. Weights are on Drive (link below) |
| `requirements.txt` | Package versions |

**Trained LoRA weights (v4):** https://drive.google.com/drive/folders/1I8l-8jSxBofbJ7Z1tDxfuX_i0D9SolJ7?usp=sharing

`lora_adapter_all80` made the test predictions. `lora_adapter_train64` is the one scored on validation. Each is about 155 MB, too big for GitHub.

## Credits

* Qwen2.5 VL by the Qwen team, Alibaba: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
* Magi v2, Sachdeva, Shin and Zisserman, "Tails Tell Tales: Chapter Wide Manga Transcriptions with Character Names" (2024): https://huggingface.co/ragavsachdeva/magiv2 (free for research and non commercial use)
* Hugging Face Transformers: https://github.com/huggingface/transformers
* PEFT for LoRA: https://github.com/huggingface/peft
* SciPy for the clustering: https://scipy.org
* marimo and molab: https://marimo.io
* gdown for the Drive download: https://github.com/wkentaro/gdown
* The official BYTE scoring script, used for every number above.

My own parts are the page by page prompting and training setup, the parser and loop cleaner, the error analysis, and the code that joins Magi with Qwen (text matching, cross page character clustering, speaker voting and reordering).

No hosted AI API was used to make training targets or predictions. Everything was produced by the open weight models above. I used an AI assistant (Claude) to help write code and to draft this README, which the task brief allows if disclosed.
