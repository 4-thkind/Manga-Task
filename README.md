# BYTE Manga Task (Advanced)

The task: given three consecutive English manga pages, pull out every line of story text in the order a reader would read it, and say who spoke each line. The same character has to keep the same label across all three pages. Sound effects, signs, credits and other non story text must be left out.

My final system joins two open weight models. A fine tuned **Qwen2.5 VL 7B** reads the text page by page, and **Magi v2**, a model made for manga, decides who is speaking and in what order the balloons should be read.

## Final results

All numbers come from the official `score.py`, on 16 development sequences that were held out from training the whole time.

| Metric | Qwen alone (fine tuned) | **Final (Qwen + Magi)** |
|---|---|---|
| text_order_score | 0.889 | **0.923** |
| balanced_joint_f1 | 0.343 | **0.427** |
| speaker_accuracy_on_matched | 0.536 | **0.627** |
| joint_f1 | 0.494 | **0.599** |
| matched_token_coverage | 0.931 | **0.958** |

What these mean in plain words:

* **text_order_score 0.923.** Per page, the predicted text is compared with the correct text in reading order, and an edit distance turns that into a score. 0.923 means a page is about 92% right on average, counting both the words and their order. This is the main accuracy number.
* **matched_token_coverage 0.958.** About 96% of the words in the correct answer were found in my output.
* **balanced_joint_f1 0.427.** This measures speakers and words together, and every character counts equally, so a minor character with two lines matters as much as the main one. It is the hardest number to push up.
* **speaker_accuracy_on_matched 0.627.** For the words that were found, about 63% got the right speaker.

The test predictions (`predictions.jsonl`) cover all 15 test sequences: 319 lines in total, 19 of them labelled as narration, one page predicted as having no story text, and 2 to 13 speaker labels per sequence (about 4 on average). The file passes the format checks in the official scorer.

## The data

* 80 development sequences with answers and 15 test sequences without. Each sequence is three page images.
* The development answers have 1753 lines in total. 10 pages have no story text at all, 76 lines are narration, and 34 lines have an unknown speaker (the scorer ignores those for speakers).
* On average a sequence has about 3.7 different characters speaking.
* I split the 80 development sequences into 64 for training and 16 for validation, by whole sequence, never by page, so no page of a validation sequence was ever seen in training. The split (seed 0) is the same in every version, which is why all numbers in this README can be compared directly.

## How the final system works

### Step 1: Qwen reads the text, one page at a time

The reader is Qwen2.5 VL 7B Instruct, a vision language model. For each page it gets:

* the images of the current page and every earlier page in the sequence
* the transcript it already wrote for the earlier pages
* a short instruction: write only this page, one balloon per line as `speaker: text`, skip sound effects and signs, read right to left and top to bottom, reuse the labels from earlier pages, write `NONE` if the page has no story text

Writing one page per call keeps the model from mixing pages up, and giving it its own earlier transcript lets it carry speaker labels forward.

A small parser turns the output into the required JSON. It also cuts generation loops. Sometimes the model starts repeating the same line or block of lines until it runs out of tokens. The labels never repeat a line more than four times in a row or a block of lines more than once, so anything beyond that is cut.

**Training.** The base model does not know the output format or what to skip, so I fine tuned it with LoRA:

* LoRA rank 16, alpha 32, on the attention and MLP layers of the language model. The vision encoder stays frozen.
* One training example is one page, with the earlier pages and their correct transcript as context. The loss is only on the answer.
* 4 epochs, learning rate `1e-4`, gradient accumulation 8, bf16, images at their native resolution.
* Validation loss over the epochs: 0.37, 0.20, 0.16, 0.19, 0.18.

### Step 2: Magi looks at the page layout

Magi v2 is used as it is, with no training. For every page it gives:

* every text box, already sorted into manga reading order (it orders the panels, then the boxes inside each panel)
* its own OCR text for each box
* every character box, and which character each text box belongs to (it uses the balloon tails for this)
* an embedding for every character crop, plus groups of crops it thinks are the same character on that page

### Step 3: joining the two

Qwen is good at reading and Magi is good at layout, so each Qwen line is matched to the Magi text box with the most similar text, and then:

* **Speaker.** The line gets the character that Magi links to its box. Magi only groups characters within one page, so I make the labels consistent across the three pages myself. I average the crop embeddings of each group on each page and cluster those groups across all three pages (average linkage, cosine distance, threshold T). Each cluster becomes one label. Lines that match no box take the cluster that most other lines with the same Qwen label went to. Narration stays as Qwen labelled it.
* **Order.** Lines are sorted by the reading order of the Magi box they matched. A line with no matching box stays right after the line it followed in Qwen's output.

The cluster threshold T and the text match threshold M were tuned on the 64 training sequences only, never on validation or test. For tuning I used the correct text with every speaker replaced by one label, so the score only measured what Magi adds. With that setup, speaker F1 on train went from 0.31 without Magi to 0.56 with it, at T = 0.3 and M = 0.5.

### Step 4: picking the final setup and making the test predictions

The notebook scores six ways of combining the two models on the 16 validation sequences and keeps the one with the best text order plus speaker score:

| Variant | text_order_score | balanced_joint_f1 |
|---|---|---|
| Qwen only | 0.889 | 0.343 |
| Magi speakers | 0.889 | 0.415 |
| **Magi speakers + Magi order (final)** | **0.923** | **0.427** |
| Magi speakers + Magi OCR for boxes Qwen missed | 0.854 | 0.404 |
| Magi speakers + order + Magi OCR for missed boxes | 0.886 | 0.417 |
| Magi speakers + drop lines Magi calls non story | 0.885 | 0.415 |

After that, Qwen is retrained with the same recipe on all 80 development sequences. That model reads the test pages, Magi runs on them, and the chosen variant joins the two into `predictions.jsonl`.

## How I got here

Four versions, all scored on the same 16 validation sequences.

| Version | Idea | text_order_score | balanced_joint_f1 |
|---|---|---|---|
| v1 zero shot | all 3 pages in one prompt, no training | 0.31 | 0.15 |
| v1 fine tuned | same, with LoRA | 0.71 | 0.26 |
| v2 zero shot | one page per call, no training | 0.45 | 0.16 |
| v2 fine tuned | one page per call, with LoRA | 0.89 | 0.31 |
| v3 | v2 with 2 epochs, Magi for speakers | 0.76 | 0.40 |
| **v4 (final)** | v2 recipe, Magi for speakers and order | **0.92** | **0.43** |

**v1, three pages at once.** I gave the model all three pages and asked for the full transcript in one answer, hoping that seeing everything together would help with speakers. Fine tuning took text order from 0.31 to 0.71. When I looked at the scores page by page, most pages were already close to perfect. The lost points came from page bookkeeping: in 2 of 16 sequences page 3 came out empty, some text was written under the wrong page, and an empty page got filled with text from its neighbour.

**v2, one page per call.** Asking for one page at a time fixed all of that. Without any training it already beat v1 (0.31 to 0.45), and fine tuned it reached 0.89. Speakers stayed weak at 0.31, because Qwen can only guess who is talking from the words.

**v3, two changes at once.** v2's validation loss was lowest after epoch 2, so I trained for 2 epochs instead of 4. That was a mistake: text order fell to 0.76. Lower loss did not mean better transcripts. The other change, taking speakers from Magi, worked and raised speaker F1 on the same text from 0.33 to 0.40.

**v4, Magi for order as well.** Before v4 I measured where v2 still lost text points on validation. If every line had been in the right order, v2 would have scored 0.93 instead of 0.89. Removing its extra lines was worth about 0.01, and recovering the 41 lines it missed (out of 362) about 0.02 more. Order was clearly the biggest loss. To check Magi's order on its own, I put the correct text of the 64 training sequences into Magi's reading order and scored it: 0.99. So Magi's order was worth trusting, and using it for Qwen's lines took text order from 0.89 to 0.92. The Qwen part of v4 reproduced v2's 0.889 exactly, which confirms that the v3 drop came only from the epoch change.

## Results per validation sequence

Each row is one of the 16 held out sequences. "Text" is its text_order_score and "Speakers" its balanced_joint_f1, first for fine tuned Qwen alone and then for the final system with Magi. It shows where Magi helped and where it hurt.

| Sequence | Text, Qwen | Text, final | Speakers, Qwen | Speakers, final |
|---|---|---|---|---|
| `seq_06ec0acde37dffa4` | 0.90 | 0.82 | 0.32 | 0.26 |
| `seq_2032620aa4e4ac7f` | 0.49 | 0.56 | 0.38 | 0.54 |
| `seq_2e4b837ead6b88ae` | 0.89 | 0.89 | 0.38 | 0.16 |
| `seq_44b9a316b2919cfe` | 0.96 | 1.00 | 0.26 | 0.37 |
| `seq_637459fc880bf35a` | 0.98 | 0.99 | 0.32 | 0.48 |
| `seq_83ff2f28f5c13c40` | 1.00 | 1.00 | 0.79 | 0.46 |
| `seq_864aaca43a3580bb` | 0.89 | 0.99 | 0.22 | 0.36 |
| `seq_91119fd54cd62066` | 0.98 | 0.98 | 0.14 | 0.50 |
| `seq_98d98dac2a89a339` | 0.77 | 0.90 | 0.33 | 0.42 |
| `seq_a3bd36201a0cd61b` | 0.90 | 0.83 | 0.02 | 0.04 |
| `seq_ac5793f8168efdd1` | 0.86 | 0.91 | 0.14 | 0.25 |
| `seq_b9b5a9ad9a64ced9` | 0.91 | 0.99 | 0.49 | 0.59 |
| `seq_d7af6c64ebc22bbe` | 1.00 | 1.00 | 0.50 | 0.99 |
| `seq_ee880b9612205860` | 0.87 | 0.93 | 0.30 | 0.12 |
| `seq_f1c007d2d64e6aad` | 1.00 | 1.00 | 0.68 | 1.00 |
| `seq_f9a4e67ae3c018f6` | 0.82 | 0.98 | 0.22 | 0.30 |

In short: text order went up on 10 sequences, stayed the same on 4 and went down on 2. Speakers went up on 12 and down on 4. The big text gains (0.77 to 0.90, 0.82 to 0.98, 0.89 to 0.99) come only from order. The final system keeps exactly the lines Qwen wrote and only changes their order, so those were pages where Qwen read the lines but put some in the wrong order.

## What worked and what did not

Worked:

* LoRA fine tuning on just 64 sequences. It taught Qwen the format, what to skip and how to read manga pages.
* One page per call, the biggest single jump in the project.
* Splitting the job between two models: Qwen for reading, Magi for layout and characters.
* Checking ideas against the correct text first (the 0.99 order check, the 0.31 to 0.56 speaker check) before trusting them on validation.

Did not work:

* Choosing the number of epochs from validation loss (v3).
* Using Magi's OCR for lines Qwen missed. Magi's reading of the text is clearly worse than Qwen's, so it added more wrong text than it recovered.
* Dropping lines Magi's classifier calls non story. It removed some real dialogue along with the sound effects.

Still weak:

* **Speakers.** 0.43 is a clear step up from 0.31, but it is still the weakest part. Off panel speakers, crowd lines and characters drawn very differently from panel to panel get split or merged wrongly. On 4 sequences Magi's speakers were worse than Qwen's, the biggest drop being 0.79 to 0.46.
* **Order on some pages.** Magi's order hurt 2 sequences: `seq_a3bd36201a0cd61b` went from 0.90 to 0.83 and `seq_06ec0acde37dffa4` from 0.90 to 0.82.
* **The hardest sequence.** `seq_2032620aa4e4ac7f` is still the lowest at 0.56.
* **Small validation set.** With only 16 sequences, a difference of about 0.01 between two variants is within noise. The jumps that matter here (0.71 to 0.89 to 0.92) are much bigger than that.

## What I would try next

* Use Magi's order only where it disagrees with Qwen at the panel level, instead of always.
* Fine tune Magi's character embeddings on the development labels, since the clustering step drives speaker quality.
* A bigger reader, Qwen2.5 VL 32B, which should fit on the same GPU.
* Lean on the balloon tails directly when Magi has no link from a text box to a character.

## Files

| Path | What it is |
|---|---|
| `predictions.jsonl` | Final test predictions (15 sequences), from v4 |
| `notebooks/manga_task_v4_final.py` | Final pipeline |
| `notebooks/manga_task_v3.py` | v3, 2 epochs and Magi speakers |
| `notebooks/manga_task_v2.py` | v2, one page per call |
| `notebooks/manga_task_v1.py` | v1, 3 pages in one prompt |
| `results/v4_final/` | Validation table, every variant's scores and predictions, threshold search, training logs, raw outputs, Magi outputs |
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
* marimo: https://marimo.io
* gdown for the Drive download: https://github.com/wkentaro/gdown
* The official BYTE scoring script, used for every number above.

No hosted AI API was used to make training targets or predictions. Everything was produced by the open weight models above. .