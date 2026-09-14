# Multi-Judge Committee for Hallucination Span Detection in Vision-Language Outputs

Code for our submission to the **SHROOM-Visions 2026** shared task on detecting and
classifying hallucinated character spans in vision-language model (VLM) outputs,
co-located with the UncertainLP workshop at EMNLP 2026.

The system treats several fine-tuned VLMs as independent annotators and combines
their span predictions through character-level voting. It ranked **first in French,
Italian and Chinese**, and placed on the podium in every language and evaluation
metric.

## Results

Challenge set, committee with votes ≥ 2:

| Metric | EN | FR | IT | ZH | Avg. |
|---|---|---|---|---|---|
| Cor_lbl | 0.4251 | 0.4738 | 0.4511 | 0.5040 | 0.4635 |
| Cor | 0.5334 | 0.5873 | 0.5628 | 0.6083 | 0.5730 |
| IoU | 0.4486 | 0.5150 | 0.4793 | 0.5344 | 0.4943 |

Official rankings: https://shroom.pythonanywhere.com/submission/

## Approach

Given an image, a question and a VLM response, each judge returns the response
verbatim with hallucinated spans wrapped in inline markup:

```
<hall label="LABEL" prob="PROBCLASS">hallucinated text</hall>
```

`LABEL` is one of `invention`, `mischaracterization`, `OCR`, `miscounting`, `other`.
`PROBCLASS` is `prob1` (≈0.33), `prob2` (≈0.67) or `prob3` (≈1.0), reflecting how
many of three annotators would be expected to agree.

Predictions are parsed to character offsets and aggregated at the character level.
A character is kept when at least two judges flag it, and its probability is the
average of the flagging judges' confidences over the full committee. This mirrors
the gold annotation, which is the union of three independent annotators.

### Judges

Five judges from four backbone families, each fine-tuned on a distinct data sample
to decorrelate their errors:

| Judge | Backbone | Setting |
|---|---|---|
| Gemma-4-ft | Gemma-4 | fine-tuned, 0-shot |
| Gemma-4-6shot | Gemma-4 | few-shot (6 demonstrations) |
| Mistral-small-ft | Mistral-Small-3.1-24B | fine-tuned, 0-shot |
| Qwen3.6-ft | Qwen3.6-27B | fine-tuned, 0-shot |
| Qwen3-vl-ft | Qwen3-VL-30B-A3B | fine-tuned, 3-shot |

Few-shot demonstrations are retrieved by weighted cosine similarity (image 0.50,
response 0.30, prompt 0.20), using
[CLIP](https://huggingface.co/openai/clip-vit-large-patch14) for images and
[BGE-M3](https://huggingface.co/BAAI/bge-m3) for text.

Fine-tuning uses LoRA (rank 64, alpha 16, dropout 0.05) with rsLoRA and LoRA+ at a
ratio of 4, two epochs, learning rate 5e-5, batch size 1 with gradient accumulation
over 8 steps, sequence length 6192, bf16.

## Installation

```bash
git clone https://github.com/toqeerehsan/vlm_hallucination_detect.git
cd vlm_hallucination_detect
pip install -r requirements.txt
```

Python 3.10+. Running the judges needs a GPU; aggregation and scoring run on CPU.

## Usage

```bash
# aggregate judge outputs into a committee
bash run_committee.sh

# score predictions against gold
python scorer.py gold.jsonl predictions.jsonl scores.txt
```

Set `LABELED_SET=1` in `run_committee.sh` for the labeled split (with scoring) or
`LABELED_SET=0` for the unlabeled challenge set. The vote threshold is set with
`MIN_VOTES`.

## Data

The SHEEP dataset is released by the task organizers under CC-BY-NC and is available
from the [task page](https://helsinki-nlp.github.io/shroom/2026). It is not
redistributed here.

## Citation

```bibtex
@inproceedings{ehsan2026vroomvroom,
  title     = {Vroom-Vroom at {SHROOM}-Visions: A Multi-Judge Committee for
               Detecting Hallucinated Spans in Vision-Language Outputs},
  author    = {Ehsan, Toqeer and Penttil{\"a}, Nico and Schmidt, Richard and
               Hajikhani, Arash and Palacin, Victoria},
  booktitle = {Proceedings of the 3rd Workshop on Uncertainty-Aware NLP
               (UncertaiNLP 2026)},
  year      = {2026}
}
```

## License

Code is released under the MIT License. Fine-tuned adapters inherit the licenses of
their base models and the non-commercial terms of the SHEEP dataset.
