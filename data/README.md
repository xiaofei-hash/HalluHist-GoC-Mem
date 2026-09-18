# HalluHist data

HalluHist contains 500 paired targets and 1,000 evaluation inputs. Each pair
uses the same image and final Yes/No question with two alternative histories:
`contaminated` and `grounded`.

## Files

- `evaluation_inputs.jsonl`: model-visible inputs only (`sample_id`, relative
  `image_path`, history, and final question).
- `evaluation_labels.jsonl`: held-out answers and diagnostic annotations. Do
  not expose this file to the evaluated model.

The benchmark is balanced over five dialogue lengths (3--7 turns), five final
question categories, and five primary history configurations. Each marginal
group contains 100 pairs; every length/category/configuration design cell
contains four pairs. The answer distribution contains 250 Yes and 250 No
paired targets.

## Images

Image files are not redistributed in this repository. The image IDs refer to
GQA images, following the source pool used by MMHalSnowball. Download the GQA
image archive from the official source and place the required JPEG files under
`images/`; the paths in `evaluation_inputs.jsonl` then resolve directly.

```bash
wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
unzip images.zip
```

The original MMHalSnowball repository documents the same image preparation:
https://github.com/whongzhong/MMHalSnowball

## Construction and evaluation

GPT-5 through ChatGPT assisted the construction of historical responses from
the image, assigned design cell, and available visual evidence. Histories were
paired so that the user questions, image, final question, and reference answer
remain shared while assistant responses differ. The released labels include
claim spans, entity keys, factual labels, and annotated prerequisite relations
for analysis. These annotations are evaluation-only metadata.

Three human annotators independently assessed a stratified sample of 125
pairs, one per design cell. Acceptance rates and mean pairwise agreement are
reported in the paper.

Use `python scripts/validate_dataset.py` to verify pairing and the benchmark
allocation. Predictions must contain one JSON object per line with
`sample_id` and `answer`; score them with:

```bash
python scripts/score.py --predictions predictions/model.jsonl
```
