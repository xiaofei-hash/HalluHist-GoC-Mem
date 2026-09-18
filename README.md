# HalluHist and GoC-Mem

This repository contains HalluHist, a paired-history benchmark for multimodal
hallucination snowballing, and the core implementation of GoC-Mem, a
Graph-of-Claims memory controller for multi-turn multimodal dialogue.

## Repository structure

```text
data/                 HalluHist inputs, labels, and data documentation
goc_mem/              claim graph, verification normalization, pruning, memory
goc_mem/prompt_templates/  the three prompt templates reported in the paper
scripts/              dataset validation and paired-metric scoring
tests/                unit tests for the core mechanism
```

## HalluHist

HalluHist provides 500 paired targets (1,000 inputs). Every pair shares the
image, historical user questions, final question, and reference answer while
contrasting a contaminated history with a grounded history. The released
evaluation measures are contaminated-history accuracy (`ACC_c`),
grounded-history accuracy (`ACC_g`), and paired accuracy (`PairAcc`).

Image files come from GQA and are downloaded separately. See
[`data/README.md`](data/README.md) for the schema and setup.

## GoC-Mem core

The backend-neutral pipeline implements the method described in the paper:

1. extract all atomic visual claims from historical assistant responses;
2. add and verify missing existence anchors together with all extracted claims;
3. use `tau_c` only to normalize low-confidence supported/contradicted labels
   to `UNCERTAIN`;
4. remove contradicted seeds and their dependency descendants;
5. render supported facts, uncertain claims, and query-relevant corrections;
6. answer the final question using the image and reconstructed memory.

`goc_mem.pipeline.Backend` is the only model-specific interface. Implement
`generate_text(prompt, image_path=None)` for the desired multimodal backend,
then call `goc_mem.pipeline.run_sample`. The default threshold is supplied by
the caller (`tau_c=0.6` in the paper).

The implementation verifies every extracted claim and auxiliary anchor and
does not impose a claim-count or character-length limit during reconstruction.
It constructs only explicit entity-existence prerequisite edges; temporal
adjacency alone does not create a dependency.

## Prompts

The files in `goc_mem/prompt_templates/` are the executable templates corresponding to the
paper's Prompt Design section:

- [`claim_extraction.txt`](goc_mem/prompt_templates/claim_extraction.txt)
- [`image_verification.txt`](goc_mem/prompt_templates/image_verification.txt)
- [`memory_guided_answering.txt`](goc_mem/prompt_templates/memory_guided_answering.txt)

## Setup and checks

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/validate_dataset.py
```

## Citation

```bibtex
@article{wu2026gocmem,
  title   = {GoC-Mem: Benchmarking and Mitigating Hallucination Snowballing in Multi-Turn Multimodal Dialogue},
  author  = {Wu, Mengfei and Zheng, Yuhui and Wu, Xia},
  year    = {2026}
}
```

## Acknowledgements

The HalluHist image pool follows MMHalSnowball and uses GQA images. Please
also cite the original MMHalSnowball and GQA resources when using the data.
