# GoC-Mem: Dependency-Aware Memory Revision and Paired-History Benchmarking for Multimodal Hallucination Snowballing

**Authors:** Mengfei Wu, Hao Shi, Xia Wu, and Yuhui Zheng.

**Corresponding author:** Yuhui Zheng (`zhengyh@vip.126.com`).

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

The backend-neutral pipeline provides the core claim-graph operations:

1. extract all atomic visual claims from historical assistant responses;
2. add and verify missing existence anchors together with all extracted claims;
3. use `tau_c` only to normalize low-confidence supported/contradicted labels
   to `UNCERTAIN`;
4. remove contradicted seeds and their dependency descendants;
5. render supported facts and qualified uncertain claims; retain corrections only in diagnostics;
6. answer the final question using the image and reconstructed memory.

`goc_mem.pipeline.Backend` is the only model-specific interface. Implement
`generate_text(prompt, image_path=None)` for the desired multimodal backend,
then call `goc_mem.pipeline.run_sample`. The default threshold is supplied by
the caller (`tau_c=0.6` in the paper).

It constructs explicit entity-existence prerequisite edges; temporal
adjacency alone does not create a dependency.

### Release scope and paper configuration

This release provides the backend-neutral core and updated answering template.
The paper's Qwen experiment controller uses separate selection and rendering
budgets: at most 8 verified nodes and at most 4 historical claims / 800 characters
in the final memory. The public core verifies all extracted claims and auxiliary
anchors and renders memory without these limits. The frozen controller used for
the paper's answer-only reruns is not included in this release; the updated
template alone is therefore not a complete reproduction configuration.

## Prompts

The files in `goc_mem/prompt_templates/` are the executable templates corresponding to the
paper's Prompt Design section:

- [`claim_extraction.txt`](goc_mem/prompt_templates/claim_extraction.txt)
- [`image_verification.txt`](goc_mem/prompt_templates/image_verification.txt)
- [`memory_guided_answering.txt`](goc_mem/prompt_templates/memory_guided_answering.txt)

### Answering-template update (2026-09-23)

The answering prompt now begins:

```text
Use the context to understand the question and resolve references.
Base your answer on visual evidence.
```

See [the expanded template](prompts/memory_guided_answering.txt). Supported
claims appear as quoted `Verified history`; uncertain claims retain their
verification confidence and the `Uncertain history` qualification. Empty
sections are omitted independently. Corrections are retained in diagnostics
but are not rendered in the revised answer prompt. The answer instruction
requests a single word or short phrase, rather than forcing every task to Yes/No.

Use a fresh output directory when running the updated template. For the
relationship to the paper's experimental configuration, see
[Release scope and paper configuration](#release-scope-and-paper-configuration).

## Setup and checks

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/validate_dataset.py
```

## Citation

```bibtex
@article{wu2026gocmem,
  title   = {GoC-Mem: Dependency-Aware Memory Revision and Paired-History Benchmarking for Multimodal Hallucination Snowballing},
  author  = {Wu, Mengfei and Shi, Hao and Wu, Xia and Zheng, Yuhui},
  year    = {2026}
}
```

## Acknowledgements

The HalluHist image pool follows MMHalSnowball and uses GQA images. Please
also cite the original MMHalSnowball and GQA resources when using the data.
