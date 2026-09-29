# MedKIT

[![Paper](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-MedKIT-yellow)](https://huggingface.co/datasets/bethgelab/MedKIT)

Official code for **MedKIT: Evaluating Knowledge Integration and
Generalization in Large Language Models**, accepted to the
**NeurIPS 2026 Evaluations & Datasets Track**
([paper](https://arxiv.org/abs/XXXX.XXXXX)).

MedKIT is a benchmark for evaluating how language models integrate and
apply new knowledge under realistic sequences of clinical updates. This
repository contains the code needed to (1) construct the benchmark from
the upstream HemOnc.org data and linked PubMed evidence and (2) reproduce
the paper's experiments on knowledge integration and generalization.

Cluster-specific settings are kept separate from the experiment code:
SLURM jobs use a generic template, while environment variables provide
paths and API keys.

## Repository layout

```
.
├── data_construction/      # Construct MedKIT from HemOnc.org + PubMed evidence
│   ├── README.md
│   ├── process_medkit.py   # Main construction pipeline
│   ├── add_conflict_flag.py
│   ├── docs/               # Construction documentation
│   └── raw_hemonc/         # Fixed HemOnc.org snapshot (2026-03-12)
└── experiments/            # Run knowledge-integration experiments
    ├── README.md
    ├── run_medkit.py       # Hydra entry point for an experiment
    ├── hemonc_batching.py  # Runtime batch construction for sequential updates
    ├── precompute_stats.py # One-shot cov stats for MEMIT / AlphaEdit
    ├── generate_memoir_features.py  # One-shot background features for MEMOIR
    ├── easyeditor/         # Method implementations (editing, continual, RAG)
    ├── further_baselines/  # DPO / GRPO / Agentic-RAG baselines + shared-eval drivers
    ├── hydra/experiments/  # Hydra configs (base hparams + 296 paper configs)
    ├── main_experiments/   # Sweep generator, submit + monitor scripts, SLURM template
    └── hparam_tuning/      # Best-params CSV used by the config generator
```

The two subprojects can be used independently:
`data_construction/` constructs the MedKIT benchmark published on the
Hugging Face Hub, while `experiments/` consumes the resulting dataset to
evaluate knowledge-integration methods. Each directory has its own
README with installation and usage instructions.

## Published dataset

The constructed MedKIT benchmark is available on the Hugging Face Hub:

https://huggingface.co/datasets/bethgelab/MedKIT

The experiment configs point directly to the published dataset
(`qa.data_path: 'hf://bethgelab/MedKIT'`) and download it on first use.
Reproducing the paper's experiments therefore does not require rerunning
the benchmark construction pipeline.

## License

- **Code** in this repository: Apache License 2.0 (see `LICENSE`).
- **MedKIT benchmark** published on Hugging Face: CC BY-NC-SA 4.0,
  consistent with the licensing terms of the underlying HemOnc.org data.
- **Raw HemOnc.org snapshot** bundled under
  `data_construction/raw_hemonc/`: CC BY-NC-SA 4.0. See
  `data_construction/raw_hemonc/NOTICE.md` for attribution.
- **PubMed metadata and abstracts** used by the construction pipeline
  are retrieved through publicly available NCBI services. Downstream
  users should comply with the applicable NCBI/NLM terms of use.

## Citation

If you use MedKIT, please cite:

```bibtex
@inproceedings{thede2026medkit,
  title     = {MedKIT: Evaluating Knowledge Integration and Generalization in Large Language Models},
  author    = {Thede, Lukas and Atri, Yash Kumar and Chen, David and Bitterman, Danielle and Bethge, Matthias and Hartvigsen, Tom and Akata, Zeynep},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

Please also cite HemOnc.org (see
`data_construction/raw_hemonc/NOTICE.md`).