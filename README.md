# Kazakh Programming Education — Controlled Morphology-Aware Answer Selection

Research software for the technical component of the *Education Sciences* manuscript on AI-supported programming education in Kazakh. This component is **empirically independent** of the classroom pilot.

**GitHub:** https://github.com/rsbekarys-create/Education  
**Archived release (Zenodo):** https://doi.org/10.5281/zenodo.24230791

## Scope

The experiment compares four systems for each of two language-model backbones:

1. `BASE_RAW` — pretrained model with raw question text;
2. `BASE_FEMSEG_MAPP` — pretrained model with FEMSeg_kaz + MAPP question representation;
3. `FT_RAW` — LoRA fine-tuning with raw question text;
4. `FT_FEMSEG_MAPP` — LoRA fine-tuning with morphology-aware representation.

Models: `Qwen/Qwen2.5-1.5B-Instruct` and `issai/LLama-3.1-KazLLM-1.0-8B`.

The released notebooks use **question-to-question contrastive fine-tuning**, question-embedding retrieval against the **training-question pool**, and return the corresponding training answer. Do not describe this as direct supervised question-to-answer LoRA training. Results for each backbone are evaluated separately.

## Repository layout

The intended release layout is:

```text
Education/
├── README.md
├── LICENSE
├── requirements.txt
├── notebooks/
│   ├── qwen25_fair_4way.ipynb
│   └── kazllm_fair_4way.ipynb
├── data/
│   └── python_dataset.json       # only if redistribution is permitted
└── models/
    ├── char2id_femseg_v3_50k.json
    └── femseg_v3_50k_epoch3.pt   # only if redistribution is permitted
```

The paths above describe the expected files, **not a verification that every file is present in the public archive**. Compare against the GitHub release and Zenodo record before submitting the manuscript.

## Dataset and splitting

The supplied dataset contained 5,944 Kazakh-language programming question–answer entries; 219 duplicate entries were identified, leaving 5,725 unique entries under the earlier pair-based count. The actual notebooks perform their own normalization/deduplication and then apply `GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=42)` **by normalized question group** to avoid overlap. Because this operates on groups, exact train/test *row* counts need to be confirmed in `split_manifest.csv` for each recorded run; they must not be inferred as 5,153/572 solely from the test fraction.

The original generation is described by the authors as curriculum-aligned and GPT-5 assisted. Exact original prompt logs, model snapshot, and generation settings are not supplied in these repository materials. Any reconstructed prompt must be labelled as such.

## Reproduction steps (Google Colab)

1. Open one notebook in a GPU-enabled Google Colab runtime. GPU VRAM requirements depend on model and workload.
2. Install dependencies using the notebook's installation cell. The separately supplied `requirements.txt` records the author's **November 2026 Colab environment** (Python 3.13.16); it does **not** establish the original historical experiment environment. The notebook's install cell is not fully pinned and may alter package versions when executed.
3. Ensure `python_dataset.json` is available at the `DATA_JSON_PATH` configured in the notebook.
4. Place `char2id_femseg_v3_50k.json` and `femseg_v3_50k_epoch3.pt` where `FEMSEG_CHAR2ID` and `FEMSEG_CKPT` point, or edit these two paths.
5. For KazLLM, obtain model access on Hugging Face and configure the `HF_TOKEN` secret in Colab. Never commit authentication tokens.
6. Run each notebook and inspect `split_manifest.csv`, `fair_4way_summary.csv`, `fair_4way_itemwise.csv`, `fair_4way_paired_statistics.csv`, and `fair_4way_protocol.json` in its output directory.
7. Report the actual model revision, CUDA/GPU information, dependency versions and computed row counts alongside the results. Check the reported results against the manuscript before asserting exact reproduction.

## Dependencies

`requirements.txt` contains pinned direct-package versions collected from the author-provided Colab environment:

```text
torch==2.14.1+cu130
numpy==2.4.6
pandas==2.2.3
transformers==5.22.1
accelerate==1.15.0
peft==0.21.2
bitsandbytes==0.50.2
sentence-transformers==5.7.0
bert-score==0.3.13
scikit-learn==1.6.1
scipy==1.16.3
pytorch-crf==0.7.2
```

**Installation caveat:** `torch==2.14.1+cu130` is a CUDA-specific build and may require the corresponding PyTorch wheel index; the file is not a universal cross-platform lockfile. This version list has not been independently validated by rerunning both end-to-end notebooks and does not pin all transitive dependencies.

## License, data and model artifacts

The repository's original-author **code** is distributed under the MIT License, subject to the terms in [`LICENSE`](LICENSE). Rights to third-party model weights, tokenizers and packages remain subject to their respective licenses. The QA data and FEMSeg checkpoint are distinct research artifacts; verify and state their distribution conditions separately rather than assuming the MIT software license automatically covers them.

If a dataset or checkpoint is unavailable publicly, state that clearly and provide an approved access method (if any). Do not suggest that code-only files suffice for full numerical reproduction.

## Citation

Archived repository DOI: https://doi.org/10.5281/zenodo.24230791. Cite the specific archived version if using its code. Check the Zenodo record metadata and contents before making an archival-completeness claim.

## Reproducibility limits

The source notebooks document model configurations, preprocessing and evaluation, but the original exact corpus-generation logs are not available. The supplied software versions were captured after the original experiments. No independent end-to-end rerun or verification of the public Zenodo archive is claimed in this README.
