# RamanGPT

**Raman spectrum &harr; crystal structure** with machine learning, in the
[AtomGPT](https://github.com/atomgptlab) spectroscopy family (cf. IRGPT,
DiffractGPT). Two directions plus a consistency loop:

- **Inverse** (*Raman spectrum &rarr; structure*): a causal LLM is fine-tuned
  (LoRA) on text prompts &mdash; chemical formula + the main Raman peaks
  (cm&#8315;&sup1;) &mdash; to generate the crystal structure. Any base LLM works.
- **Forward** (*structure &rarr; Raman spectrum*): an [ALIGNN](https://github.com/atomgptlab/alignn)
  model predicts the 200-bin Raman spectrum (50&ndash;1000 cm&#8315;&sup1;).
- **Consistency loop**: inverse &rarr; relax with
  [ALIGNN-FF](https://github.com/atomgptlab/alignn) &rarr; forward, then rank
  candidates by spectral cosine similarity. Deployed at
  [atomgpt.org/raman](https://atomgpt.org/raman).

Data and the trained inverse model are on the Hugging Face Hub:
- Dataset: [`knc6/ramangpt_dft`](https://huggingface.co/datasets/knc6/ramangpt_dft) (5,099-material Computational Raman Database, Alpaca format)
- Model: [`knc6/ramangpt_dft_model`](https://huggingface.co/knc6/ramangpt_dft_model) (Mistral-7B LoRA adapter)

## Install
```bash
pip install -r requirements.txt       # torch, transformers, peft, jarvis-tools, ...
pip install unsloth                    # optional: ~2x faster train/infer
```

## Data
```bash
# use the published dataset
huggingface-cli download knc6/ramangpt_dft --repo-type dataset --local-dir data/raman_split
# or rebuild from a JARVIS-style id_prop.json (jid, atoms, 200-bin Raman target)
python scripts/build_raman_data.py     # -> data/raman_split/{train,val,test}.json
```
Each record is Alpaca-style (`instruction`, `input`, `id`, `output`); the `input`
carries the formula, an optional space-group hint, and the top-20 Raman peaks as
`<cm-1>(<rel intensity>)`.

## Train (any base LLM)
`scripts/train.py` fine-tunes a LoRA adapter; swap `--base` for any Hugging Face
causal LM, and `--backend unsloth` for ~2x speed.
```bash
python scripts/train.py --base unsloth/mistral-7b --data data/raman_split --output runs/mistral7b --epochs 2
python scripts/train.py --base unsloth/Qwen3-4B-Base --data data/raman_split --output runs/qwen3_4b
```
Recipe: LoRA r16/&alpha;16; lr 2e-4 linear; effective batch 8; 2 epochs; seed 3407.

## Evaluate / Inference
```bash
python scripts/eval_raman.py --base unsloth/mistral-7b-bnb-4bit \
  --adapter knc6/ramangpt_dft_model --data data/raman_split/test.json --out runs/eval.json
python scripts/infer.py --adapter knc6/ramangpt_dft_model \
  --formula MoS2 --peaks "383 cm-1(1.0), 408 cm-1(0.9)"
```

### Reported accuracy
- **Inverse** (508 held-out): reduced-formula consistency **90.7%**, lattice MAE
  **1.16&ndash;2.36 &Aring;**, vs **17.2%** for a formula-conditioned retrieval
  baseline given identical inputs.
- **Forward** (5,099-material CRD): **43.4%** of held-out materials reach an
  active-region cosine &ge; 0.354.

## Forward model (structure &rarr; Raman spectrum)
An [ALIGNN](https://github.com/atomgptlab/alignn) multi-output regressor
(`alignn_atomwise_pure`, `output_features=200`) maps a structure to the 200-bin
Raman spectrum (50&ndash;1000 cm&#8315;&sup1;). Train with `alignn`'s
`train_alignn.py` on the same JARVIS Raman `id_prop.json`, then predict:
```bash
python scripts/predict_forward.py --config results/config.json \
  --model results/best_model.pt --poscar POSCAR     # prints <cm-1>  <intensity>
```
```python
from scripts.predict_forward import load_model, predict
from jarvis.core.atoms import Atoms
net, cfg = load_model("results/config.json", "results/best_model.pt")
spectrum = predict(net, cfg, Atoms.from_poscar("POSCAR"))   # 200-d numpy array
```

## Extending
- **Other base LLMs:** change `--base` (and `--backend`).
- **Other spectroscopies:** the same pipeline builds IRGPT, UVVisGPT, DiffractGPT.

## Citation
RamanGPT (arXiv:2606.03764); built on JARVIS-DFT
(https://doi.org/10.1038/s41524-020-00440-1) and the AtomGPT inverse-design
framework.
