# HTS classification of messy engineering BOM text with a fine-tuned Qwen2.5

Fine-tunes **Qwen2.5-3B-Instruct** with **QLoRA** to map short, engineer-written
bill-of-materials (BOM) descriptions such as `100n Bypass 16V/20% C_0402` or `600R L_0402`
to their **10-digit US Harmonized Tariff Schedule (HTSUS) code**.

Course project for CS3807 Deep Learning Laboratory, Shiv Nadar University Chennai.
Base paper: *ATLAS: Benchmarking and Adapting LLMs for Global Trade via Harmonized Tariff Code
Classification* (Yuvraj & Devarakonda, arXiv:2509.18400, 2025). ATLAS fine-tunes a 70B model on
clean customs-ruling text; this project uses real, messy BOM text, a 3B model on one free GPU,
and a project-wise test split.

## Repository contents

| File | Purpose |
|---|---|
| `repos.txt` | The 169 open-source hardware repositories that are harvested |
| `harvest_boms.py` | Clones the repos and extracts each part's description and manufacturer part number from KiCad schematics and BOM spreadsheets → `bom_candidates.csv` |
| `label_parts.py` | Looks up every part number on the Mouser and Digi-Key APIs and attaches the US HTS code (gold / single / conflict) → `parts_labeled.csv` |
| `build_dataset.py` | Cleans, de-duplicates and splits by project into train / val / test with leakage checks → `dataset/` |
| `finetune_qwen_hts.ipynb` | Google Colab notebook: baselines, QLoRA fine-tuning, evaluation at 2/4/6/8/10 digits |
| `demo_qwen_hts.ipynb` | Demo: dataset overview, results and significance tests from saved predictions, and live classification with the trained adapter |
| `dataset/` | Final train / validation / test splits (`.csv` and chat-format `.jsonl`) and `stats.md` |
| `results/run1/` | First training run: notebook with outputs, `results.json`, `predictions.csv`, `loss_curve.png` |
| `results/run2/` | Second training run, used for all analysis in the report: notebook with outputs, `predictions_all.csv` (per-row predictions of every method, including the ablation) and `results_table.csv` recomputed from it |
| `results/run3/` | Third training run with identical settings; the demo adapter comes from this run (the adapter itself is not committed, as it exceeds GitHub's file-size limit) |

## Results (237 test rows from 26 held-out projects)

| Method | @4 digits | @6 digits | @10 digits | Avg digits correct |
|---|---|---|---|---|
| TF-IDF nearest neighbour | 81.4 | 69.6 | 58.6 | 7.68 |
| Qwen2.5-3B, no fine-tuning | 1.7 | 0.0 | 0.0 | 0.35 |
| Qwen2.5-3B QLoRA, no augmentation | 89.0 | 76.4 | 59.1 | 8.06 |
| **Qwen2.5-3B QLoRA (final)** | **87.8** | **76.8** | **64.6** | **8.17** |

Numbers are from run 2. Fine-tuning beats the baseline significantly at 4 and 6 digits (exact McNemar, p = 0.024 and 0.019); augmentation improves exact 10-digit accuracy (p = 0.041). Across the three runs, exact 10-digit accuracy was 63.3%, 64.6% and 60.8%.

## Reproducing the results

### 1. Set up (Python 3.9+)

```bash
pip install requests python-dotenv pandas openpyxl odfpy
```

Create a `.env` file in the repository folder with your own API credentials
(free from mouser.com/api-search and developer.digikey.com). **Never commit this file.**

```
MOUSER_API_KEY="your_key_here"
DIGIKEY_CLIENT_ID="your_client_id_here"
DIGIKEY_CLIENT_SECRET="your_client_secret_here"
```

### 2. Build the dataset

```bash
python3 harvest_boms.py --clone repos.txt --repos-dir repos   # ~30 min, a few GB of disk
python3 label_parts.py                                         # re-run daily until Digi-Key reports 0 remaining
python3 build_dataset.py --augment 1                           # writes dataset/ and dataset/stats.md
```

API responses are cached in `cache/`, so every step can be safely re-run.

### 3. Fine-tune and evaluate

1. Open `finetune_qwen_hts.ipynb` in Google Colab and select a **T4 GPU** runtime.
2. **Runtime → Run all**, and upload `dataset/train.csv`, `val.csv` and `test.csv` when prompted.
3. The notebook writes `results.json`, `results_table.csv`, `predictions.csv` and `loss_curve.png`.

Takes about 30–40 minutes with Qwen2.5-3B on a free T4. Seed: 42.

## Method summary

- **Input:** the description text the engineer wrote (value, description, spec fields, package). The part number is used only to find the label and is never shown to the model.
- **Labels:** US HTS codes from Mouser and Digi-Key. *gold* = both agree; *single* = one source; *conflict* = excluded from evaluation.
- **Split:** by project, so test projects are writers never seen in training; test rows whose part number appears in training are removed.
- **Model:** Qwen2.5-3B-Instruct, 4-bit NF4 QLoRA, rank 16, alpha 32, lr 2e-4, 3 epochs, batch 16.
- **Baselines:** character n-gram TF-IDF nearest neighbour; zero-shot Qwen2.5-3B.
- **Metric:** accuracy at 2, 4, 6, 8 and 10 digits, reported separately for messy and catalogue-copied text.

## Data and licences

Descriptions come from public open-source hardware repositories (licences recorded per row in the
`license` column; some projects state none). HTS labels come from distributor product data accessed
through their official APIs. Raw API responses are not redistributed here.
