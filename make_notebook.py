import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Fine-tuning Qwen2.5 to classify messy engineering BOM text into US HTS codes

**Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**.

When the upload box appears (cell 2), select `train.csv`, `val.csv` and `test.csv` from your `dataset/` folder.

What this notebook does:
1. **Baseline 1: TF-IDF nearest neighbour.** Copies the code of the most similar training description. Cheap retrieval baseline.
2. **Baseline 2: base Qwen, zero-shot.** The same model before fine-tuning, with the same prompt.
3. **QLoRA fine-tuning** of Qwen on `train.csv`.
4. **Fine-tuned Qwen** on the same test set.
5. Scores every method at **2 / 4 / 6 / 8 / 10 digits**, separately for truly messy rows and rows copied from distributor catalogues, then saves `results.json` and `predictions.csv`.

Runtime on a free T4: about 40 min (24 min main training, 12 min ablation).""")

code("""# 1. Install (about 1 min)
!pip -q install -U "transformers>=4.44" "peft>=0.12" "bitsandbytes>=0.43" accelerate\n!pip -q install "pandas==2.2.3"   # keep Colab\'s own pandas version\n# If pandas changed version: Runtime -> Restart session, then Run all again.""")

code("""# 2. Config + data upload
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
EPOCHS     = 3
LR         = 2e-4
BATCH      = 16
LORA_R     = 16
SEED       = 42

import os
need = [f for f in ("train.csv", "val.csv", "test.csv") if not os.path.exists(f)]
if need:
    from google.colab import files
    print("Upload:", need)
    files.upload()
print({f: os.path.exists(f) for f in ("train.csv", "val.csv", "test.csv")})""")

code("""# 3. Load data + scoring helpers
import re, json, time, random
import numpy as np, pandas as pd, torch

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
tr = pd.read_csv("train.csv", dtype=str).fillna("")
va = pd.read_csv("val.csv",   dtype=str).fillna("")
te = pd.read_csv("test.csv",  dtype=str).fillna("")
for d in (tr, va, te):
    d["gold"] = d["hts"].str.replace(r"\\D", "", regex=True)
    d["messy"] = d.get("copied_from_distributor", pd.Series([""] * len(d))).astype(str).str.lower() != "true"
print(f"train {len(tr)} | val {len(va)} | test {len(te)}  (test messy {te.messy.sum()}, copied {(~te.messy).sum()})")
print("distinct codes  train", tr.gold.nunique(), "| test", te.gold.nunique(),
      "| test codes unseen in train:", (~te.gold.isin(set(tr.gold))).sum(), "rows")
ALL_CODES = set(tr.gold) | set(va.gold) | set(te.gold)

def to_code(out):
    m = re.search(r"\\d{4}\\.?\\d{2}\\.?\\d{2,4}", str(out))
    return re.sub(r"\\D", "", m.group(0)) if m else re.sub(r"\\D", "", str(out))[:10]

DEPTHS = (2, 4, 6, 8, 10)
def score(preds, df):
    preds = [to_code(p) for p in preds]
    rows = {}
    for name, mask in (("all", np.ones(len(df), bool)), ("messy", df.messy.values), ("copied", ~df.messy.values)):
        g = df.gold.values[mask]; p = np.array(preds, dtype=object)[mask]
        if len(g) == 0:
            continue
        r = {f"acc@{k}": round(100 * np.mean([len(pp) >= k and pp[:k] == gg[:k] for pp, gg in zip(p, g)]), 1) for k in DEPTHS}
        r["well_formed_%"] = round(100 * np.mean([len(pp) == 10 for pp in p]), 1)
        r["in_label_set_%"] = round(100 * np.mean([pp in ALL_CODES for pp in p]), 1)
        r["n"] = int(len(g))
        rows[name] = r
    return preds, rows

RESULTS, PREDS = {}, {"text": te.text.tolist(), "gold": te.gold.tolist(), "messy": te.messy.tolist()}
def report(method, preds):
    p, rows = score(preds, te)
    RESULTS[method] = rows; PREDS[method] = p
    print(method); print(pd.DataFrame(rows).T.to_markdown())""")

code("""# 4. Baseline 1: TF-IDF character n-gram nearest neighbour (seconds, CPU)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=True, sublinear_tf=True)
Xtr = vec.fit_transform(tr.text)
nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(Xtr)
_, idx = nn.kneighbors(vec.transform(te.text))
report("tfidf_knn", tr.gold.values[idx[:, 0]].tolist())""")

code("""# 5. Load Qwen in 4-bit
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
BF16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8   # T4 -> False -> fp16
DTYPE = torch.bfloat16 if BF16 else torch.float16
print("GPU:", torch.cuda.get_device_name(0), "| compute dtype:", DTYPE)

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", torch_dtype=DTYPE,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                           bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=DTYPE))

SYS = ("You are a US customs classification assistant. Given an engineering bill-of-materials "
       "description of a part, reply with only its 10-digit US HTS code in the form XXXX.XX.XXXX.")
def prompt(text):
    return tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": text}],
                                   tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def predict(texts, bs=32):
    model.eval(); tok.padding_side = "left"; outs = []
    for i in range(0, len(texts), bs):
        enc = tok([prompt(t) for t in texts[i:i + bs]], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        g = model.generate(**enc, max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outs""")

code("""# 6. Baseline 2: base Qwen, zero-shot
t0 = time.time()
base_raw = predict(te.text.tolist())
print(f"{time.time() - t0:.0f}s | sample outputs:", base_raw[:5])
report("qwen_zero_shot", base_raw)""")

code("""# 7. QLoRA setup + tokenised training data (loss only on the answer tokens)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = get_peft_model(model, LoraConfig(
    r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()

END = "<|im_end|>"
def encode(text, hts):
    p = tok(prompt(text), add_special_tokens=False)["input_ids"]
    a = tok(hts + END, add_special_tokens=False)["input_ids"]
    return {"input_ids": p + a, "labels": [-100] * len(p) + a}
train_data = [encode(t, h) for t, h in zip(tr.text, tr.hts)]
print("examples:", len(train_data), "| max tokens:", max(len(x["input_ids"]) for x in train_data))

def collate(batch):
    L = max(len(x["input_ids"]) for x in batch)
    ids  = [x["input_ids"] + [tok.pad_token_id] * (L - len(x["input_ids"])) for x in batch]
    labs = [x["labels"]    + [-100] * (L - len(x["labels"]))                 for x in batch]
    att  = [[1] * len(x["input_ids"]) + [0] * (L - len(x["input_ids"]))      for x in batch]
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labs), "attention_mask": torch.tensor(att)}""")

code("""# 8. Train
from transformers import Trainer, TrainingArguments
import math
STEPS = math.ceil(len(train_data) / BATCH) * EPOCHS
args = TrainingArguments(
    output_dir="qwen_hts_lora", num_train_epochs=EPOCHS, per_device_train_batch_size=BATCH,
    learning_rate=LR, lr_scheduler_type="cosine", warmup_steps=max(1, int(0.05 * STEPS)), weight_decay=0.0,
    fp16=not BF16, bf16=BF16, logging_steps=10, save_strategy="no", report_to="none",
    remove_unused_columns=False, seed=SEED, optim="paged_adamw_8bit")
trainer = Trainer(model=model, args=args, train_dataset=train_data, data_collator=collate)
t0 = time.time()
out = trainer.train()
TRAIN_MIN = (time.time() - t0) / 60
print(f"training done in {TRAIN_MIN:.1f} min | final train loss {out.training_loss:.4f}")
LOSS_LOG = [(h["step"], h["loss"]) for h in trainer.state.log_history if "loss" in h]
# If the loss shows nan, set LR = 1e-4 in cell 2 and run again.""")

code("""# 9. Validation check (sanity only; no tuning on test)
_, val_rows = score(predict(va.text.tolist()), va)
print(pd.DataFrame(val_rows).T.to_markdown())
RESULTS["_val_finetuned"] = val_rows""")

code("""# 10. Fine-tuned Qwen on the test set
ft_raw = predict(te.text.tolist())
report("qwen_finetuned", ft_raw)""")

code("""# 11. Results table + save everything
rows = []
for m in ("tfidf_knn", "qwen_zero_shot", "qwen_finetuned"):
    for subset, r in RESULTS[m].items():
        rows.append({"method": m, "subset": subset, **r})
table = pd.DataFrame(rows)
print(table.to_markdown(index=False))

meta = {"model": MODEL_NAME, "epochs": EPOCHS, "lr": LR, "batch": BATCH, "lora_r": LORA_R,
        "train_rows": len(tr), "val_rows": len(va), "test_rows": len(te),
        "train_minutes": round(TRAIN_MIN, 1), "gpu": torch.cuda.get_device_name(0),
        "loss_log": LOSS_LOG}
json.dump({"meta": meta, "results": RESULTS}, open("results.json", "w"), indent=2)
pd.DataFrame(PREDS).to_csv("predictions.csv", index=False)
table.to_csv("results_table.csv", index=False)
model.save_pretrained("qwen_hts_lora_adapter"); tok.save_pretrained("qwen_hts_lora_adapter")
!zip -qr qwen_hts_lora_adapter.zip qwen_hts_lora_adapter

from google.colab import files
for f in ("results.json", "results_table.csv", "predictions.csv", "qwen_hts_lora_adapter.zip"):
    files.download(f)
print("Send results.json + predictions.csv to Claude. Adapter saved as qwen_hts_lora_adapter.zip (optional download).")""")

code("""# 12. (Optional) loss curve figure for the report
import matplotlib.pyplot as plt
s, l = zip(*LOSS_LOG)
plt.figure(figsize=(6, 3.5)); plt.plot(s, l); plt.xlabel("step"); plt.ylabel("training loss")
plt.title(f"QLoRA fine-tuning, {MODEL_NAME.split('/')[-1]}"); plt.tight_layout()
plt.savefig("loss_curve.png", dpi=200); files.download("loss_curve.png")""")

md("""## Ablation: the same training without ERP-style augmentation

Retrains a fresh adapter on the real training rows only (no augmented copies), with identical settings, and scores it on the same test set. Then saves every method's per-row predictions to `predictions_all.csv` (used for the significance tests and error analysis in the report).""")

code("""# 13. Ablation: fresh adapter, real training rows only
model = model.unload()          # remove the trained adapter -> back to the 4-bit base model
model = get_peft_model(model, LoraConfig(
    r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
tr_real = tr[tr.source == "real"]
abl_data = [encode(t, h) for t, h in zip(tr_real.text, tr_real.hts)]
STEPS = math.ceil(len(abl_data) / BATCH) * EPOCHS
abl_args = TrainingArguments(
    output_dir="qwen_hts_lora_abl", num_train_epochs=EPOCHS, per_device_train_batch_size=BATCH,
    learning_rate=LR, lr_scheduler_type="cosine", warmup_steps=max(1, int(0.05 * STEPS)), weight_decay=0.0,
    fp16=not BF16, bf16=BF16, logging_steps=10, save_strategy="no", report_to="none",
    remove_unused_columns=False, seed=SEED, optim="paged_adamw_8bit")
print("ablation training rows:", len(abl_data), "| steps:", STEPS)
t0 = time.time()
abl_out = Trainer(model=model, args=abl_args, train_dataset=abl_data, data_collator=collate).train()
print(f"done in {(time.time() - t0) / 60:.1f} min | mean train loss {abl_out.training_loss:.4f}")
report("qwen_finetuned_no_aug", predict(te.text.tolist()))""")

code("""# 14. Save every method's per-row test predictions
pd.DataFrame(PREDS).to_csv("predictions_all.csv", index=False)
files.download("predictions_all.csv")""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
    "kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, "finetune_qwen_hts.ipynb")
print("ok")
