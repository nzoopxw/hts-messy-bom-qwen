"""
build_dataset.py — turn parts_labeled.csv into train / val / test sets for
fine-tuning Qwen to map messy engineering descriptions -> US HTS codes.

DESIGN DECISIONS (worth stating in the report)
----------------------------------------------
1. SPLIT BY PROJECT, not by row. Every hardware project is written by different
   engineers in a different style, so test/val projects are styles the model has
   never seen. A random row split would leak each project's writing style.
2. NO PART-NUMBER LEAKAGE. A part in a val/test project whose MPN also appears in
   a training project is dropped from val/test (the model could otherwise
   memorise the part instead of reading the description).
3. CLEAN LABELS FOR EVALUATION. val/test use only 'gold' (both distributors
   agree) and 'single' labels; 'conflict' rows go to train only (or are dropped
   with --drop-conflicts).
4. DEDUPLICATION. Identical (text, code) pairs are kept once.
5. CLASS BALANCE. Common codes (chip resistors, ceramic caps) are capped per
   split so they don't swamp everything else (--cap-train / --cap-eval).
6. AUGMENTATION IS TRAIN-ONLY and flagged. Optional ERP-style variants of
   training rows: UPPERCASE, comma-separated, cut to 30 characters (like the
   061 sheet). val/test stay 100% real text.

USAGE
-----
    python3 build_dataset.py                          # defaults
    python3 build_dataset.py --augment 1              # +1 synthetic variant per train row
    python3 build_dataset.py --extra-train atlas.csv  # add e.g. CROSS/ATLAS rows to train only
                                                      # (csv with columns: text,hts)
OUTPUT (in ./dataset/)
----------------------
    train.jsonl / val.jsonl / test.jsonl   chat format for SFT:
        {"messages":[system,user(text),assistant("8533.21.0030")], ...metadata}
    train.csv / val.csv / test.csv         same rows, flat, for inspection/eval
    stats.md                               numbers for the report
"""

import argparse
import json
import os
import random
import re

import pandas as pd

SYSTEM_PROMPT = ("You are a US customs classification assistant. Given an engineering "
                 "bill-of-materials description of a part, reply with only its 10-digit "
                 "US HTS code in the form XXXX.XX.XXXX.")


def digits(s):
    return re.sub(r"\D", "", str(s or ""))


def fmt_hts(d):
    d = digits(d)
    return f"{d[:4]}.{d[4:6]}.{d[6:10]}" if len(d) == 10 else ""


def norm_text(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


def squash(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


# ── ERP-style augmentation (train only) ───────────────────────────────────────
ABBREV = [(r"\bresistor\b", "RES"), (r"\bcapacitor\b", "CAP"), (r"\binductor\b", "IND"),
          (r"\bconnector\b", "CONN"), (r"\bceramic\b", "CER"), (r"\bswitch\b", "SW"),
          (r"\bcrystal\b", "XTAL"), (r"\btransistor\b", "XSTR"), (r"\bregulator\b", "REG"),
          (r"\bmicrocontroller\b", "MCU"), (r"\boscillator\b", "OSC")]


def erp_variant(text, rng):
    t = text
    for pat, rep in ABBREV:
        if rng.random() < 0.7:
            t = re.sub(pat, rep, t, flags=re.I)
    style = rng.choice(["upper_comma", "upper_trunc", "comma_trunc"])
    toks = [x for x in re.split(r"[\s/]+", t) if x]
    if "comma" in style:
        t = ",".join(toks)
    if "upper" in style:
        t = t.upper()
    if "trunc" in style:
        t = t[:30]
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="parts_labeled.csv")
    ap.add_argument("--outdir", default="dataset")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--cap-train", type=int, default=80, help="max rows per HTS code in train")
    ap.add_argument("--cap-eval", type=int, default=20, help="max rows per HTS code in val/test")
    ap.add_argument("--augment", type=int, default=0, help="synthetic ERP-style variants per train row")
    ap.add_argument("--drop-conflicts", action="store_true")
    ap.add_argument("--extra-train", default=None, help="csv with text,hts to add to TRAIN only")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    try:
        df = pd.read_csv(args.inp, dtype=str).fillna("")
    except pd.errors.ParserError:
        df = pd.read_csv(args.inp, dtype=str, engine="python").fillna("")
    n0 = len(df)
    df["text"] = df["text"].map(norm_text)
    df["hts"] = df["hts"].map(fmt_hts)
    df = df[(df.hts != "") & (df.text != "")]
    if args.drop_conflicts:
        df = df[df.label_status != "conflict"]
    n_labeled = len(df)

    # flags used for analysis
    def copied(r):
        t = squash(r.text)
        return any(len(squash(d)) >= 12 and squash(d) in t for d in (r.get("dk_desc", ""), r.get("mouser_desc", "")))

    def pn_only(r):  # text carries nothing beyond the part number itself
        rest = squash(r.text).replace(squash(r.get("mpn", "")), "")
        return len(rest) < 4

    df["copied_from_distributor"] = df.apply(copied, axis=1)
    df["pn_only"] = df.apply(pn_only, axis=1)
    df["source"] = "real"

    # dedupe identical (text, code) — keep first occurrence
    df = df.drop_duplicates(subset=["text", "hts"]).reset_index(drop=True)
    n_dedup = len(df)

    # ── split by project ──────────────────────────────────────────────────────
    sizes = df.groupby("project").size().to_dict()
    projects = sorted(sizes)
    rng.shuffle(projects)
    total = sum(sizes.values())
    split_of, filled = {}, {"test": 0, "val": 0}
    targets = {"test": args.test_frac * total, "val": args.val_frac * total}
    for p in projects:
        # don't let one huge project eat an entire eval split
        for s in ("test", "val"):
            if filled[s] < targets[s] and sizes[p] <= max(targets[s] * 0.6, 30):
                split_of[p] = s
                filled[s] += sizes[p]
                break
        else:
            split_of[p] = "train"
    df["split"] = df.project.map(split_of)

    # ── leakage + label-quality rules for eval splits ─────────────────────────
    train_pns = set(df.loc[df.split == "train", "pn_key"])
    ev = df.split.isin(["val", "test"])
    leak = ev & df.pn_key.isin(train_pns)
    bad_label = ev & (df.label_status == "conflict")
    pn_only_eval = ev & df.pn_only
    # dropped (not moved to train) so no project ever appears in two splits
    df = df[~(leak | bad_label | pn_only_eval)]
    n_leak = int(leak.sum())
    n_eval_dropped = int((bad_label | pn_only_eval).sum())

    # ── per-code caps ─────────────────────────────────────────────────────────
    parts = []
    for s, c in (("train", args.cap_train), ("val", args.cap_eval), ("test", args.cap_eval)):
        sub = df[df.split == s].sample(frac=1, random_state=args.seed)   # shuffle, then keep first c per code
        parts.append(sub.groupby("hts").head(c))
    df = pd.concat(parts).reset_index(drop=True)

    # ── optional extra training data + augmentation ───────────────────────────
    extra = pd.DataFrame()
    if args.extra_train:
        extra = pd.read_csv(args.extra_train, dtype=str).fillna("")
        extra["text"] = extra["text"].map(norm_text)
        extra["hts"] = extra["hts"].map(fmt_hts)
        extra = extra[(extra.text != "") & (extra.hts != "")]
        extra = extra.assign(split="train", source="extra", project="extra", label_status="external")
    aug = []
    if args.augment:
        for _, r in df[df.split == "train"].iterrows():
            for _ in range(args.augment):
                v = erp_variant(r.text, rng)
                if v and v != r.text:
                    aug.append({**r.to_dict(), "text": v, "source": "augmented"})
    full = pd.concat([df, extra, pd.DataFrame(aug)], ignore_index=True).fillna("")

    # ── write files ───────────────────────────────────────────────────────────
    keep_cols = ["split", "source", "text", "hts", "label_status", "project", "license", "mpn",
                 "manufacturer", "copied_from_distributor", "dk_desc", "mouser_desc"]
    for s in ("train", "val", "test"):
        sub = full[full.split == s].sample(frac=1, random_state=args.seed)
        sub[[c for c in keep_cols if c in sub]].to_csv(os.path.join(args.outdir, f"{s}.csv"), index=False)
        with open(os.path.join(args.outdir, f"{s}.jsonl"), "w") as f:
            for _, r in sub.iterrows():
                f.write(json.dumps({
                    "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                 {"role": "user", "content": r.text},
                                 {"role": "assistant", "content": r.hts}],
                    "hts": r.hts, "source": r.source, "project": r.project,
                    "label_status": r.label_status,
                }, ensure_ascii=False) + "\n")

    # ── stats for the report ──────────────────────────────────────────────────
    real = full[full.source == "real"]
    train_codes = set(full.loc[full.split == "train", "hts"])
    train_h4 = {c[:4] for c in train_codes}
    lines = ["# Messy-BOM HTS dataset — build stats", "",
             f"- Candidate rows read: {n0}",
             f"- Rows with a 10-digit label: {n_labeled}",
             f"- After (text, code) dedup: {n_dedup}",
             f"- Eval rows dropped for part-number leakage into train: {n_leak}",
             f"- Eval rows dropped for conflicting labels or part-number-only text: {n_eval_dropped}",
             f"- Projects: {real.project.nunique()} "
             f"(train {real[real.split=='train'].project.nunique()}, "
             f"val {real[real.split=='val'].project.nunique()}, "
             f"test {real[real.split=='test'].project.nunique()})", "",
             "| split | rows | real | augmented | extra | distinct codes | headings (4-digit) "
             "| gold labels | copied from distributor | codes unseen in train |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for s in ("train", "val", "test"):
        sub = full[full.split == s]
        r_ = sub[sub.source == "real"]
        unseen = "" if s == "train" else (
            f"{(~sub.hts.isin(train_codes)).sum()} rows "
            f"({(~sub.hts.str[:4].isin(train_h4)).sum()} with unseen heading)")
        lines.append(
            f"| {s} | {len(sub)} | {len(r_)} | {(sub.source=='augmented').sum()} | "
            f"{(sub.source=='extra').sum()} | {sub.hts.nunique()} | {sub.hts.str[:4].nunique()} | "
            f"{(r_.label_status=='gold').sum()} | {r_.copied_from_distributor.astype(str).eq('True').sum()} | {unseen} |")
    lines += ["", "## Most common headings (real rows, all splits)", "",
              "| heading | rows |", "|---|---|"]
    for h, n in real.hts.str[:4].value_counts().head(20).items():
        lines.append(f"| {h} | {n} |")
    lines += ["", "## Example test rows", "", "| text | hts |", "|---|---|"]
    for _, r in full[full.split == "test"].head(12).iterrows():
        lines.append(f"| {r.text.replace('|', '/')} | {r.hts} |")
    with open(os.path.join(args.outdir, "stats.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:16]))
    print(f"\nWrote {args.outdir}/train|val|test .jsonl/.csv and stats.md")


if __name__ == "__main__":
    main()
