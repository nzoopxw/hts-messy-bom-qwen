"""
harvest_boms.py — collect engineer-written part descriptions + manufacturer part
numbers (MPNs) from open-source hardware repositories.

These descriptions are the "messy" inputs for the HTS dataset: they are written
by many different engineers, in many styles ("0.01R/1%/R1206", "100n 16V X7R",
"Conn_02x05_Odd_Even"), unlike the uniform distributor catalog strings used in
published HTS benchmarks. The MPN is what lets us attach a real US HTS label
later (label_parts.py looks each MPN up on Digi-Key / Mouser).

SOURCES PARSED
--------------
  * KiCad 6+ schematics   (*.kicad_sch)  — placed symbols and their properties
  * KiCad 4/5 schematics  (*.sch)        — $Comp blocks and their F-fields
  * BOM spreadsheets      (*bom*.csv/.tsv/.xlsx/.xls/.ods) — header auto-detect

USAGE
-----
    # 1. clone repos listed in repos.txt (one "owner/name" per line)
    python3 harvest_boms.py --clone repos.txt --repos-dir repos
    # 2. or just harvest repos you already have
    python3 harvest_boms.py --repos-dir repos

OUTPUT
------
  bom_candidates.csv — one row per (project, MPN, description) with columns:
    project, source_file, ref, value, description, spec, footprint, package,
    mpn, manufacturer, digikey_pn, mouser_pn, lcsc_pn, license, text
  `text` is the messy input string built only from what the engineer wrote
  (value, description, spec fields, package) — never the MPN field itself.
"""

import argparse
import csv
import glob
import os
import re
import subprocess
import sys
from collections import OrderedDict

# ── Field-name aliases (case-insensitive, spaces/underscores ignored) ─────────
def _k(s):
    return re.sub(r"[\s_\-\.]", "", str(s).lower())

ALIASES = {
    "mpn": ["mpn", "partnumber", "manufacturerpartnumber", "mfrpartnumber", "mfrpn",
            "mfgpn", "mfgpartnumber", "pn", "mfpn", "mfn", "manufacturerpn",
            "macrofabmpn", "manufacturerpartno", "partno", "mfrno", "ampn"],
    "manufacturer": ["manufacturer", "mfg", "mfr", "manufacturername", "amanufacturer", "vendor"],
    "digikey_pn": ["digikey", "digikeypn", "dkpn", "dpn", "sndk", "digikeypartnumber", "digikeypart"],
    "mouser_pn": ["mouser", "mouserpn", "mouserpartnumber"],
    "lcsc_pn": ["lcsc", "lcscpart", "lcscpartnumber", "jlc", "jlcpcb", "lcscpn"],
    "description": ["description", "desc", "details", "characteristics", "info", "notes", "note", "comment"],
    "value": ["value", "val", "designation"],
    "footprint": ["footprint", "package", "case", "pcbfootprint"],
    "ref": ["reference", "references", "ref", "designator", "designators", "refdes", "parts", "reference(s)"],
}
SPEC_FIELDS = ["tolerance", "voltage", "dielectric", "rating", "current", "color", "power", "maxcurr", "type"]
ALIAS_LOOKUP = {a: f for f, al in ALIASES.items() for a in al}

PLACEHOLDER = re.compile(r"^(~|-+|n/?a|none|nc|dnp|dnf|tbd|\?+|x|0|null|nan|generic|any)$", re.I)
GENERIC_DESC = re.compile(  # boilerplate KiCad library descriptions — not engineer-written
    r"^(resistor|unpolarized capacitor|polarized capacitor|capacitor|inductor|led|"
    r"light emitting diode|diode|ferrite bead|crystal|generic connector.*|"
    r"connector.*|test point|mounting hole.*|fiducial.*|power flag.*|"
    r"generic connector, single row.*|"
    r"(resistor|unpolarized capacitor|polarized capacitor|capacitor|inductor|diode|led)"
    r"(, small symbol| small| \(small\))?(, (polarized|unpolarized|small))*)$", re.I)


def clean(s):
    s = "" if s is None else str(s)
    s = s.replace("\\n", " ").replace("\n", " ").strip().strip('"').strip()
    return "" if PLACEHOLDER.match(s) else s


def short_package(fp):
    """'Resistor_SMD:R_0402_1005Metric' -> 'R_0402'; 'Package_QFP:LQFP-48_7x7mm_P0.5mm' -> 'LQFP-48'."""
    if not fp:
        return ""
    fp = fp.split(":")[-1]
    fp = re.sub(r"_\d+Metric.*$", "", fp)
    fp = re.sub(r"_(Pad|HandSolder|ThermalVias|P\d|\d+(\.\d+)?x\d+(\.\d+)?mm).*$", "", fp)
    return fp[:30]


# ── KiCad 6+ (.kicad_sch) — minimal s-expression parser ──────────────────────
TOKEN = re.compile(r'\(|\)|"(?:[^"\\]|\\.)*"|[^\s()"]+')


def parse_sexpr(text):
    stack, cur = [], []
    for m in TOKEN.finditer(text):
        t = m.group(0)
        if t == "(":
            stack.append(cur)
            cur = []
        elif t == ")":
            done = cur
            cur = stack.pop()
            cur.append(done)
        else:
            cur.append(t[1:-1].replace('\\"', '"') if t.startswith('"') else t)
    return cur


def kicad6_parts(path):
    try:
        tree = parse_sexpr(open(path, encoding="utf-8", errors="ignore").read())
    except Exception:
        return []
    root = tree[0] if tree and isinstance(tree[0], list) else []
    out = []
    for node in root:  # top-level only -> placed symbols, not lib_symbols definitions
        if not (isinstance(node, list) and node and node[0] == "symbol"):
            continue
        props = {}
        for sub in node:
            if isinstance(sub, list) and len(sub) >= 3 and sub[0] == "property":
                props[sub[1]] = sub[2]
        if props:
            out.append(props)
    return out


# ── KiCad 4/5 (.sch) ──────────────────────────────────────────────────────────
F_LINE = re.compile(r'^F\s+(\d+)\s+"((?:[^"\\]|\\.)*)"(?:.*?"([^"]+)")?\s*$')


def kicad5_parts(path):
    out, comp = [], None
    for line in open(path, encoding="utf-8", errors="ignore"):
        line = line.rstrip("\n")
        if line.startswith("$Comp"):
            comp = {}
        elif line.startswith("$EndComp"):
            if comp:
                out.append(comp)
            comp = None
        elif comp is not None and line.startswith("F "):
            m = F_LINE.match(line)
            if not m:
                continue
            idx, val, name = int(m.group(1)), m.group(2), m.group(3)
            std = {0: "Reference", 1: "Value", 2: "Footprint", 3: "Datasheet"}
            key = std.get(idx, name)
            if key:
                comp[key] = val
    return out


# ── BOM spreadsheets ──────────────────────────────────────────────────────────
def read_tables(path):
    import pandas as pd
    try:
        if path.lower().endswith((".csv", ".tsv", ".txt")):
            return [pd.read_csv(path, header=None, dtype=str, sep=None, engine="python",
                                on_bad_lines="skip", encoding_errors="ignore")]
        return list(pd.read_excel(path, header=None, dtype=str, sheet_name=None).values())
    except Exception:
        return []


def bom_parts(path):
    out = []
    for df in read_tables(path):
        hdr_row = None
        for i in range(min(25, len(df))):
            keys = [_k(x) for x in df.iloc[i].tolist()]
            if sum(1 for k in keys if k in ALIAS_LOOKUP) >= 2:
                hdr_row = i
                break
        if hdr_row is None:
            continue
        headers = [str(x).strip() for x in df.iloc[hdr_row].tolist()]
        for _, r in df.iloc[hdr_row + 1:].iterrows():
            props = {h: r.iloc[j] for j, h in enumerate(headers) if h and h != "nan"}
            out.append(props)
    return out


# ── Normalise one part's fields ───────────────────────────────────────────────
def normalise(props):
    rec = {f: "" for f in ALIASES}
    spec = []
    for name, val in props.items():
        v = clean(val)
        if not v:
            continue
        k = _k(name)
        field = ALIAS_LOOKUP.get(k)
        if field and not rec[field]:
            rec[field] = v
        elif k in SPEC_FIELDS:
            spec.append(v)
    rec["spec"] = " ".join(OrderedDict.fromkeys(spec))
    # several alternates in one cell ("PMEG2005EGWJ, PMEG3005EGWX") -> keep the first
    for f in ("mpn", "digikey_pn", "mouser_pn"):
        if rec[f]:
            rec[f] = re.split(r"\s*[,;/|]\s+|\s+or\s+", rec[f])[0].strip()
    return rec


def build_text(rec):
    """Messy input string: only engineer-written fields, never the MPN column."""
    desc = rec["description"]
    if GENERIC_DESC.match(desc or ""):
        desc = ""
    pieces = [rec["value"], desc, rec["spec"], rec["package"]]
    seen, words = set(), []
    for p in pieces:
        p = (p or "").strip()
        if p and p.lower() not in seen:
            seen.add(p.lower())
            words.append(p)
    return " ".join(words)[:120]


def repo_license(repo_dir):
    for f in sorted(glob.glob(os.path.join(repo_dir, "LICEN[CS]E*")) + glob.glob(os.path.join(repo_dir, "COPYING*"))):
        try:
            head = open(f, encoding="utf-8", errors="ignore").read(400)
        except Exception:
            continue
        for pat, name in [(r"CERN.*?OHL.*?S", "CERN-OHL-S"), (r"CERN.*?OHL.*?P", "CERN-OHL-P"),
                          (r"CERN.*?OHL.*?W", "CERN-OHL-W"), (r"CERN", "CERN-OHL"),
                          (r"Attribution-ShareAlike", "CC-BY-SA"), (r"Attribution", "CC-BY"),
                          (r"Apache", "Apache-2.0"), (r"MIT License|Permission is hereby granted", "MIT"),
                          (r"BSD", "BSD"), (r"GNU GENERAL PUBLIC", "GPL"), (r"Solderpad", "Solderpad")]:
            if re.search(pat, head, re.I):
                return name
        return os.path.basename(f)
    return "UNKNOWN"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos-dir", default="repos")
    ap.add_argument("--clone", help="text file with owner/name per line to shallow-clone first")
    ap.add_argument("--out", default="bom_candidates.csv")
    args = ap.parse_args()
    os.makedirs(args.repos_dir, exist_ok=True)

    if args.clone:
        for line in open(args.clone):
            name = line.split("#")[0].strip()
            if not name:
                continue
            dest = os.path.join(args.repos_dir, name.replace("/", "_"))
            if os.path.isdir(dest):
                continue
            r = subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:limit=5m",
                                f"https://github.com/{name}", dest],
                               capture_output=True, text=True, timeout=300)
            print(("ok   " if r.returncode == 0 else "FAIL ") + name)

    rows, stats = {}, []
    for repo in sorted(os.listdir(args.repos_dir)):
        rdir = os.path.join(args.repos_dir, repo)
        if not os.path.isdir(rdir):
            continue
        lic = repo_license(rdir)
        files = [f for f in glob.glob(os.path.join(rdir, "**", "*"), recursive=True)
                 if "/.git/" not in f and os.path.isfile(f)]
        n_parts = n_mpn = 0
        for f in files:
            fl = f.lower()
            if fl.endswith(".kicad_sch"):
                parts = kicad6_parts(f)
            elif fl.endswith(".sch"):
                parts = kicad5_parts(f)
            elif ("bom" in os.path.basename(fl) or "bill" in os.path.basename(fl)) and \
                    fl.endswith((".csv", ".tsv", ".xlsx", ".xls", ".ods")):
                parts = bom_parts(f)
            else:
                continue
            for props in parts:
                rec = normalise(props)
                ref = rec["ref"]
                if ref.startswith("#") or ref.upper().startswith(("TP", "MH", "FID", "H")) and not rec["mpn"]:
                    continue
                n_parts += 1
                if not rec["mpn"] and not rec["digikey_pn"] and not rec["mouser_pn"]:
                    continue
                n_mpn += 1
                rec["package"] = short_package(rec["footprint"])
                rec["text"] = build_text(rec)
                if not rec["text"]:
                    continue
                key = (repo, _k(rec["mpn"] or rec["digikey_pn"] or rec["mouser_pn"]), rec["text"].lower())
                if key in rows:
                    continue
                rec.update(project=repo, source_file=os.path.relpath(f, rdir), license=lic)
                rows[key] = rec
        stats.append((repo, lic, n_parts, n_mpn))

    cols = ["project", "license", "source_file", "ref", "text", "value", "description", "spec",
            "footprint", "package", "mpn", "manufacturer", "digikey_pn", "mouser_pn", "lcsc_pn"]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows.values():
            w.writerow(r)

    per = {}
    for (p, _, _), _r in rows.items():
        per[p] = per.get(p, 0) + 1
    print(f"\n{'project':<45}{'license':<14}{'parts':>7}{'w/ PN':>7}{'kept':>7}")
    for repo, lic, n, m in stats:
        print(f"{repo:<45}{lic:<14}{n:>7}{m:>7}{per.get(repo, 0):>7}")
    print(f"\nUnique (project, part, text) rows: {len(rows)}  "
          f"| unique MPNs: {len({k[1] for k in rows})}  -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
