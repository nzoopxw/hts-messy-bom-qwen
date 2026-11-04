"""
label_parts.py — attach US HTS labels to the messy BOM rows by looking up each
manufacturer part number (MPN) on Mouser and Digi-Key.

Run this on YOUR machine (it needs your API keys in .env):
    MOUSER_API_KEY="your_key_here"
    DIGIKEY_CLIENT_ID="your_client_id_here"
    DIGIKEY_CLIENT_SECRET="your_client_secret_here"

STRATEGY (fits inside the free daily limits)
-------------------------------------------
1. Mouser part-number search, 10 MPNs per call. ~3,300 MPNs -> ~330 calls,
   well under Mouser's 1,000 calls/day.
2. Digi-Key product details, ONE part per call (1,000 calls/day). Used for:
     a) every MPN Mouser didn't find or didn't give a USHTS for, and
     b) a cross-check on as many Mouser-labelled parts as the budget allows,
        so we know which labels both distributors agree on ("gold").
   --dk-budget caps Digi-Key calls per run (default 900). Everything is cached
   in ./cache, so re-running tomorrow simply continues where it stopped.

USAGE
-----
    pip install requests python-dotenv pandas
    python3 label_parts.py                      # reads bom_candidates.csv
    python3 label_parts.py --dk-budget 0        # Mouser only
    python3 label_parts.py --limit 50           # smoke test on first 50 MPNs

OUTPUT
------
  parts_labeled.csv — every candidate row plus:
    mouser_hts, mouser_desc, mouser_category, dk_hts, dk_desc, dk_category,
    hts (final 10-digit label), label_status:
       gold       both distributors give the same 10-digit code
       single     only one distributor has a code
       conflict   both have codes and they differ (kept, but excluded from test)
       unlabeled  no code found
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
MOUSER_API_KEY = os.environ.get("MOUSER_API_KEY")
DK_ID = os.environ.get("DIGIKEY_CLIENT_ID")
DK_SECRET = os.environ.get("DIGIKEY_CLIENT_SECRET")

CACHE = "cache"
MOUSER_BATCH = 10
MOUSER_SLEEP = 2.2      # stay under 30 calls/min
DK_SLEEP = 0.5          # Digi-Key allows 120/min


def digits(s):
    return re.sub(r"\D", "", str(s or ""))


def norm(pn):
    return re.sub(r"[^A-Z0-9]", "", str(pn or "").upper())


def cache_get(kind, key):
    p = os.path.join(CACHE, f"{kind}_{hashlib.md5(key.encode()).hexdigest()[:16]}.json")
    if os.path.exists(p):
        with open(p) as f:
            return True, json.load(f)
    return False, None


def cache_put(kind, key, data):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"{kind}_{hashlib.md5(key.encode()).hexdigest()[:16]}.json")
    with open(p, "w") as f:
        json.dump(data, f)


# ── Mouser ────────────────────────────────────────────────────────────────────
EMPTY = {"SearchResults": {"Parts": []}}


def clean_pn(pn):
    """Mouser rejects some characters; keep printable ASCII, no '|', max 40 chars."""
    pn = re.sub(r"[^\x21-\x7e]", "", str(pn)).replace("|", "")
    return pn[:40]


def mouser_call(key):
    """One Mouser part-number call. Returns (status, data). Never prints the URL (it holds the key)."""
    url = "https://api.mouser.com/api/v1/search/partnumber"
    body = {"SearchByPartRequest": {"mouserPartNumber": key, "partSearchOptions": "Exact"}}
    for _ in range(4):
        time.sleep(MOUSER_SLEEP)
        try:
            r = requests.post(url, params={"apiKey": MOUSER_API_KEY}, json=body, timeout=60)
        except requests.RequestException as e:
            print(f"  Mouser network error ({type(e).__name__}), retrying")
            time.sleep(10)
            continue
        if r.status_code == 429:
            print("  Mouser rate limit — waiting 60s")
            time.sleep(60)
            continue
        if r.status_code != 200:
            return r.status_code, None
        data = r.json()
        if data.get("Errors"):
            msgs = " ".join(str(e.get("Message", e)) for e in data["Errors"])
            if re.search(r"(?i)quota|limit|exceed|maximum", msgs):
                return "quota", None
            return "error", None
        return 200, data
    return "retry_failed", None


def mouser_lookup(mpns):
    mpns = [p for p in (clean_pn(m) for m in mpns) if p]
    if not mpns:
        return EMPTY
    key = "|".join(mpns)
    hit, data = cache_get("mouserpn", key)
    if hit:
        return data
    status, data = mouser_call(key)
    if status == 200:
        cache_put("mouserpn", key, data)
        return data
    if status == "quota":
        raise RuntimeError("mouser_quota")
    # batch rejected (usually one bad part number): retry each part on its own
    if len(mpns) > 1:
        print(f"  batch rejected ({status}); retrying its {len(mpns)} parts one by one")
        parts = []
        for m in mpns:
            hit, one = cache_get("mouserpn", m)
            if not hit:
                st, one = mouser_call(m)
                if st == "quota":
                    raise RuntimeError("mouser_quota")
                one = one if st == 200 else EMPTY      # skip a part Mouser won't accept
                cache_put("mouserpn", m, one)
            parts += ((one or {}).get("SearchResults") or {}).get("Parts") or []
        data = {"SearchResults": {"Parts": parts}}
        cache_put("mouserpn", key, data)
        return data
    cache_put("mouserpn", key, EMPTY)
    return EMPTY


def mouser_ushts(part):
    for c in part.get("ProductCompliance") or []:
        if c.get("ComplianceName") == "USHTS":
            return c.get("ComplianceValue")
    return ""


# ── Digi-Key ──────────────────────────────────────────────────────────────────
class DigiKey:
    def __init__(self):
        self.token, self.exp = None, 0

    def _token(self):
        if self.token and time.time() < self.exp - 60:
            return self.token
        r = requests.post("https://api.digikey.com/v1/oauth2/token", data={
            "client_id": DK_ID, "client_secret": DK_SECRET,
            "grant_type": "client_credentials"}, timeout=30)
        r.raise_for_status()
        j = r.json()
        self.token, self.exp = j["access_token"], time.time() + int(j.get("expires_in", 600))
        return self.token

    def details(self, pn):
        """Returns (from_cache, product_dict_or_None)."""
        hit, data = cache_get("dkdetail", pn)
        if hit:
            return True, data
        url = ("https://api.digikey.com/products/v4/search/"
               f"{urllib.parse.quote(pn, safe='')}/productdetails")
        for _ in range(4):
            time.sleep(DK_SLEEP)
            r = requests.get(url, headers={
                "Authorization": f"Bearer {self._token()}",
                "X-DIGIKEY-Client-Id": DK_ID,
                "X-DIGIKEY-Locale-Site": "US",
                "X-DIGIKEY-Locale-Language": "en",
                "X-DIGIKEY-Locale-Currency": "USD"}, timeout=60)
            if r.status_code == 401:
                self.token = None
                continue
            if r.status_code == 429:
                print("  Digi-Key rate limit — stopping Digi-Key for this run")
                raise RuntimeError("dk_quota")
            if r.status_code in (400, 404):
                cache_put("dkdetail", pn, None)
                return False, None
            if r.status_code != 200:          # temporary server error: skip, retry next run
                print(f"  Digi-Key HTTP {r.status_code} for one part — skipped for now")
                return False, None
            prod = r.json().get("Product")
            cache_put("dkdetail", pn, prod)
            return False, prod
        return False, None


def dk_fields(prod):
    if not prod:
        return "", "", ""
    d = prod.get("Description") or {}
    desc = (d.get("ProductDescription") or "") if isinstance(d, dict) else str(d)
    cat = (prod.get("Category") or {}).get("Name") or ""
    hts = (prod.get("Classifications") or {}).get("HtsusCode") or ""
    return digits(hts), desc, cat


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="bom_candidates.csv")
    ap.add_argument("--out", default="parts_labeled.csv")
    ap.add_argument("--dk-budget", type=int, default=900, help="max NEW Digi-Key calls this run")
    ap.add_argument("--limit", type=int, default=None, help="only first N unique part numbers")
    args = ap.parse_args()

    missing = [k for k, v in {"MOUSER_API_KEY": MOUSER_API_KEY, "DIGIKEY_CLIENT_ID": DK_ID,
                              "DIGIKEY_CLIENT_SECRET": DK_SECRET}.items() if not v]
    if missing:
        sys.exit(f"Missing in .env: {', '.join(missing)}")

    df = pd.read_csv(args.inp, dtype=str).fillna("")
    df["lookup_pn"] = df["mpn"].where(df["mpn"] != "", df["digikey_pn"].where(df["digikey_pn"] != "", df["mouser_pn"]))
    df["pn_key"] = df["lookup_pn"].map(norm)
    pns = df.drop_duplicates("pn_key")[["pn_key", "lookup_pn", "mpn"]]
    pns = pns[pns.pn_key != ""]
    if args.limit:
        pns = pns.head(args.limit)
    print(f"{len(df)} rows, {len(pns)} unique part numbers")

    info = {k: {"mouser_hts": "", "mouser_desc": "", "mouser_category": "",
                "dk_hts": "", "dk_desc": "", "dk_category": ""} for k in pns.pn_key}

    # 1) Mouser, 10 per call (only real MPNs — Mouser can't resolve Digi-Key PNs)
    mpn_list = pns[pns.mpn != ""].lookup_pn.tolist()
    batches = [mpn_list[i:i + MOUSER_BATCH] for i in range(0, len(mpn_list), MOUSER_BATCH)]
    print(f"Mouser: {len(batches)} batches")
    for bi, b in enumerate(batches, 1):
        try:
            data = mouser_lookup(b)
        except RuntimeError:
            print(f"  Mouser daily limit reached at batch {bi}/{len(batches)} — "
                  "continuing with Digi-Key; re-run tomorrow to finish Mouser")
            break
        parts = ((data or {}).get("SearchResults") or {}).get("Parts") or []
        for p in parts:
            k = norm(p.get("ManufacturerPartNumber"))
            if k not in info:
                continue
            h = digits(mouser_ushts(p))
            if info[k]["mouser_hts"] and not h:
                continue   # keep the listing that has a code
            cat = p.get("Category") or ""
            desc = p.get("Description") or ""
            if desc.lower().startswith(cat.lower()):
                desc = desc[len(cat):].strip(" ,-")
            info[k].update(mouser_hts=h, mouser_desc=desc, mouser_category=cat)
        if bi % 25 == 0 or bi == len(batches):
            print(f"  {bi}/{len(batches)}")

    # 2) Digi-Key: misses first, then cross-check the rest, within budget
    dk = DigiKey()
    lookup = dict(zip(pns.pn_key, pns.lookup_pn))
    misses = [k for k in info if not info[k]["mouser_hts"]]
    checks = [k for k in info if info[k]["mouser_hts"]]
    order = misses + checks
    new_calls = 0
    print(f"Digi-Key: {len(misses)} Mouser misses + {len(checks)} cross-checks; budget {args.dk_budget}")
    try:
        for i, k in enumerate(order, 1):
            if new_calls >= args.dk_budget and not cache_get("dkdetail", lookup[k])[0]:
                continue   # skip uncached lookups once the budget is spent; cached ones still load
            from_cache, prod = dk.details(lookup[k])
            if not from_cache:
                new_calls += 1
            h, d, c = dk_fields(prod)
            info[k].update(dk_hts=h, dk_desc=d, dk_category=c)
            if i % 100 == 0:
                print(f"  {i}/{len(order)} (new calls {new_calls})")
    except RuntimeError:
        pass

    # 3) Final label per row
    lab = pd.DataFrame.from_dict(info, orient="index").reset_index().rename(columns={"index": "pn_key"})
    out = df.merge(lab, on="pn_key", how="left").fillna("")

    def decide(r):
        m = r.mouser_hts if len(r.mouser_hts) == 10 else ""
        d = r.dk_hts if len(r.dk_hts) == 10 else ""
        if m and d:
            return (m, "gold") if m == d else (d, "conflict")
        if m or d:
            return (m or d, "single")
        return ("", "unlabeled")

    out[["hts", "label_status"]] = out.apply(lambda r: pd.Series(decide(r)), axis=1)
    # strip stray line breaks / control characters so the CSV always parses cleanly
    for c in out.columns:
        out[c] = out[c].astype(str).str.replace(r"[\r\n\t\x00-\x1f]+", " ", regex=True).str.strip()
    out.to_csv(args.out, index=False)

    vc = out.drop_duplicates("pn_key").label_status.value_counts()
    print("\nPer unique part:", vc.to_dict())
    print(f"Rows with a label: {(out.hts != '').sum()} / {len(out)}  "
          f"| distinct 10-digit codes: {out.loc[out.hts != '', 'hts'].nunique()}")
    print(f"New Digi-Key calls this run: {new_calls}. Uncached Digi-Key lookups remaining: "
          f"{sum(1 for k in order if not cache_get('dkdetail', lookup[k])[0])} "
          "(re-run tomorrow to continue).")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
