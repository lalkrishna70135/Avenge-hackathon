"""Per-call collection fields for the simulated bot, from IndiaMART's own category data.

Live VANI renders its prompt template with `quantity_unit_options` and `specification_options` for the
buyer's product. Without them the template hides its Quantity and Specification Collection blocks and the
bot never asks for either, so a prompt change to ask limits could not change anything. The simulation fills
them the same way:
  * the buyer's product is matched to an MCAT in data/catalog/ref_category_taxonomy.csv (token match),
  * that MCAT's two most important spec fields and their most common options come from
    data/catalog/ref_category_specs.csv (two, as the Call Quality Matrix scores Spec1 and Spec2),
  * units are not in the extract, so sarvam-105b-conversations names 2-3 units for the product (cached).
Both CSVs come from the hackathon's Seller Enrichment datasets.
"""
from __future__ import annotations

import json
import math
import re
import threading
from functools import lru_cache
from pathlib import Path

from . import sarvam

ROOT = Path(__file__).resolve().parent.parent
CATALOG_DIR = ROOT / "data" / "catalog"
TAXONOMY = CATALOG_DIR / "ref_category_taxonomy.csv"
SPECS = CATALOG_DIR / "ref_category_specs.csv"
INDEX = CATALOG_DIR / "spec_index.json"
SPECS_PER_CALL = 2
OPTIONS_PER_SPEC = 5
# fields a buyer is never asked about on a call
_NOT_BUYER_FACING = re.compile(r"item code|model (?:name|no|number)|country of origin|minimum order|production capacity|"
                               r"warranty|delivery time|packaging|price|brand", re.I)
_STOP = {"and", "of", "for", "the", "with", "in", "a", "an", "to", "on", "by", "or", "new", "best", "type", "other"}


def tokens(text: str) -> set[str]:
    out = set()
    for t in re.findall(r"[a-z0-9]+", (text or "").lower()):
        if t in _STOP or len(t) < 2:
            continue
        if len(t) > 4 and t.endswith("ies"):
            t = t[:-3] + "y"
        elif len(t) > 4 and t.endswith(("ches", "shes", "sses", "xes")):
            t = t[:-2]
        elif len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.add(t)
    return out


def build_index() -> dict:
    """MCATs that have spec data -> name, popularity (products filling its specs) and its top buyer-facing specs."""
    import pandas as pd
    tax = pd.read_csv(TAXONOMY, usecols=["mcat_id", "mcat_name"]).drop_duplicates("mcat_id")
    sp = pd.read_csv(SPECS, usecols=["mcat_id", "spec_name", "priority", "products_using", "top_options"])
    pop = sp.groupby("mcat_id")["products_using"].sum()
    sp = sp[sp["top_options"].notna() & ~sp["spec_name"].astype(str).str.contains(_NOT_BUYER_FACING)]
    sp = sp.sort_values(["mcat_id", "priority", "products_using"], ascending=[True, True, False])
    specs: dict[int, list] = {}
    for mid, g in sp.groupby("mcat_id"):
        rows = []
        for name, opts in zip(g["spec_name"], g["top_options"]):
            vals = list(dict.fromkeys(o.strip() for o in str(opts).split("|") if o.strip()))[:OPTIONS_PER_SPEC]
            if vals and name not in (r[0] for r in rows):
                rows.append([str(name), vals])
            if len(rows) == SPECS_PER_CALL:
                break
        if rows:
            specs[int(mid)] = rows
    names = dict(zip(tax["mcat_id"].astype(int), tax["mcat_name"].astype(str)))
    index = {str(m): {"name": names[m], "pop": int(pop.get(m, 0)), "specs": s} for m, s in specs.items() if m in names}
    sarvam.write_atomic(INDEX, json.dumps(index, ensure_ascii=False))
    return index


@lru_cache(maxsize=1)
def _index() -> tuple[dict, dict, dict]:
    if INDEX.exists():
        index = json.loads(INDEX.read_text())
    elif TAXONOMY.exists() and SPECS.exists():
        index = build_index()
    else:
        index = {}
    inverted: dict[str, list[str]] = {}
    for mid, m in index.items():
        m["tokens"] = tokens(m["name"])
        for t in m["tokens"]:
            inverted.setdefault(t, []).append(mid)
    n = max(len(index), 1)
    idf = {t: math.log(n / len(ms)) for t, ms in inverted.items()}
    return index, inverted, idf


def match(product: str) -> dict | None:
    """Best MCAT for a free-text product. Tokens are weighted by rarity (idf), so 'ultrasound' outweighs 'mini'.
    An MCAT whose every name token appears in the product wins (rarest, most specific match first, then the
    most popular); failing that, one that misses a single token but shares at least two rare ones."""
    index, inverted, idf = _index()
    p = tokens(product)
    best, best_key = None, None
    for mid in {m for t in p for m in inverted.get(t, ())}:
        m = index[mid]
        hit = m["tokens"] & p
        missing = len(m["tokens"]) - len(hit)
        if missing > 1 or (missing == 1 and len(hit) < 2):
            continue
        key = (missing == 0, round(sum(idf[t] for t in hit), 6), m["pop"])
        if best_key is None or key > best_key:
            best, best_key = mid, key
    if best is None:
        return None
    return {"mcat_id": int(best), "mcat": index[best]["name"], "specs": index[best]["specs"]}


UNITS_PROMPT = ("An IndiaMART buyer wants to buy: {product} (IndiaMART category: {mcat}). List the 2 or 3 units a "
                "seller of this would quote the quantity in, most common first, as a JSON array of short English "
                'unit names, for example ["Piece", "Box"]. Reply with the JSON array only.')


def units(product: str, mcat: str | None) -> list[str]:
    try:
        out = sarvam.chat([{"role": "user", "content": UNITS_PROMPT.format(product=product, mcat=mcat or product)}],
                          model=sarvam.FAST_MODEL, temperature=0.0, max_tokens=200, as_json=True)
    except (sarvam.EmptyReply, ValueError):     # unusable reply: the generic unit. Network errors propagate, so a
        return ["Piece"]                        # resumed run never gives one persona different fields in A and B
    out = [str(u).strip() for u in out if str(u).strip()] if isinstance(out, list) else []
    return out[:3] or ["Piece"]


_fields_lock = threading.Lock()


@lru_cache(maxsize=512)
def _fields(product: str) -> str:
    m = match(product)
    return json.dumps({"mcat_id": m and m["mcat_id"], "mcat": m and m["mcat"],
                       "quantity_unit_options": units(product, m and m["mcat"]),
                       "specification_options": dict((m or {}).get("specs") or [])}, ensure_ascii=False)


def collection_fields(product: str) -> dict:
    """{mcat_id, mcat, quantity_unit_options, specification_options} for one call. Computed once per product
    per process (the lock stops parallel calls from each asking Sarvam for units and getting different answers)."""
    with _fields_lock:
        return json.loads(_fields((product or "").strip() or "a product"))
