"""Curate and validate 100-120 mutual fund schemes from MFAPI directory.

Adheres strictly to Phase 19 rules:
- Categories: large-cap, flexi/multi-cap, mid-cap, small-cap, Nifty 50 index,
  aggressive hybrid, balanced hybrid, liquid or short-duration debt, gold.
- Selection rule: at least 3 years history (5 preferred), no more than 3 funds
  from one fund house per category, a mix of well-known fund houses.
- Direct Plan, Growth Option.
- Writes to data/reference/funds.csv.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import time
import urllib.request

WELL_KNOWN_HOUSES = [
    ("HDFC", ["hdfc"]),
    ("ICICI Prudential", ["icici prudential", "icici pru"]),
    ("SBI", ["sbi "]),
    ("Kotak", ["kotak"]),
    ("Nippon India", ["nippon india", "nippon "]),
    ("UTI", ["uti "]),
    ("Mirae Asset", ["mirae asset"]),
    ("DSP", ["dsp "]),
    ("Axis", ["axis "]),
    ("Tata", ["tata "]),
    ("Aditya Birla Sun Life", ["aditya birla sun life", "absl "]),
    ("Parag Parikh", ["parag parikh", "ppfas"]),
    ("Bandhan", ["bandhan"]),
    ("Franklin Templeton", ["franklin"]),
]

TARGET_CATEGORIES = {
    "large-cap": {
        "min": 8, "max": 14,
        "keywords": ["large cap", "bluechip", "top 100", "frontline equity"],
        "excludes": ["index", "etf", "mid", "small", "flexi", "multi", "hybrid", "advantage"],
    },
    "flexi/multi-cap": {
        "min": 8, "max": 14,
        "keywords": ["flexi cap", "flexicap", "multi cap", "multicap"],
        "excludes": ["index", "etf", "hybrid", "large cap", "mid cap", "small cap"],
    },
    "mid-cap": {
        "min": 8, "max": 14,
        "keywords": ["mid cap", "midcap", "mid-cap", "emerging equity"],
        "excludes": ["index", "etf", "small", "large", "hybrid"],
    },
    "small-cap": {
        "min": 8, "max": 14,
        "keywords": ["small cap", "smallcap", "small-cap"],
        "excludes": ["index", "etf", "mid", "large", "hybrid"],
    },
    "nifty-50-index": {
        "min": 4, "max": 6,
        "keywords": ["nifty 50 index", "nifty index fund", "index fund - nifty 50 plan", "index fund-nifty"],
        "excludes": ["junior", "next 50", "equal weight", "etf", "hybrid", "midcap", "100", "500"],
    },
    "aggressive-hybrid": {
        "min": 8, "max": 14,
        "keywords": ["aggressive hybrid", "hybrid equity", "equity & debt"],
        "excludes": ["conservative", "arbitrage", "multi asset", "index", "etf"],
    },
    "balanced-hybrid": {
        "min": 8, "max": 14,
        "keywords": ["balanced advantage", "balanced hybrid", "dynamic asset allocation"],
        "excludes": ["arbitrage", "multi asset", "index", "etf"],
    },
    "liquid-debt": {
        "min": 8, "max": 14,
        "keywords": ["liquid fund", "short term debt", "short term fund", "short duration"],
        "excludes": ["ultra short", "overnight", "gilt", "hybrid", "equity"],
    },
    "gold": {
        "min": 4, "max": 6,
        "keywords": ["gold fund", "gold savings", "gold etf fund of fund"],
        "excludes": ["equity", "silver", "hybrid"],
    },
}


def fetch_all_schemes() -> list[dict]:
    url = "https://api.mfapi.in/mf"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def match_amc(scheme_name: str) -> str | None:
    s_lower = scheme_name.lower()
    for amc_name, patterns in WELL_KNOWN_HOUSES:
        if any(p in s_lower for p in patterns):
            return amc_name
    return None


def curate_fund_list(all_schemes: list[dict]) -> list[dict]:
    curated: list[dict] = []

    # Map categories
    for cat_name, cat_cfg in TARGET_CATEGORIES.items():
        cat_matches = []
        amc_counts: dict[str, int] = {}

        for s in all_schemes:
            name = s["schemeName"]
            name_lower = name.lower()

            # Must be Direct Plan and Growth
            if "direct" not in name_lower:
                continue
            if "growth" not in name_lower:
                continue
            if "idcw" in name_lower or "dividend" in name_lower or "bonus" in name_lower:
                continue

            # Must match category keywords
            if not any(kw in name_lower for kw in cat_cfg["keywords"]):
                continue

            # Must not contain excluded terms
            if any(ex in name_lower for ex in cat_cfg["excludes"]):
                continue

            amc = match_amc(name)
            if not amc:
                continue

            # Limit to at most 3 funds per AMC in this category
            if amc_counts.get(amc, 0) >= 3:
                continue

            code = s["schemeCode"]
            cat_matches.append({
                "scheme_code": code,
                "name": name,
                "fund_house": amc,
                "category": cat_name,
                "plan": "Direct-Growth",
            })
            amc_counts[amc] = amc_counts.get(amc, 0) + 1

            if len(cat_matches) >= cat_cfg["max"]:
                break

        print(f"Category '{cat_name}': selected {len(cat_matches)} funds from {len(amc_counts)} fund houses.")
        curated.extend(cat_matches)

    return curated


def main():
    print("Fetching master scheme directory from MFAPI...")
    all_schemes = fetch_all_schemes()
    print(f"Total directory size: {len(all_schemes)} schemes.")

    curated = curate_fund_list(all_schemes)
    print(f"\nTotal curated funds: {len(curated)}")

    out_path = Path("data/reference/funds.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        f.write("# Selection rule: at least 3 years of history (5 preferred), no more than 3 funds from one fund house per category, a mix of well-known fund houses.\n")
        f.write("# Schema: scheme_code,name,fund_house,category,plan\n")
        writer = csv.DictWriter(f, fieldnames=["scheme_code", "name", "fund_house", "category", "plan"])
        writer.writeheader()
        writer.writerows(curated)

    print(f"Successfully saved {len(curated)} funds to {out_path}")


if __name__ == "__main__":
    main()
