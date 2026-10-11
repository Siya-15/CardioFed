"""Download, load and clean the UCI Heart Disease data (4 hospitals).

    python -m cardiofed.data download   # fetch the 4 raw files from UCI (already in data/raw/)
    python -m cardiofed.data clean      # raw -> data/processed/heart_clean.csv + cleaning report

Every cleaning rule is a small, named step that records what it changed, so the report
(reports/data_cleaning.json) doubles as documentation of the decisions.
"""
from __future__ import annotations

import argparse
import io
import json
import urllib.request
import zipfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cardiofed import config as C


# ------------------------------------------------------------------ download
def download(force: bool = False) -> None:
    """Fetch processed.<hospital>.data from UCI. Falls back to the dataset zip."""
    C.DATA_RAW.mkdir(parents=True, exist_ok=True)
    for site, (_, expected) in C.HOSPITALS.items():
        path = C.DATA_RAW / f"processed.{site}.data"
        if path.exists() and not force:
            print(f"[download] {path.name} already present")
            continue
        try:
            raw = urllib.request.urlopen(f"{C.UCI_BASE_URL}processed.{site}.data", timeout=30).read()
        except Exception as exc:  # noqa: BLE001 - try the zip mirror on any network error
            print(f"[download] direct file failed ({exc}); trying dataset zip")
            with zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(C.UCI_ZIP_URL, timeout=60).read())) as z:
                raw = z.read(f"processed.{site}.data")
        path.write_bytes(raw)
        rows = sum(1 for line in raw.decode().splitlines() if line.strip())
        status = "OK" if rows == expected else f"WARNING: expected {expected}"
        print(f"[download] {path.name}: {rows} rows ({status})")


# ------------------------------------------------------------------ load
def load_raw() -> pd.DataFrame:
    """All 4 hospitals in one frame. '?' is UCI's missing-value marker."""
    frames = []
    for site in C.HOSPITALS:
        path = C.DATA_RAW / f"processed.{site}.data"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run: python -m cardiofed.data download")
        df = pd.read_csv(path, header=None, names=C.RAW_COLUMNS, na_values="?")
        df.insert(0, "hospital", site)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    return df.apply(lambda s: pd.to_numeric(s, errors="coerce") if s.name != "hospital" else s)


# ------------------------------------------------------------------ clean
@dataclass
class CleaningLog:
    steps: list[dict] = field(default_factory=list)

    def add(self, step: str, why: str, **counts):
        self.steps.append({"step": step, "why": why, **{k: _jsonable(v) for k, v in counts.items()}})


def _jsonable(v):
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, dict): return {str(k): _jsonable(x) for k, x in v.items()}
    return v


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningLog]:
    log = CleaningLog()
    df = df.copy()
    n0 = len(df)

    # 1. Exact duplicates: the same 14 values at the same hospital. Kept once, otherwise a
    #    copy can land in train and test and inflate scores.
    dup = df.duplicated()
    dup_sites = df.loc[dup, "hospital"].value_counts().to_dict()
    df = df[~dup].reset_index(drop=True)
    log.add("drop_exact_duplicates", "identical records would leak between train and test",
            rows_removed=int(dup.sum()), by_hospital=dup_sites)

    # 2. Zero used as a missing-value code. Cholesterol 0 mg/dl and blood pressure 0 mmHg are
    #    impossible in a living patient; Switzerland recorded *every* cholesterol as 0.
    for col in ["chol", "trestbps"]:
        zeros = df[col] == 0
        by_site = df.loc[zeros, "hospital"].value_counts().to_dict()
        df.loc[zeros, col] = np.nan
        log.add(f"zero_to_missing_{col}", f"{col} = 0 is physiologically impossible (missing-value code)",
                values_changed=int(zeros.sum()), by_hospital=by_site)

    # 3. Remaining impossible values -> missing (none expected after step 2, but enforced so
    #    a re-download or new site can't sneak garbage in).
    for col, (lo, hi) in C.VALID_RANGES.items():
        bad = df[col].notna() & ~df[col].between(lo, hi)
        df.loc[bad, col] = np.nan
        log.add(f"range_check_{col}", f"valid range {lo}..{hi}", values_changed=int(bad.sum()))

    # 4. Outliers that ARE possible are kept. IQR flags are reported (EDA) but the rows stay:
    #    e.g. cholesterol 603 mg/dl is extreme but real (familial hypercholesterolaemia).
    #    Tree models are insensitive to them; the linear baseline winsorises inside its own
    #    pipeline using training-fold bounds only.
    flags = {}
    for col in ["age", "trestbps", "chol", "thalach", "oldpeak"]:
        s = df[col].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        flags[col] = {"iqr_low": round(float(lo), 2), "iqr_high": round(float(hi), 2),
                      "n_outside": int((~s.between(lo, hi)).sum())}
    log.add("flag_iqr_outliers", "plausible extremes are real patients - flagged, not removed", columns=flags)

    neg = int((df["oldpeak"] < 0).sum())
    log.add("keep_negative_oldpeak", "negative ST depression = ST elevation; rare but clinically valid",
            rows=neg, by_hospital=df.loc[df["oldpeak"] < 0, "hospital"].value_counts().to_dict())

    # 5. Targets.
    df[C.TARGET] = (df["num"] > 0).astype(int)
    # Hungary only coded presence (0/1), never grades 2-4, so its grade-1 rows are not
    # comparable with the other sites. Severity is defined only where grades were recorded.
    graded = df["hospital"] != "hungarian"
    df["severity_available"] = graded.astype(int)
    log.add("define_targets", "has_disease = num > 0; severity only for sites that graded 1-4",
            positives=int(df[C.TARGET].sum()), negatives=int((1 - df[C.TARGET]).sum()),
            severity_rows=int((graded & (df["num"] > 0)).sum()))

    # 6. Columns measured almost only at one site. Kept in the clean file for analysis; the
    #    feature-set decision is made (and justified) in train.py.
    miss = df.groupby("hospital")[["ca", "thal", "slope"]].apply(lambda g: g.isna().mean().round(3))
    log.add("site_specific_missingness", "ca/thal recorded almost only at Cleveland",
            missing_share=miss.to_dict())

    log.add("summary", "rows in -> rows out", rows_in=n0, rows_out=len(df))
    return df, log


def load_clean() -> pd.DataFrame:
    path = C.DATA_PROCESSED / "heart_clean.csv"
    if not path.exists():
        main(["clean"])
    return pd.read_csv(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["download", "clean"])
    ap.add_argument("--force", action="store_true", help="re-download even if files exist")
    args = ap.parse_args(argv)
    if args.command == "download":
        download(force=args.force)
        return
    df, log = clean(load_raw())
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.DATA_PROCESSED / "heart_clean.csv", index=False)
    (C.REPORTS / "data_cleaning.json").write_text(json.dumps(log.steps, indent=2))
    for s in log.steps:
        extra = {k: v for k, v in s.items() if k not in ("step", "why", "columns", "missing_share")}
        print(f"[clean] {s['step']:<32} {extra}")
    print(f"[clean] wrote {C.DATA_PROCESSED / 'heart_clean.csv'} ({len(df)} rows)")


if __name__ == "__main__":
    main()
