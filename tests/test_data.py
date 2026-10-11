import numpy as np
import pandas as pd

from cardiofed import config as C
from cardiofed.data import clean, load_raw
from cardiofed.evaluation import train_test_indices
from cardiofed.features import FEATURES, api_to_row, build_features


def test_raw_files_have_expected_rows():
    raw = load_raw()
    counts = raw.groupby("hospital").size().to_dict()
    assert counts == {site: n for site, (_, n) in C.HOSPITALS.items()}


def test_cleaning_rules():
    df, log = clean(load_raw())
    assert len(df) == 918                                   # 2 exact duplicates removed
    assert not df.drop(columns=["hospital"]).duplicated().any()
    assert (df["chol"] != 0).all() and (df["trestbps"] != 0).all()   # zero codes -> NaN
    assert df.loc[df.hospital == "switzerland", "chol"].isna().all()
    for col, (lo, hi) in C.VALID_RANGES.items():
        s = df[col].dropna()
        assert s.between(lo, hi).all(), col
    # plausible outliers are kept, not dropped
    assert df["chol"].max() == 603 and (df["oldpeak"] < 0).sum() == 12
    # Budapest never graded severity, so it is excluded from the severity target
    assert df.loc[df.hospital == "hungarian", "severity_available"].eq(0).all()
    assert {s["step"] for s in log.steps} >= {"drop_exact_duplicates", "zero_to_missing_chol", "define_targets"}


def test_split_is_stratified_and_disjoint():
    df, _ = clean(load_raw())
    tr, te = train_test_indices(df)
    assert set(tr).isdisjoint(te) and len(tr) + len(te) == len(df)
    test = df.iloc[te]
    for site in C.HOSPITALS:  # ~20% of every hospital goes to test, with the same disease rate
        site_all, site_te = df[df.hospital == site], test[test.hospital == site]
        assert abs(len(site_te) / len(site_all) - 0.2) < 0.03
        assert abs(site_te[C.TARGET].mean() - site_all[C.TARGET].mean()) < 0.05


def test_features_fixed_order_and_unknown_categories():
    df = pd.DataFrame([{"age": 50, "sex": 1, "cp": 9, "thalach": 150}])  # cp=9 is not a valid code
    X = build_features(df)
    assert list(X.columns) == FEATURES
    assert pd.isna(X.loc[0, "cp"])
    assert np.isclose(X.loc[0, "hr_reserve_pct"], 150 / 170 * 100)
    assert list(X["slope"].cat.categories) == sorted(C.CATEGORY_CODES["slope"].values())


def test_api_vocabulary_translation():
    row = api_to_row({"sex": "female", "cp": "asymptomatic", "thal": None, "age": 40})
    assert row == {"sex": 0, "cp": 4, "thal": None, "age": 40}
