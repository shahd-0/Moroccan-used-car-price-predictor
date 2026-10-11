"""Shared data cleaning for the Moroccan used car price predictor.

The notebook follows the same rules step by step; train.py and predict.py use
these functions so the model is always trained and queried consistently.
"""
import re

import numpy as np
import pandas as pd

# ---- Cleaning rules (change these to tune the cleaning) -------------------
MIN_PRICE = 10_000        # MAD. Below this are mostly placeholders (1, 100, 1000 ...)
MAX_PRICE = 1_500_000     # MAD. Above this are mostly typing mistakes (1111111111 ...)
VALID_FUELS = {"Diesel", "Petrol", "Hybrid", "Electrique", "Lpg"}

# Arabic city names found in the Location column -> the spelling used elsewhere
ARABIC_CITIES = {
    "أكادير": "Agadir", "الجديدة": "El Jadida", "الدار البيضاء": "Casablanca",
    "الرباط": "Rabat", "القنيطرة": "Kénitra", "المحمدية": "Mohammedia",
    "الناظور": "Nador", "تطوان": "Tétouan", "تمارة": "Temara", "طنجة": "Tanger",
    "فاس": "Fès", "مراكش": "Marrakech", "مكناس": "Meknès", "وجدة": "Oujda",
}

NUMERIC_FEATURES = ["Year", "Mileage", "Fiscal Power"]
CATEGORICAL_FEATURES = [
    "Brand", "Model", "Fuel", "Gearbox", "Location",
    "Condition", "Origin", "First Owner", "Number of Doors",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Price"
UNKNOWN = "Unknown"


def mileage_to_midpoint(value):
    """'200 000 - 249 999' -> 224999.5. Anything that is not a range -> NaN."""
    if pd.isna(value):
        return np.nan
    value = str(value).replace(" ", "")
    if "-" not in value:
        return np.nan
    start, end = value.split("-")
    try:
        return (int(start) + int(end)) / 2
    except ValueError:
        return np.nan


def clean_listings(df: pd.DataFrame) -> pd.DataFrame:
    """Return the cleaned training table (features + Price)."""
    d = df.copy()

    # 1. Price is the target: rows without it cannot be used
    d = d.dropna(subset=[TARGET])

    # 2. Numeric columns
    d["Year"] = pd.to_numeric(d["Year"], errors="coerce")
    d["Fiscal Power"] = pd.to_numeric(
        d["Fiscal Power"].str.replace(" CV", "", regex=False), errors="coerce"
    )
    d["Mileage"] = d["Mileage"].apply(mileage_to_midpoint)
    d = d.dropna(subset=["Year", "Fiscal Power", "Mileage"])
    d[["Year", "Fiscal Power", "Mileage"]] = d[["Year", "Fiscal Power", "Mileage"]].astype(int)

    # 3. Text columns: trim and Title Case, then fix Arabic city names
    for col in ["Brand", "Model", "Fuel", "Gearbox", "Location"]:
        d[col] = d[col].astype(str).str.strip().str.title()
    d["Location"] = d["Location"].replace(ARABIC_CITIES)

    # 4. Remove invalid categories: numeric-only brands (unusable ids) and
    #    a few shifted rows where Fuel holds a mileage range
    d = d[~d["Brand"].str.isnumeric()]
    d = d[d["Fuel"].isin(VALID_FUELS)]

    # 5. Price outliers: out-of-range values and repeated-digit placeholders (99999, 1111111)
    d = d[(d[TARGET] >= MIN_PRICE) & (d[TARGET] <= MAX_PRICE)]
    placeholder = d[TARGET].astype(int).astype(str).str.fullmatch(r"(\d)\1{3,}")
    d = d[~placeholder]

    # 6. Duplicate listings
    d = d.drop_duplicates()

    # 7. Optional columns: keep the row and mark the value as unknown
    for col in ["Condition", "Origin", "First Owner"]:
        d[col] = d[col].fillna(UNKNOWN)
    d["Number of Doors"] = d["Number of Doors"].map(
        lambda v: UNKNOWN if pd.isna(v) else str(int(v))
    )

    return d[FEATURES + [TARGET]].reset_index(drop=True)
