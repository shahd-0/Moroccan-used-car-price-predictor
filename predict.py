"""Terminal tool: estimate the fair market price of a used car in Morocco and
say whether the seller's price is underpriced, fair, or overpriced.

Usage (from the repo root, after running `python src/train.py`):
    python src/predict.py                # answers questions one by one
    python src/predict.py --brand Dacia --model Sandero --year 2015 --mileage 150000 \
        --fiscal-power 6 --fuel Diesel --gearbox Manual --location Casablanca --price 85000
"""
import argparse
import difflib
import sys
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.pkl"
UNKNOWN = "Unknown"
YEAR_RANGE = (1980, 2025)


def load_bundle():
    if not MODEL_PATH.exists():
        sys.exit("No trained model found. Run `python src/train.py` first.")
    return joblib.load(MODEL_PATH)


# ---------- input helpers ----------------------------------------------------
# A value given as a command-line flag that is invalid stops the program with an
# error message. A value typed at a prompt that is invalid is simply asked again.
def ask_number(label, given, low, high):
    while True:
        raw = given if given is not None else input(f"{label}: ")
        try:
            value = int(float(str(raw).replace(" ", "").replace(",", "")))
            if low <= value <= high:
                return value
            message = f"{label.lower()} must be between {low} and {high}."
        except ValueError:
            message = f"please enter a number for {label.lower()}."
        if given is not None:
            sys.exit(f"Error: {message}")
        given = None
        print(f"  ! {message}")


def ask_choice(label, given, options, batch, optional=False, allow_unseen=False):
    """Match typed text to the known values (case-insensitive, with suggestions)."""
    lookup = {o.lower(): o for o in options}
    if optional and given is None and batch:
        return UNKNOWN
    while True:
        suffix = " (Enter to skip)" if optional else ""
        raw = str(given if given is not None else input(f"{label}{suffix}: ")).strip()
        if raw == "" and optional:
            return UNKNOWN
        if raw.lower() in lookup:
            return lookup[raw.lower()]
        close = difflib.get_close_matches(raw.lower(), list(lookup), n=4, cutoff=0.6)
        hint = f" Did you mean: {', '.join(lookup[c] for c in close)}?" if close else ""
        if given is not None:                       # value came from a flag
            if allow_unseen:
                print(f"Warning: '{raw}' was not in the training data, so the estimate is less reliable.")
                return raw.title()
            sys.exit(f"Error: unknown {label.lower()} '{raw}'.{hint}")
        given = None
        print(f"  ! '{raw}' not found.{hint}")
        if allow_unseen and not close and \
                input("  Use it anyway (estimate will be less reliable)? [y/N] ").lower() == "y":
            return raw.title()


def collect_inputs(args, choices):
    required = [args.brand, args.model, args.year, args.mileage, args.fiscal_power,
                args.fuel, args.gearbox, args.location, args.price]
    batch = all(v is not None for v in required)     # everything given as flags: no questions
    if not batch:
        print("\nMoroccan used car price check. Answer the questions below.\n")

    brand = ask_choice("Brand", args.brand, choices["brands"], batch)
    model_options = choices["models_by_brand"].get(brand, [])
    model = ask_choice("Model", args.model, model_options, batch, allow_unseen=True)
    year = ask_number("Year", args.year, *YEAR_RANGE)
    mileage = ask_number("Mileage (km)", args.mileage, 0, 1_000_000)
    fiscal = ask_number("Fiscal power (CV)", args.fiscal_power, 1, 60)
    fuel = ask_choice("Fuel", args.fuel, choices["fuels"], batch)
    gearbox = ask_choice("Gearbox", args.gearbox, choices["gearboxes"], batch)
    location = ask_choice("City", args.location, choices["locations"], batch)
    condition = ask_choice("Condition", args.condition, choices["conditions"], batch, optional=True)
    origin = ask_choice("Origin", args.origin, choices["origins"], batch, optional=True)
    first_owner = ask_choice("First owner (Yes/No)", args.first_owner, ["Yes", "No"], batch, optional=True)
    doors = ask_choice("Number of doors (3/5)", args.doors, ["3", "5"], batch, optional=True)
    price = ask_number("Seller price (MAD)", args.price, 1, 100_000_000)

    row = pd.DataFrame([{
        "Year": year, "Mileage": mileage, "Fiscal Power": fiscal,
        "Brand": brand, "Model": model, "Fuel": fuel, "Gearbox": gearbox, "Location": location,
        "Condition": condition, "Origin": origin, "First Owner": first_owner, "Number of Doors": doors,
    }])
    return row, price


# ---------- prediction ---------------------------------------------------------
def verdict(price, low, high):
    if price < low:
        return "UNDERPRICED - cheaper than similar listings. Could be a good deal, or check for hidden problems."
    if price > high:
        return "OVERPRICED - more expensive than similar listings. Worth negotiating."
    return "FAIR - in line with similar listings."


def main():
    parser = argparse.ArgumentParser(description="Check whether a used car's price is fair.")
    for flag, kind in [("brand", str), ("model", str), ("year", str), ("mileage", str),
                       ("fiscal-power", str), ("fuel", str), ("gearbox", str), ("location", str),
                       ("condition", str), ("origin", str), ("first-owner", str), ("doors", str),
                       ("price", str)]:
        parser.add_argument(f"--{flag}", type=kind, default=None)  # stored as args.fiscal_power etc.
    args = parser.parse_args()

    bundle = load_bundle()
    row, seller_price = collect_inputs(args, bundle["choices"])

    predicted = float(bundle["model"].predict(row)[0])
    low, high = predicted * bundle["ratio_low"], predicted * bundle["ratio_high"]

    print("\n" + "-" * 52)
    print(f"Predicted market price : {predicted:>12,.0f} MAD")
    print(f"Estimated fair range   : {low:>12,.0f} - {high:,.0f} MAD")
    print(f"Seller price           : {seller_price:>12,.0f} MAD")
    print("-" * 52)
    print(f"Result: {verdict(seller_price, low, high)}")
    print(f"\nTypical error of this model: about {bundle['metrics']['MAE']:,.0f} MAD "
          f"({bundle['model_name']}). Hidden damage and negotiation are not captured.")


if __name__ == "__main__":
    main()
