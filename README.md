# Moroccan Used Car Price Predictor

A machine learning project that estimates the market price of a used car in Morocco from its listing details. The long-term goal is to tell a buyer whether a listed price looks underpriced, fair, or overpriced.

> **Status: work in progress.** Data cleaning, feature preparation, and model training are done. Evaluation, model comparison, and the terminal tool are still under revision.

## Dataset

[Moroccan Used Cars Dataset (MUCars, 2024)](https://data.mendeley.com/), with 101,896 listings and 15 columns. After removing rows with a missing price or key features, about 72,366 listings remain. The dataset isn't included in this repo, but it is in `data/README.md`.

Features currently used: `Year`, `Mileage`, `Fiscal Power`, `Brand`, `Model`, `Fuel`, `Gearbox`, `Location`. The target is `Price` (MAD).

## What's done

- Loaded and explored the data (shape, types, missing values, summary statistics)
- Dropped rows with a missing price, since price is the target
- Converted `Year` and `Fiscal Power` ("8 CV") to numbers
- Converted `Mileage` ranges (for example "200 000 - 249 999") to range midpoints
- Standardized text columns (trimmed and title-cased brand, model, fuel, gearbox, location)
- One-hot encoded categorical features
- Handled price outliers and cleaned brand names
- Trained a Random Forest Regressor on an 80/20 train/test split
- Evaluated the model against MAE, RMSE and R²
- Compared Linear Regression, Decision Tree, and Random Forest


## Results

80/20 train/test split, `random_state=42`. Errors are in Moroccan dirhams (MAD).

| Model | MAE (MAD) | RMSE (MAD) | R² | Median % error |
|---|---|---|---|---|
| Linear Regression | 25,548 | 74,595 | 0.652 | 10.7% |
| Decision Tree | 23,351 | 70,564 | 0.689 | 9.5% |
| **Random Forest** | **18,763** | **66,014** | **0.728** | **7.2%** |

The Random Forest is used by the terminal tool. Its fair range is the predicted price x [0.82, 1.19].


## Run it

```bash
git clone https://github.com/<shahd-0>/moroccan-used-car-price-predictor.git
cd moroccan-used-car-price-predictor
pip install -r requirements.txt
```

1. Put `cars_dataframe.csv` in `data/` (see `data/README.md`).
2. Train and save the model (about 4 minutes): `python src/train.py`
3. Check a listing:

```bash
python src/predict.py          
```

or in one line:

```bash
python src/predict.py --brand Dacia --model Sandero --year 2015 --mileage 150000 \
    --fiscal-power 6 --fuel Diesel --gearbox Manual --location Casablanca --price 85000
```

Brand, model, and city are matched case-insensitively, and typos get suggestions ("Did you mean: Dacia?"). Condition, origin, first owner, and doors are optional; press Enter to skip.

The notebook `notebook/used_car_price_predictor.ipynb` walks through the same pipeline with outputs.

## Project structure

```
moroccan-used-car-price-predictor/
├── data/
│   └── README.md                 
├── docs/
│   └── results.md                
├── notebook/
│   └── used_car_price_predictor.ipynb
├── src/
│   ├── preprocessing.py          
│   ├── train.py                   
│   └── predict.py                
├── requirements.txt
└── README.md
```

## Limitations

- These are **listing prices**, not final sale prices, so negotiation is not captured.
- The dataset has no accident history or hidden-damage information, both of which strongly affect price.
- Mileage is only available as ranges, so the midpoint is used.
- Accuracy is lower for expensive and rare cars (RMSE is much larger than MAE), and the model has seen few examples of some brands and models.
- The fair range is derived from the model's own test errors. The underpriced/fair/overpriced verdict has not been validated against expert-labelled listings.
- The price limits (10,000 to 1,500,000 MAD) are a judgement call and can be changed in `src/preprocessing.py`.
- Data comes from a single year (2024).

## Possible improvements

- Tune the Random Forest hyperparameters and try gradient boosting
- Harmonize model names (for example "Megane_4" vs "Megane 4") and use the Equipment column
- A small web interface

## Author

Shahd Islam, Computer Science (AI and Machine Learning), Al Akhawayn University
  
