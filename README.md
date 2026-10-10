# Moroccan Used Car Price Predictor

A machine learning project that estimates the market price of a used car in Morocco from its listing details. The long-term goal is to tell a buyer whether a listed price looks underpriced, fair, or overpriced.

> **Status: work in progress.** Data cleaning, feature preparation, and model training are done. Evaluation, model comparison, and the terminal tool are still to do (see the roadmap).

## Dataset

[Moroccan Used Cars Dataset (MUCars, 2024)](https://data.mendeley.com/), with 101,896 listings and 15 columns. After removing rows with a missing price or key features, about 75,000 listings remain.

Features currently used: `Year`, `Mileage`, `Fiscal Power`, `Brand`, `Model`, `Fuel`, `Gearbox`, `Location`. The target is `Price` (MAD).

## What's done

- Loaded and explored the data (shape, types, missing values, summary statistics)
- Dropped rows with a missing price, since price is the target
- Converted `Year` and `Fiscal Power` ("8 CV") to numbers
- Converted `Mileage` ranges (for example "200 000 - 249 999") to range midpoints
- Standardized text columns (trimmed and title-cased brand, model, fuel, gearbox, location)
- One-hot encoded categorical features
- Trained a Random Forest Regressor on an 80/20 train/test split

## Roadmap

- [ ] Evaluate the model (MAE, RMSE, R²)
- [ ] Compare Linear Regression, Decision Tree, and Random Forest
- [ ] Handle price outliers and clean brand/model names
- [ ] Classify a listing as underpriced, fair, or overpriced by comparing the predicted and seller price
- [ ] Terminal tool: enter car details and get a predicted price and verdict
- [ ] Save the trained model so it doesn't retrain on every run

## Results

To be added after evaluation.

## Limitations

- Prices are listing prices, not final sale prices, so negotiation isn't reflected.
- The dataset has no accident history or hidden-damage information, both of which strongly affect price.
- Mileage is only available as ranges, so the midpoint is an approximation.
- Data comes from a single year (2024), so the model may not reflect later market changes.

## Project structure

```
moroccan-used-car-price-predictor/
├── data/             
├── docs/             
├── notebook/
│   └── used_car_price_predictor.ipynb
├── Project_Proposal.docx
├── requirements.txt
└── README.md
```

## Getting started

```bash
git clone https://github.com/<shahd-0>/moroccan-used-car-price-predictor.git
cd moroccan-used-car-price-predictor
pip install -r requirements.txt
jupyter notebook notebook/used_car_price_predictor.ipynb
```

The notebook reads the dataset from `../data/cars_dataframe.csv`.

## Tech stack

Python, pandas, NumPy, scikit-learn, Jupyter

## Author

Shahd Islam, Computer Science (AI and Machine Learning), Al Akhawayn University
