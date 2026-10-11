| Model | MAE (MAD) | RMSE (MAD) | R² | Median % error |
|---|---|---|---|---|
| Linear Regression | 25,548 | 74,595 | 0.652 | 10.7% |
| Decision Tree | 23,351 | 70,564 | 0.689 | 9.5% |
| Random Forest | 18,763 | 66,014 | 0.728 | 7.2% |

Best model: **Random Forest**. Fair range = predicted price x [0.82, 1.19] (covers the middle 80% of test listings).
Rows used: 72,366 (train 57,892 / test 14,474).
