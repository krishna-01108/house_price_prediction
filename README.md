# 🏠 House Price Prediction — Ames Housing Dataset (XGBoost)

Predicting residential house sale prices in Ames, Iowa using gradient-boosted trees (XGBoost), with a focus on feature engineering, proper encoding strategy, and target transformation to handle skewed price distributions.

---

## 📊 Project Overview

This project builds an end-to-end regression pipeline to predict `SalePrice` from ~80 raw features describing house characteristics (size, quality, location, age, amenities, etc.). The main goal was to move beyond a naive baseline and systematically improve model performance through:

- Domain-aware missing value imputation
- Engineered features (total area, age, bathroom counts, etc.)
- Proper encoding of ordinal vs. nominal categorical variables
- Log-transformation of the (right-skewed) target variable
- Hyperparameter-tuned XGBoost regression

---

## 📁 Dataset

**Source:** [Ames Housing Dataset](http://jse.amstat.org/v19n3/decock.pdf) — a modern alternative to the classic Boston Housing dataset, compiled by Dean De Cock.

- ~2,930 rows, 80+ features
- Features include lot size, building type, year built/remodeled, basement/garage details, quality ratings, and neighborhood
- Target variable: `SalePrice` (USD)



---

## 🧠 Approach

### 1. Missing Value Handling
Many `NaN` values in this dataset don't mean "missing data" — they mean the feature doesn't exist for that house (e.g., no pool, no basement, no garage). These were imputed accordingly:
- Categorical "quality" columns (`Pool QC`, `Fireplace Qu`, `Garage Type`, etc.) → filled with `"None"`
- Numeric columns tied to absent features (`Garage Area`, `BsmtFin SF 1`, etc.) → filled with `0`
- `Lot Frontage` → filled using the **median per neighborhood**, since lot size correlates strongly with location

### 2. Feature Engineering
New features created to capture signal not directly present in raw columns:
| Feature | Description |
|---|---|
| `HouseAge` | Years between sale and construction |
| `YearsSinceRemodel` | Years since last remodel |
| `TotalSF` | Combined basement + 1st floor + 2nd floor square footage |
| `TotalBathrooms` | Weighted total of full/half baths (incl. basement) |
| `HasPool`, `HasGarage`, `HasFireplace` | Binary presence flags |
| `Remodeled` | Whether the house was ever remodeled |

### 3. Encoding Strategy
- **Ordinal quality columns** (`Exter Qual`, `Kitchen Qual`, `Bsmt Qual`, etc.) — manually mapped to numeric rank (`Ex=5, Gd=4, TA=3, Fa=2, Po=1, None=0`) to preserve their true order, rather than relying on arbitrary alphabetical encoding
- **Nominal categorical columns** (`Neighborhood`, `Exterior 1st`, etc.) — ordinal-encoded for compatibility with XGBoost's split-based tree structure
- **Unseen categories** in validation/test folds handled gracefully via `handle_unknown='use_encoded_value'`

### 4. Target Transformation
`SalePrice` is right-skewed (a long tail of high-value homes). Applied `log1p` transformation to the target before training, and `expm1` to invert predictions back to the original dollar scale — implemented cleanly using `sklearn.compose.TransformedTargetRegressor` so the whole pipeline remains a single fit/predict interface.

### 5. Model
- **Algorithm:** XGBoost Regressor
- **Validation:** 5-fold cross-validation (in addition to a held-out test split) to get a stable performance estimate given the moderate dataset size
- **Tuning:** `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda` tuned via randomized search

---

## 📈 Results

| Metric | Score |
|---|---|
| R²  | *0.9438371658325195* |
| MAPE | *0.07194089144468307* |

---

## 🛠️ Tech Stack
- Python
- pandas, numpy
- scikit-learn (Pipeline, ColumnTransformer, TransformedTargetRegressor)
- XGBoost
- matplotlib / seaborn (EDA & visualization)

---

## 📂 Project Structure
```
house-price-prediction/
├── data/
│   └── AmesHousing.csv          # not included — see Dataset section
├── notebooks/
│   └── eda_and_modeling.ipynb   # exploratory analysis + experimentation
├── models/
│   └── xgb_model.pkl            # saved trained model (optional)
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

1. Clone the repo
```bash
git clone https://github.com/<your-username>/house-price-prediction.git
cd house-price-prediction
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Add the dataset
Download `AmesHousing.csv` from Kaggle and place it in the `data/` folder.

4. Run the notebook
Open `notebooks/eda_and_modeling.ipynb` and run all cells to reproduce the EDA, preprocessing, and model training.

---

## 🔍 Key Learnings
- Log-transforming the target doesn't always help — its benefit depends on how skewed the actual target distribution is, and whether R² or a percentage-based metric (like MAPE) is the priority. On a dataset with mild skew, it can even slightly hurt R² while improving MAPE.
- Ordinal encoding of quality-ranked categories (using true domain order) outperforms naive alphabetical/generic ordinal encoding.
- Feature engineering (especially combining sparse area/room columns into totals) contributed more to score improvement than hyperparameter tuning alone.
- Cross-validation is essential on medium-sized datasets — a single train/test split can vary by several R² points depending on the random seed.

---

## 🚀 Future Improvements
- Try proper K-fold target encoding for `Neighborhood` instead of ordinal encoding
- Add SHAP-based feature importance analysis for interpretability
- Compare against other models (Random Forest, LightGBM, CatBoost, stacked ensembles)
- Deploy as a simple API/Streamlit app for interactive price prediction

---

## 📄 License
This project is open-source and available under the MIT License.
