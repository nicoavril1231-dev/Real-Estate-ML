# Real Estate Price Prediction - ML Pipeline

A complete end-to-end machine learning pipeline for predicting real estate prices using advanced regression techniques and feature engineering.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-green)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This project provides a production-ready machine learning pipeline for real estate price prediction. It includes comprehensive data preprocessing, feature engineering, model training, and evaluation capabilities.

### Key Features

- **Complete Data Pipeline**: From raw data to production-ready predictions
- **Feature Engineering**: 29 engineered features including interactions and transformations
- **Multiple ML Models**: Ridge, Random Forest, XGBoost, and more
- **Model Evaluation**: Comprehensive metrics (RMSE, MAE, R2, MAPE)
- **Production Ready**: Modular code structure with reusable components
- **Data Quality**: Built-in data cleaning and validation

### Performance

Best model (XGBoost) achieves:
- **R² Score**: 0.60
- **RMSE**: 36,931 EUR
- **MAE**: 20,382 EUR
- **MAPE**: 30.25%

## Project Structure

```
real-estate-ml-pipeline/
├── data/                          # Data files
│   ├── real_estate_data_clean.csv # Clean dataset
│   ├── train_data_clean.csv       # Training set
│   └── test_data_clean.csv        # Test set
├── models/                        # Saved models and results
│   └── test_results.csv           # Model comparison results
├── notebooks/                     # Jupyter notebooks
│   ├── 01_EDA.ipynb              # Exploratory analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── scripts/                       # Utility scripts
│   ├── generate_complete_dataset.py  # Data generation
│   ├── data_quality_check.py         # Quality control
│   └── test_pipeline.py              # Pipeline testing
├── src/                          # Source code modules
│   ├── data_preprocessing.py     # Preprocessing pipeline
│   ├── models.py                 # Model training
│   └── evaluation.py             # Evaluation utilities
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
```

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/real-estate-ml-pipeline.git
cd real-estate-ml-pipeline
```

2. **Create virtual environment** (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

## Quick Start

### Option 1: Run Complete Test Pipeline

```bash
python scripts/test_pipeline.py
```

This will:
- Load and preprocess data
- Train multiple models (Ridge, RandomForest, XGBoost)
- Display evaluation metrics
- Save results to `models/test_results.csv`

### Option 2: Use Python API

```python
from src.data_preprocessing import RealEstatePreprocessor, load_data
from xgboost import XGBRegressor
import numpy as np

# Load data
df_train = load_data('data/train_data_clean.csv')
df_test = load_data('data/test_data_clean.csv')

# Preprocess
preprocessor = RealEstatePreprocessor()
X_train, y_train = preprocessor.preprocess(df_train, target_col='price', fit=True)
X_test, y_test = preprocessor.preprocess(df_test, target_col='price', fit=False)

# Train model
model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
from sklearn.metrics import r2_score, mean_squared_error
print(f"R2: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):,.2f} EUR")
```

### Option 3: Jupyter Notebooks

```bash
jupyter notebook
```

Open notebooks in order:
1. `01_EDA.ipynb` - Data exploration
2. `02_feature_engineering.ipynb` - Feature creation
3. `03_model_training.ipynb` - Model training

## Features

### Data Processing

The pipeline includes:

- **Outlier Treatment**: IQR-based capping
- **Feature Engineering**:
  - Numeric: price_per_m2, surface_per_room, log_surface, surface_squared
  - Categorical: age_category, surface_category, floor_category
  - Interactions: city × surface
  - Composite: comfort_score (elevator + parking + balcony)
- **Encoding**: One-hot, ordinal, and target encoding
- **Scaling**: RobustScaler (resistant to outliers)

### Models Supported

- Linear: Ridge, Lasso, ElasticNet
- Tree-based: Decision Tree, Random Forest
- Boosting: Gradient Boosting, XGBoost, LightGBM, CatBoost

### Evaluation Metrics

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- R² Score

## Usage Examples

### Train Multiple Models

```python
from src.models import ModelTrainer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

trainer = ModelTrainer()
trainer.add_model('RandomForest', RandomForestRegressor(n_estimators=100, random_state=42))
trainer.add_model('XGBoost', XGBRegressor(n_estimators=100, random_state=42))

results = trainer.train_all(X_train, y_train, X_test, y_test)
print(results)
```

### Make Predictions on New Data

```python
import pandas as pd

# New property data
new_property = pd.DataFrame({
    'city': ['Paris'],
    'surface_m2': [45.0],
    'rooms': [2],
    'age_years': [10.0],
    'floor': [3],
    'has_elevator': [1],
    'has_parking': [0],
    'has_balcony': [1],
    'energy_class': ['B'],
    'price': [0]  # Placeholder
})

# Preprocess
X_new, _ = preprocessor.preprocess(new_property, target_col='price', fit=False)

# Predict
predicted_price = model.predict(X_new)[0]
print(f"Predicted price: {predicted_price:,.2f} EUR")
```

### Save and Load Models

```python
import joblib

# Save
joblib.dump(model, 'models/my_model.pkl')
joblib.dump(preprocessor, 'models/my_preprocessor.pkl')

# Load
model = joblib.load('models/my_model.pkl')
preprocessor = joblib.load('models/my_preprocessor.pkl')
```

## Data Format

Input data should be a CSV file with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| city | str | City name (e.g., 'Paris', 'Lyon') |
| surface_m2 | float | Surface area in square meters |
| rooms | int | Number of rooms |
| age_years | float | Age of property in years |
| floor | int | Floor number |
| has_elevator | int | 1 if elevator, 0 otherwise |
| has_parking | int | 1 if parking, 0 otherwise |
| has_balcony | int | 1 if balcony, 0 otherwise |
| energy_class | str | Energy class (A-G) |
| price | float | Price in EUR (target variable) |

## Scripts

### Generate Dataset

```bash
python scripts/generate_complete_dataset.py
```

Generates synthetic real estate data with realistic features.

### Data Quality Check

```bash
python scripts/data_quality_check.py
```

Performs comprehensive data quality checks and cleaning:
- Missing values detection
- Duplicate removal
- Outlier detection
- Consistency validation

### Test Pipeline

```bash
python scripts/test_pipeline.py
```

Runs end-to-end pipeline test with multiple models.

## Development

### Running Tests

```bash
python scripts/test_pipeline.py
```

### Adding New Features

Edit `src/data_preprocessing.py`, method `create_features()`:

```python
def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Add your custom feature here
    df['my_new_feature'] = df['feature1'] * df['feature2']

    return df
```

### Adding New Models

```python
from src.models import ModelTrainer
from sklearn.ensemble import GradientBoostingRegressor

trainer = ModelTrainer()
trainer.add_model('GradientBoosting', GradientBoostingRegressor())
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- scikit-learn for machine learning tools
- XGBoost team for the gradient boosting library
- Python data science community

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Made with Python and scikit-learn**
