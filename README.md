# Housing Regression Project

A machine learning project for predicting housing prices using various regression algorithms on the Boston Housing dataset.

## Overview

This project implements and compares three different regression models to predict median home values:
- **Linear Regression**: Simple linear model for baseline comparison
- **Ridge Regression**: Regularized linear model to prevent overfitting
- **Random Forest**: Ensemble method for capturing non-linear relationships

## Features

- Data loading and preprocessing utilities
- Multiple regression model implementations
- Model evaluation with MSE and R² metrics
- Model persistence using joblib
- Modular code structure for easy extension

## Project Structure

```
HousingRegression/
├── regression.py      # Main script for training and evaluating models
├── utils.py          # Utility functions for data processing and evaluation
├── requirements.txt  # Python dependencies
└── README.md        # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/BhardwajAkash29/HousingRegression.git
cd HousingRegression
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

The project uses the Boston Housing dataset, which contains:
- **506 samples** of housing data
- **13 features** including crime rate, property tax, pupil-teacher ratio, etc.
- **Target variable**: Median home value (MEDV)

### Features:
- `CRIM`: Crime rate per capita
- `ZN`: Proportion of residential land zoned for lots over 25,000 sq.ft.
- `INDUS`: Proportion of non-retail business acres
- `CHAS`: Charles River dummy variable
- `NOX`: Nitric oxides concentration
- `RM`: Average number of rooms per dwelling
- `AGE`: Proportion of owner-occupied units built prior to 1940
- `DIS`: Weighted distances to employment centers
- `RAD`: Index of accessibility to radial highways
- `TAX`: Property tax rate
- `PTRATIO`: Pupil-teacher ratio
- `B`: Proportion of blacks by town
- `LSTAT`: % lower status of the population

## Usage

Run the main regression script:
```bash
python regression.py
```

This will:
1. Load and preprocess the Boston Housing dataset
2. Split data into training and testing sets (80/20 split)
3. Train three different regression models
4. Evaluate each model using MSE and R² metrics
5. Save trained models as `.joblib` files

## Model Performance

The script outputs performance metrics for each model:
- **MSE (Mean Squared Error)**: Lower values indicate better performance
- **R² (R-squared)**: Higher values (closer to 1) indicate better fit

Example output:
```
LinearRegression: MSE = 21.52, R² = 0.67
Ridge: MSE = 21.63, R² = 0.67
RandomForest: MSE = 11.25, R² = 0.87
```

## Customization

### Adding New Models
To add a new regression model:
1. Import the model in `regression.py`
2. Add it to the `models` dictionary
3. The evaluation pipeline will automatically include it

### Modifying Data Processing
The `utils.py` file contains utility functions that can be modified:
- `load_data()`: Customize data loading
- `split_data()`: Adjust train-test split parameters
- `evaluate_model()`: Add new evaluation metrics

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- joblib
- pytest (for testing)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- Boston Housing dataset from the UCI Machine Learning Repository
- Scikit-learn for machine learning algorithms
- Python community for excellent libraries

---

**Author**: Akash Bhardwaj  
**GitHub**: [@BhardwajAkash29](https://github.com/BhardwajAkash29)
