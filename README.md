# Wine Quality Prediction with Linear Regression

## Overview
This project builds a **linear regression model** to predict wine quality using the **UCI Wine Quality dataset**.  
The workflow covers data analysis, feature engineering, scaling, model training, hyperparameter tuning, and evaluation.

---

## Dataset
- **Source:** UCI Machine Learning Repository  
- **Data:** Red and white wine physicochemical properties  
- **Target:** Wine quality score (integer)

---

## Methodology
The following steps were implemented:

- Initial data analysis (missing values, duplicates, data types)
- Manual **feature engineering** (4 new domain-based features)
- Feature scaling using **StandardScaler**
- Data split into **train / validation / test** sets (60 / 20 / 20)
- Training a **baseline Linear Regression** model
- Hyperparameter tuning with **Ridge, Lasso, and ElasticNet**
- Model evaluation using **R², RMSE, and MAE**

---

## Feature Engineering
New features include:
- Sulfur dioxide ratio  
- Acidity ratio  
- Alcohol–acidity interaction  
- Total acidity  

These features improved model performance and interpretability.

---

## Results
- Best model selected via validation performance
- Regularized regression improved results over baseline
- Alcohol content showed the strongest positive impact on quality

---

## Outputs
- `wine_quality_analysis.png` — model and data visualizations  
- Trained and evaluated regression models  

---

## Tools
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  

---

## Conclusion
The project demonstrates a complete regression pipeline, including preprocessing, feature engineering, tuning, and evaluation on a real-world dataset.

