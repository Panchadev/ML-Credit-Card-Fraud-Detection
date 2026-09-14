# ML-Credit-Card-Fraud-Detection
# Credit Card Fraud Detection using Machine Learning

A machine learning project for detecting fraudulent credit card transactions using a **Random Forest Classifier**.

## Project Overview

Credit card fraud detection is a binary classification problem where transactions are classified as either **Normal** or **Fraudulent**. This project analyzes transaction data, investigates class imbalance, examines feature correlations, and builds a Random Forest model to identify fraudulent transactions.

## Dataset

The dataset contains **284,807 transactions** with 31 columns, including:

* `Time` — Time elapsed between transactions
* `V1`–`V28` — Anonymized transaction features
* `Amount` — Transaction amount
* `Class` — Target variable (`0` = Normal, `1` = Fraud)

The dataset contains **492 fraudulent transactions** and **284,315 normal transactions**, demonstrating a highly imbalanced classification problem.

## Workflow

1. Imported and prepared the dataset using Pandas and NumPy.
2. Performed exploratory data analysis using descriptive statistics.
3. Analyzed the severe class imbalance between fraudulent and normal transactions.
4. Examined transaction amount distributions for both classes.
5. Generated a correlation matrix to analyze relationships between features.
6. Separated input features (`X`) and target variable (`Y`).
7. Split the dataset into **80% training and 20% testing** sets.
8. Trained a **Random Forest Classifier** using Scikit-learn.
9. Evaluated the model using multiple classification metrics.
10. Visualized the results using a confusion matrix.

## Model Performance

The Random Forest model achieved:

| Metric                           |  Score |
| -------------------------------- | -----: |
| Accuracy                         | 99.96% |
| Precision                        | 97.47% |
| Recall                           | 78.57% |
| F1-Score                         | 87.01% |
| Matthews Correlation Coefficient |  0.875 |

The results show very high overall accuracy and precision while maintaining a strong ability to identify fraudulent transactions.

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Google Colab
* Random Forest

## Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── ML_Credit_Card_Fraud_Detection.ipynb
└── README.md
```

## How to Run

1. Clone the repository.
2. Open the `.ipynb` notebook in Jupyter Notebook or Google Colab.
3. Place the credit card dataset in the required location.
4. Run the notebook cells sequentially.

## Key Takeaway

This project demonstrates the application of supervised machine learning to a highly imbalanced fraud detection problem and highlights the importance of evaluating classification models using metrics beyond accuracy.
