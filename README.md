# Customer Churn Prediction

A simple ANN (Artificial Neural Network) project to predict whether a bank customer will leave or stay.

## Dataset
- **Source:** Churn_Modelling.csv
- **Total Rows:** 10,000 customers
- **Target:** `Exited` column (1 = churned, 0 = stayed)

## What I Did
- Dropped unnecessary columns (RowNumber, CustomerId, Surname)
- Applied One-Hot Encoding on Geography and Gender
- Standardized features using StandardScaler
- Built and compared 3 ANN models using PyTorch

## Models
| Model | Test Accuracy |
|-------|--------------|
| Base Model | 85.45% |
| Modified Model 1 (extra hidden layer) | 86.65% |
| Modified Model 2 (Tanh activation) | 86.15% |

**Best Model:** Modified Model 1

## Best Model Architecture
```
Input (11) → Linear(16) → ReLU → Linear(12) → ReLU → Linear(8) → ReLU → Linear(1) → Sigmoid
```

## Libraries Used
- Python, PyTorch, Pandas, Scikit-learn, Matplotlib

## What I Learned
- How to build an ANN from scratch using PyTorch
- How to detect overfitting using Train vs Validation Loss
- Why accuracy alone is not enough for imbalanced datasets
