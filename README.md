# Laptop Price Analysis and Predictive Modeling

An end-to-end machine learning project that predicts laptop prices from hardware specifications, while uncovering key pricing drivers and market segments through interpretable models and unsupervised learning.
The project combines regression modeling, regularization techniques, PCA, and clustering to analyze both **price accuracy** and **market structure**.

## Technology Stack
- **Programming Language**: Python
- **Data Processing**: pandas, NumPy
- **Visualization**: matplotlib, seaborn, plotly
- **Machine Learning**: scikit-learn, XGBoost
- **Modeling Techniques**:
  - Linear Regression, Ridge, Lasso
  - Logistic Regression (for exploratory feature diagnosis)
  - KNN, Random Forest (benchmarking)
  - XGBoost Regressor
  - PCA + K-Means Clustering
 
## Key Features
- **End-to-End ML Pipeline**: from data preprocessing to model evaluation
- **High-Dimensional Feature Handling**: 870+ features via one-hot encoding
- **Regularization-Aware Modeling**: systematic comparison of unregularized vs. L1/L2 models
- **Non-linear Price Modeling**: XGBoost to capture premium price jumps
- **Market Segmentation**:
  - PCA for dimensionality reduction
  - K-Means clustering to identify budget, mainstream, and premium laptop tiers
- **Interpretability-Focused Analysis**:
  - Correlation analysis
  - Ridge vs. Lasso coefficient comparison
  - Cluster-level price interpretation
 
## The Process
**1. Data Preprocessing**
- Checked and confirmed no missing values
- One-hot encoded categorical variables (brand, CPU/GPU models, form factors)
- Applied Min–Max scaling to numerical features
- Performed an 80/20 train-test split with fixed random seed for reproducibility

**2. Exploratory Data Analysis (EDA)**
- Analyzed price distribution and applied log transformation to stabilize variance
- Identified strong multicollinearity (e.g., screen width vs. height)
- Conducted exploratory logistic regression (high-end vs. budget) to study feature stability under L1/L2 regularization

**3. Unsupervised Learning & Market Structure**
- Applied PCA on core numerical specifications
- Used K-Means clustering on PCA-reduced space
- Identified four clear market segments corresponding to pricing tiers (budget → premium)

**4. Predictive Modeling**
- Established baselines using Linear, Ridge, and Lasso regression
- Demonstrated failure of unregularized linear regression due to overfitting
- Introduced XGBoost to capture non-linear pricing patterns
- Benchmarked performance using MSE, RMSE, MAE, and R²

**5. Model Comparison & Selection**
- Ridge Regression as a strong, interpretable baseline
- XGBoost selected as the final model with the best test performance (R² ≈ 0.90)

## What I Learned
- Regularization is critical in high-dimensional, sparse feature spaces
- Excellent training performance can be misleading without proper generalization checks
- EDA + unsupervised learning can guide model choice before fitting complex models
- Tree-based models outperform linear models when pricing involves non-linear jumps
- PCA and clustering are powerful tools not only for dimensionality reduction, but also for business insight generation

## How This Project Can Be Improved
- Incorporate hyperparameter tuning via Grid Search or Bayesian Optimization
- Add SHAP or permutation importance for deeper model interpretability
- Explore hierarchical or brand-aware models to better capture brand premiums
- Deploy the model as an API or interactive dashboard for real-world usage
- Include time-aware pricing data to model market trends and depreciation

## How to Run the Project
**1. Clone the repository**
```bash
git clone https://github.com/your-username/Laptop-Price-Prediction.git
cd Laptop-Price-Prediction
```
**2. (Recommended) Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate     # macOS/Linux
# venv\Scripts\activate      # Windows
```
**3. Install dependencies**

If you already included pip install ... cells inside the notebook, you can skip this step.

Otherwise, install manually:
```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost jupyter
```
**4. Start Jupyter Notebook from the project root**

Important: Run jupyter notebook at the repo root (the folder that contains Laptop_Price_Analysis.ipynb).

This ensures relative paths like data/processed_data/... resolve correctly.
```bash
jupyter notebook
```
**5. Open and run the main notebook**

Open Laptop_Price_Analysis.ipynb

Run all cells (Kernel → Restart & Run All)
