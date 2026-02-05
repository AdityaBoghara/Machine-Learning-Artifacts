# Ridge Regression: Professional Reference Notes

## Overview

Ridge Regression is a regularized linear regression technique that addresses multicollinearity and overfitting by adding an L2 penalty term to the ordinary least squares (OLS) objective function. It was introduced by Hoerl and Kennard in 1970 and remains one of the most widely used regularization methods in predictive modeling.

### Position in Supervised Machine Learning

Ridge Regression is part of the supervised machine learning family, specifically under regression algorithms. In the supervised ML hierarchy:

**Supervised ML Algorithms:**
- **Regression** (for continuous output)
  - Linear Regression
  - **Ridge Regression (L2 Regularization)**
  - Lasso Regression (L1 Regularization)
  - Elastic Net
  - Polynomial Regression
- **Classification** (for categorical output)
  - Logistic Regression
  - Decision Trees
  - Random Forest
  - AdaBoost, XGBoost

## Mathematical Foundation

### Problem Context: Overfitting in Linear Regression

**The Overfitting Problem:**
When training a linear regression model, we often encounter:
- **Training data**: High accuracy (low bias)
- **Test data**: Low accuracy (high variance)

This is the classic overfitting scenario that Ridge Regression addresses through **L2 Regularization**.

### Objective Function

Ridge regression modifies the standard linear regression cost function by adding a penalty term:

**Standard Linear Regression Cost Function:**
```
J(β) = Σ(h(xi) - yi)²
```

**Ridge Regression Cost Function (L2 Regularization):**
```
J(β) = Σ(h(xi) - yi)² + λΣ(slope)²
```

Or in matrix notation:
```
J(β) = ||y - Xβ||² + λ||β||²
```

Where:
- **y** is the response vector (n × 1)
- **X** is the design matrix (n × p)
- **β** is the coefficient vector (p × 1)
- **λ** (lambda) is the regularization parameter (λ ≥ 0) - **hyperparameter**
- **||·||²** denotes the L2 norm (sum of squared values)
- The penalty is applied to the **slope coefficients**, not the intercept

### Understanding the Lambda (λ) - Slope Relationship

**Key Insight:** As λ increases, the slope coefficients shrink toward zero.

- **λ = 0**: No regularization → Standard OLS regression
- **λ → ∞**: Maximum regularization → Coefficients approach zero
- **Optimal λ**: Found through cross-validation

**Coefficient Path Behavior:**
In Ridge regression, the coefficient paths **smoothly decrease** toward zero as λ increases. Unlike Lasso, coefficients never become exactly zero - they asymptotically approach zero.

### Closed-Form Solution

Unlike iterative methods, Ridge regression has an analytical solution:

```
β̂_ridge = (X'X + λI)⁻¹X'y
```

Where **I** is the p × p identity matrix. This formulation ensures the matrix is always invertible, even when X'X is singular.

## Key Characteristics

### Bias-Variance Tradeoff

Ridge regression intentionally introduces bias to reduce variance, addressing the overfitting problem.

#### Understanding Bias and Variance

**Scenario 1: Generalized Model (Ideal)**
- Training Accuracy: Very good (90%+) → **Low Bias**
- Test Accuracy: Very good (90%+) → **Low Variance**
- **Result**: Perfect! Model generalizes well

**Scenario 2: Overfitting (Problem Ridge Solves)**
- Training Accuracy: Very good (90%+) → **Low Bias**
- Test Accuracy: Very bad (50%) → **High Variance**
- **Result**: Model memorized training data, fails on new data
- **Solution**: Use Ridge regression to add regularization

**Scenario 3: Underfitting**
- Training Accuracy: Low (60%) → **High Bias**
- Test Accuracy: Low (55%) → **High Variance**
- **Result**: Model is too simple, hasn't learned patterns
- **Solution**: More complex model, more features, or different algorithm

#### How Ridge Addresses Overfitting

**Without Ridge (Standard Linear Regression):**
- Fits training data perfectly
- Coefficients can become very large
- Small changes in data cause large changes in coefficients
- Poor generalization to test data

**With Ridge (L2 Regularization):**
- Adds penalty for large coefficients: `λΣ(slope)²`
- Forces coefficients to be smaller
- More stable predictions
- Better generalization

#### Lambda's Effect on Bias-Variance

- **Low λ (λ → 0)**: 
  - Approaches OLS
  - Low bias, high variance
  - Risk of overfitting
  
- **Optimal λ**: 
  - Balanced bias-variance tradeoff
  - Best generalization
  - Found through cross-validation
  
- **High λ (λ → ∞)**: 
  - High bias, low variance
  - Coefficients shrink toward zero
  - Risk of underfitting

### Coefficient Shrinkage

Unlike variable selection methods, Ridge regression:
- Shrinks all coefficients toward zero proportionally
- Never sets coefficients exactly to zero
- Retains all predictors in the model
- Reduces the magnitude of coefficients, especially for highly correlated predictors

## Practical Applications

### When to Use Ridge Regression

1. **Multicollinearity**: When predictor variables are highly correlated
2. **High-dimensional data**: When p (predictors) approaches or exceeds n (observations)
3. **Overfitting prevention**: When OLS shows signs of overfitting
4. **Prediction focus**: When prediction accuracy is more important than interpretation

### Industry Use Cases

- **Finance**: Credit scoring, portfolio optimization, risk modeling
- **Healthcare**: Disease prediction, treatment response modeling
- **Marketing**: Customer lifetime value prediction, churn modeling
- **Manufacturing**: Quality control, process optimization
- **Real Estate**: Property valuation models

## Hyperparameter Tuning

### Data Splitting Strategy

Before tuning hyperparameters, proper data splitting is essential:

**Three-Way Split:**
```
Dataset (1000 samples)
├── Training Data (70% = 700): To train the model
├── Validation Data (15% = 150): For hyperparameter tuning
└── Test Data (15% = 150): To check final performance
```

**Purpose of Each Set:**
1. **Training**: Learn the model parameters (β coefficients)
2. **Validation**: Tune hyperparameters (λ value)
3. **Test**: Evaluate final model performance (unbiased estimate)

### Cross-Validation Techniques

#### 1. K-Fold Cross-Validation (Recommended)

**Most commonly used for Ridge regression hyperparameter tuning**

**Process:**
```
k = 5, Dataset = 800 samples, Test = 100 samples

Experiment 1: [Train][Train][Train][Train][Valid] → Acc₁
Experiment 2: [Train][Train][Train][Valid][Train] → Acc₂
Experiment 3: [Train][Train][Valid][Train][Train] → Acc₃
Experiment 4: [Train][Valid][Train][Train][Train] → Acc₄
Experiment 5: [Valid][Train][Train][Train][Train] → Acc₅

Final Accuracy = (Acc₁ + Acc₂ + Acc₃ + Acc₄ + Acc₅) / 5
```

**Advantages:**
- Efficient use of data
- Reduces variance in performance estimate
- Works well for hyperparameter tuning

**Typical values:** k = 5 or k = 10

#### 2. Stratified K-Fold

**Use case:** Primarily for classification, but concept applies when ensuring balanced splits

```
k = 5, Test size = 100
Ensures almost equal proportional distribution in each fold
```

#### 3. Leave-One-Out Cross-Validation (LOOCV)

**Process:**
```
Dataset = 500 samples
- Training: 499 samples
- Validation: 1 sample
Repeat 500 times, each time leaving a different sample out
```

**Advantages:**
- Maximum use of training data
- Deterministic (no randomness)

**Disadvantages:**
- **Computationally expensive** for large datasets
- Can lead to **overfitting**
  - Training Accuracy: ↑↑
  - Test Accuracy: ↓
- Not recommended for Ridge regression with large datasets

#### 4. Leave-P-Out Cross-Validation

Similar to LOOCV but leaves P values aside for validation instead of just 1.

#### 5. Time Series Cross-Validation

**Use case:** Only for time-series applications where order matters

```
Time: Jan → Dec
Day 1, Day 2, Day 3, ..., Day 10 [Validation]
Cannot shuffle data - must maintain temporal order
```

**Examples:** 
- Product reviews over time
- Stock prices
- Sentiment analysis with time component

### Selecting Lambda (λ)

The regularization parameter is typically selected using:

1. **K-Fold Cross-Validation** (Most common)
   - Test different λ values on each fold
   - Select λ with lowest average validation error
   
2. **Grid Search**: Test predefined λ values and select optimal
   ```python
   lambda_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
   ```

3. **Random Search**: Sample λ values from a distribution

4. **Regularization Path**: Evaluate model performance across λ sequence

### Common Lambda Ranges

- Start with λ values on a logarithmic scale: [0.001, 0.01, 0.1, 1, 10, 100, 1000]
- Refine around the optimal value identified
- Consider domain-specific constraints

### Cross-Validation Best Practices

**Random State:**
- Set a random seed for reproducibility
- Ensures consistent fold splits across experiments

**Avoid Data Leakage:**
- Never use test data for hyperparameter tuning
- Validation and test sets must remain separate
- Preprocessing (scaling) should be fit on training data only

## Implementation Considerations

### Data Preprocessing

**Critical preprocessing steps:**

1. **Standardization**: Ridge regression is scale-sensitive
   - Center features (subtract mean)
   - Scale to unit variance (divide by standard deviation)
   - Do NOT standardize the intercept/constant term

2. **Missing Values**: Handle before modeling
   - Imputation strategies
   - Complete case analysis
   - Multiple imputation

3. **Outliers**: Assess impact on model performance
   - Ridge is less robust to outliers than some alternatives
   - Consider outlier detection and treatment

### Model Evaluation

Ridge regression, being a regression algorithm, uses specific performance metrics for continuous output:

#### R-Squared (Coefficient of Determination)

**Formula:**
```
R² = 1 - [Σ(yi - ŷi)²] / [Σ(yi - ȳ)²]
```

Where:
- yi = actual value
- ŷi = predicted value  
- ȳ = mean of actual values

**Interpretation:**
```
R² = 1 - (small number / big number) = 1 - small number → close to 1
```
- **R² → 1**: Model is highly accurate (explains most variance)
- **R² → 0**: Model performs poorly
- The closer to 1, the better the model

**Problem with R²:** If you add more features (even unrelated ones), R² will still increase, which can be misleading.

**Example:**
- Original model: R² = 90%
- Add 1 unrelated feature: R² = 92% (appears better but might not be)

#### Adjusted R-Squared

**Formula:**
```
Adjusted R² = 1 - [(1 - R²) × (N - 1) / (N - P - 1)]
```

Where:
- N = Number of data points
- P = Number of independent features

**Purpose:** Penalizes the addition of irrelevant features, providing a more honest assessment of model quality.

#### Mean Squared Error (MSE)

**Formula:**
```
MSE = (1/n)Σ(yi - ŷi)²
```

**Characteristics:**
- **Cost function** → Also used as performance metric
- Quadratic equation
- **Unit**: Squared units of the target variable

**Advantages:**
1. Differentiable (useful for optimization)
2. Has one local minimum = one global minimum
3. Converges faster in gradient descent

**Disadvantages:**
1. **Not robust to outliers** (squaring amplifies large errors)
2. Not in the same unit as the target variable

**Example:**
If predicting salary in lakhs:
- Error = 2.5 lakhs
- MSE contribution = (2.5)² = 6.25 lakhs² ← Different unit!

#### Mean Absolute Error (MAE)

**Formula:**
```
MAE = (1/n)Σ|yi - ŷi|
```

**Advantages:**
1. **Robust to outliers** (no squaring)
2. **Same unit** as target variable
3. Easy to interpret

**Disadvantages:**
1. Convergence usually takes more time
2. Optimization is a complex task (not differentiable at zero)
3. Time consuming

#### Root Mean Squared Error (RMSE)

**Formula:**
```
RMSE = √MSE = √[(1/n)Σ(yi - ŷi)²]
```

**Advantages:**
1. **Same unit** as target variable (fixes MSE unit problem)
2. Differentiable (inherits from MSE)

**Disadvantages:**
1. **Not robust to outliers** (inherits from MSE)

#### Metric Selection Guide

| Scenario | Recommended Metric |
|----------|-------------------|
| General model comparison | R², Adjusted R² |
| Need interpretable error | MAE, RMSE |
| Outliers present in data | MAE |
| Need fast convergence | MSE, RMSE |
| Penalize large errors more | MSE, RMSE |
| Multiple models, different features | Adjusted R² |

## Comparison with Alternative Methods

### Linear Regression with OLS (Ordinary Least Squares)

**OLS Objective:**
```
Minimize: J(β₀, β₁) = (1/2n)Σ(yi - β₀ - β₁xi)²
```

**Optimization Approach:**
- Uses calculus to find optimal β₀ and β₁
- Takes partial derivatives and sets to zero
- Solves system of equations

**OLS Formulas (Derived from calculus):**
```
β₁ = Σ(xi - x̄)(yi - ȳ) / Σ(xi - x̄)²

β₀ = ȳ - β₁x̄
```

Where:
- β₀ = intercept
- β₁ = slope
- x̄ = mean of x
- ȳ = mean of y

### Ridge vs. OLS (Ordinary Least Squares)

| Aspect | OLS | Ridge |
|--------|-----|-------|
| **Cost Function** | Σ(h(x) - y)² | Σ(h(x) - y)² + λΣ(slope)² |
| **Multicollinearity** | Unstable coefficients | Stabilized coefficients |
| **Variance** | High with many predictors | Reduced through regularization |
| **Bias** | Unbiased | Biased but controlled |
| **Invertibility** | Requires X'X invertible | Always invertible (X'X + λI) |
| **Overfitting** | Prone to overfitting | Reduces overfitting |
| **Coefficient magnitude** | Can be very large | Constrained by penalty |
| **When to use** | Few features, no multicollinearity | Many features, multicollinearity present |

**Key Insight:** Ridge adds the term `+ λI` to `X'X`, ensuring the matrix is always invertible even when features are perfectly correlated.

### Polynomial Regression Context

Ridge regression can be applied to polynomial regression to prevent overfitting:

**Polynomial Degrees:**
- **Degree 0**: h(x) = β₀ (constant)
- **Degree 1**: h(x) = β₀ + β₁x (linear)
- **Degree 2**: h(x) = β₀ + β₁x + β₂x² (quadratic)
- **Degree n**: h(x) = β₀ + β₁x + β₂x² + ... + βₙxⁿ

**Problem:** Higher degree polynomials can severely overfit
**Solution:** Apply Ridge regularization

```
Cost = Σ(h(xi) - yi)² + λΣ(βⱼ)²  (for j ≥ 1)
```

This prevents the polynomial coefficients from becoming too large, reducing overfitting while maintaining model flexibility.

### Ridge vs. Lasso (L1 Regularization)

**Ridge Regression (L2):**
```
Cost = Σ(h(xi) - yi)² + λΣ(slope)²
```

**Lasso Regression (L1):**
```
Cost = Σ(h(xi) - yi)² + λΣ|slope|
```

| Aspect | Ridge (L2) | Lasso (L1) |
|--------|------------|------------|
| Penalty | Sum of squared coefficients | Sum of absolute coefficients |
| Feature selection | No (all coefficients retained) | Yes (can set coefficients to zero) |
| Coefficient path | **Smooth shrinkage toward zero** | **Coefficients drop to exactly zero abruptly** |
| Multicollinearity | Distributes weight among correlated features | Arbitrarily selects one |
| Use case | Reduce overfitting | Reduce overfitting + Feature selection |
| Output | All features with reduced coefficients | Sparse model with subset of features |

**Example Scenario:**
```
h(x) = 0.5 + 0.65x₁ + 0.72x₂ + 0.34x₃ + 0.12x₄
```

- **Ridge**: All coefficients remain but get smaller as λ increases
  - Result: `0.5 + 0.45x₁ + 0.52x₂ + 0.24x₃ + 0.08x₄`
  
- **Lasso**: Some coefficients become exactly 0
  - Result: `0.5 + 0.65x₁ + 0.72x₂ + 0x₃ + 0x₄`

### Ridge vs. Elastic Net

**Elastic Net Cost Function:**
```
Cost = Σ(h(xi) - yi)² + λ₁Σ(slope)² + λ₂Σ|slope|
```

Elastic Net combines L1 and L2 penalties to get **both** benefits:
- **Reduces overfitting** (from Ridge component)
- **Performs feature selection** (from Lasso component)
- Requires tuning **two hyperparameters** (λ₁ and λ₂)
- More flexible but more complex

**When to Use Each:**
- **Ridge**: When you want to keep all features and reduce overfitting
- **Lasso**: When you need feature selection and sparse models
- **Elastic Net**: When you need both overfitting reduction and feature selection

## Theoretical Properties

### Statistical Properties

1. **Consistency**: Ridge estimators can be consistent under certain conditions
2. **Mean Squared Error**: Ridge can achieve lower MSE than OLS when λ is properly chosen
3. **Degrees of Freedom**: Effective degrees of freedom decrease with increasing λ

### Geometric Interpretation

Ridge regression can be visualized as:
- Constraint optimization: minimize RSS subject to Σβ²ⱼ ≤ t
- The constraint region is a hypersphere in coefficient space
- OLS solution is modified to fall within this constraint region

## Advanced Considerations

### Computational Efficiency

- **Training complexity**: O(p²n + p³) for direct solution
- **Prediction complexity**: O(p) per observation
- Efficient for moderate p; consider iterative methods for very large p

### Bayesian Interpretation

Ridge regression is equivalent to:
- Maximum a posteriori (MAP) estimation
- With Gaussian prior on coefficients: β ~ N(0, σ²/λ)
- Provides probabilistic framework for interpretation

### Extensions and Variations

1. **Kernel Ridge Regression**: Non-linear extension using kernel methods
2. **Weighted Ridge**: Different penalties for different coefficients
3. **Grouped Ridge**: Penalty based on groups of coefficients
4. **Dynamic Ridge**: Time-varying regularization parameters

## Best Practices

### End-to-End ML Project Lifecycle

Ridge regression fits into a complete machine learning workflow:

```
1. Data Collection → 2. Feature Engineering → 3. Feature Selection
                             ↓
4. Model Training ← 6. Deployment ← 5. Model Evaluation
        ↓                                    ↑
    (Ridge Regression)                       |
                                    Cross-validation
```

**Typical Workflow:**
1. **Data Collection**: Gather relevant dataset
2. **Feature Engineering**: Create and transform features
3. **Feature Selection**: Identify most important predictors
4. **Model Training**: Train Ridge regression with optimal λ
5. **Model Evaluation**: Assess performance on validation/test data
6. **Deployment**: Deploy to production (web app, API, etc.)

**CI/CD Deployment:**
- **Continuous Integration**: Code changes automatically tested
- **Continuous Deployment**: Model updates automatically deployed
- Essential for maintaining model performance over time

### Model Development Workflow

1. **Exploratory Data Analysis**
   - Understand correlations between features
   - Check for multicollinearity (correlation matrix)
   - Visualize distributions
   - Identify outliers

2. **Feature Engineering**
   - Create relevant predictors
   - Handle categorical variables (encoding)
   - Create interaction terms if needed
   - Transform skewed distributions

3. **Data Splitting**
   - 70% Training, 15% Validation, 15% Test
   - Or: 80% Training+Validation (with K-fold), 20% Test
   - Maintain representative distributions

4. **Preprocessing** (Critical for Ridge!)
   - **Standardize features** (Ridge is scale-sensitive)
   - Handle missing values
   - Remove or treat outliers
   - Encode categorical variables

5. **Hyperparameter Tuning**
   - Use K-fold cross-validation (k=5 or k=10)
   - Test range of λ values: [0.001, 0.01, 0.1, 1, 10, 100, 1000]
   - Select λ with best validation performance

6. **Model Training**
   - Fit Ridge on full training data with optimal λ
   - Verify coefficients are reasonable
   - Check for convergence

7. **Validation**
   - Assess on held-out validation data
   - Calculate R², RMSE, MAE
   - Compare against baseline models

8. **Diagnostics**
   - Residual analysis
   - Check for heteroscedasticity
   - Verify coefficient stability
   - Ensure no data leakage

9. **Final Testing**
   - Evaluate on completely unseen test data
   - This gives unbiased performance estimate
   - Only do this ONCE at the end

10. **Deployment**
    - Save model (pickle, joblib)
    - Create prediction pipeline
    - Monitor performance over time
    - Retrain periodically

### Common Pitfalls to Avoid

- Forgetting to standardize features
- Using the same data for tuning and evaluation
- Over-interpreting coefficient magnitudes
- Ignoring domain knowledge in feature selection
- Not validating on truly held-out data
- Applying Ridge when interpretability requires sparse models

## Software Implementation

### Python (scikit-learn)

```python
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Cross-validation to find optimal lambda
alphas = [0.001, 0.01, 0.1, 1, 10, 100]
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_scaled, y_train)

# Train final model
ridge = Ridge(alpha=ridge_cv.alpha_)
ridge.fit(X_scaled, y_train)
```

### R

```R
library(glmnet)

# Fit ridge regression (alpha = 0)
ridge_model <- cv.glmnet(x = X_train, y = y_train, 
                         alpha = 0, standardize = TRUE)

# Optimal lambda
optimal_lambda <- ridge_model$lambda.min

# Predictions
predictions <- predict(ridge_model, newx = X_test, 
                      s = optimal_lambda)
```

## Performance Optimization

### When Ridge May Underperform

- When true model is sparse (Lasso or Elastic Net preferred)
- When features are already uncorrelated (OLS may suffice)
- When non-linear relationships dominate (consider kernel methods)
- When interpretability requires exact zeros (use Lasso)

### Ensemble Approaches

Ridge can be combined with:
- **Stacking**: Use Ridge as a meta-learner
- **Bagging**: Reduce variance further through bootstrap aggregation
- **Model averaging**: Combine multiple Ridge models with different λ

## Summary

Ridge regression is a fundamental tool in the data scientist's toolkit, offering:
- Robust handling of multicollinearity
- Reduced overfitting through regularization
- Stable and interpretable coefficient estimates
- Computational efficiency with closed-form solution
- Strong theoretical foundations

The key to successful application lies in proper data preprocessing, thoughtful hyperparameter tuning, and understanding when Ridge is the appropriate choice versus alternative regularization methods.

## Quick Reference Guide

### When to Use Ridge Regression

✅ **Use Ridge when:**
- Features are highly correlated (multicollinearity)
- Number of features is large
- OLS shows overfitting (high training acc, low test acc)
- You want to keep all features in the model
- Prediction accuracy is the primary goal

❌ **Don't use Ridge when:**
- You need sparse models (use Lasso instead)
- You need exact feature selection (use Lasso instead)
- Features are already uncorrelated and OLS works well
- You need both regularization AND feature selection (use Elastic Net)

### Key Formulas at a Glance

| Component | Formula |
|-----------|---------|
| **Ridge Cost** | J(β) = Σ(h(x) - y)² + λΣ(slope)² |
| **Closed Form** | β̂ = (X'X + λI)⁻¹X'y |
| **R-Squared** | R² = 1 - Σ(y - ŷ)² / Σ(y - ȳ)² |
| **Adjusted R²** | Adj R² = 1 - [(1-R²)(N-1)/(N-P-1)] |
| **MSE** | MSE = (1/n)Σ(y - ŷ)² |
| **MAE** | MAE = (1/n)Σ\|y - ŷ\| |
| **RMSE** | RMSE = √MSE |

### Algorithm Comparison Cheat Sheet

| Algorithm | Penalty | Feature Selection | Coefficient Path | Best For |
|-----------|---------|-------------------|------------------|----------|
| **OLS** | None | No | N/A | Small, uncorrelated features |
| **Ridge** | L2 (Σβ²) | No | Smooth shrinkage | Multicollinearity, all features important |
| **Lasso** | L1 (Σ\|β\|) | Yes | Abrupt drop to zero | Feature selection, sparse models |
| **Elastic Net** | L1 + L2 | Yes | Combined behavior | Both regularization & selection |

### Hyperparameter Tuning Checklist

- [ ] Split data: 70% train, 15% validation, 15% test
- [ ] Standardize features (fit on training only!)
- [ ] Choose CV method (K-fold with k=5 or k=10 recommended)
- [ ] Define lambda range: [0.001, 0.01, 0.1, 1, 10, 100, 1000]
- [ ] Run cross-validation
- [ ] Select lambda with best validation performance
- [ ] Train final model on full training data
- [ ] Evaluate on test set (only once!)
- [ ] Check R², RMSE, and residual plots

### Common Mistakes to Avoid

1. ❌ Forgetting to standardize features
2. ❌ Using test data for hyperparameter tuning
3. ❌ Not using cross-validation
4. ❌ Applying same preprocessing to test data (fit on train only)
5. ❌ Interpreting coefficient magnitudes without considering scale
6. ❌ Using Ridge when you actually need feature selection
7. ❌ Setting λ = 0 (that's just OLS!)
8. ❌ Not checking for multicollinearity first

---

**References for Further Reading:**
- Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). An Introduction to Statistical Learning
