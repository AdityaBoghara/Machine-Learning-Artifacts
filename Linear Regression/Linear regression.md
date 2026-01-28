
# Linear Regression 

## Overview
Linear regression is a **supervised learning** algorithm used to model the relationship between a **continuous dependent variable** and one or more **independent variables** by fitting a linear function to observed data. The goal is to learn parameters that produce predictions with minimal error.

---

## Model Formulation

### Simple Linear Regression
$\hat{y} = \beta_0 + \beta_1 x$

Where:

- $\hat{y}$ = predicted output 
- $x$ = input feature  
- $\beta_0$ = intercept  
- $\beta_1$= slope (coefficient)  

The error for each data point is:
$\varepsilon_i = y_i - \hat{y}_i$

---

## Objective and Cost Function
The objective is to estimate $\beta_0$ and $\beta_1$ such that predictions are as close as possible to the true values. This is achieved by minimizing the **Mean Squared Error (MSE)** cost function:

$J(\beta_0, \beta_1) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2$

The cost function is **convex** for linear regression, which guarantees a **single global minimum**.

---

## Gradient Descent Optimization
Gradient descent is an **iterative optimization algorithm** used to minimize the cost function by updating parameters in the direction of the negative gradient.

### Gradients
$\frac{\partial J}{\partial \beta_0} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)$

$\frac{\partial J}{\partial \beta_1} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)x_i$

### Update Rules
$\beta_0 := \beta_0 - \alpha \frac{\partial J(\beta_0, \beta_1)}{\partial \beta_0}$

$\beta_1 := \beta_1 - \alpha \frac{\partial J(\beta_0, \beta_1)}{\partial \beta_1}$

Where:
- $\alpha$ = learning rate (controls step size)

This process is repeated until the algorithm converges.

---

## Convergence Function (Gradient Descent)

### Definition
The **convergence function** defines how parameters are updated iteratively until the cost function reaches its minimum.

### General Update Rule
$\beta_j := \beta_j - \alpha \frac{\partial J(\beta)}{\partial \beta_j}$

---

## Convergence Process
1. Initialize parameters $\beta$
2. Compute predictions and cost $J(\beta)$
3. Compute gradients
4. Update parameters
5. Repeat until convergence

---

## When Convergence Is Achieved
Convergence occurs when:
- The change in the cost function becomes negligible, or  
- Parameter updates approach zero  

At this point, the model has reached the **global minimum** of the cost function.

---

## Intuition
- Gradient descent moves parameters toward the **steepest decrease** in error  
- The learning rate controls **speed vs. stability**
  - Too large → overshooting or divergence  
  - Too small → slow convergence  

---

## Final Outcome
At convergence:
- The cost function is minimized  
- Parameters are stable  
- The best-fit line (or hyperplane for multiple features) is obtained  

This approach scales efficiently to large datasets and extends naturally to **multiple linear regression** using vectorized operations.

---

## Model Assumptions, Failure Modes, and Practical Judgment

This section captures how linear regression is evaluated and reasoned about in professional and interview settings, beyond mathematical correctness.

---

## Core Assumptions

Linear regression relies on several assumptions that are often **not enforced by the algorithm itself**. Violations typically degrade performance silently.

- **Linearity**  
  The relationship between features and the target is assumed to be linear. Non-linear patterns lead to systematic bias in predictions.

- **Independence of observations**  
  Each data point is assumed to be independent. Correlated observations (e.g., time series without controls) invalidate error estimates.

- **Homoscedasticity**  
  Error variance is assumed to be constant across all values of the input features. Heteroscedasticity leads to unreliable confidence intervals and unstable coefficients.

- **Low multicollinearity**  
  Features are assumed not to be highly correlated. Strong multicollinearity inflates variance of coefficients and makes interpretations unreliable.

- **Error distribution (practical, not strict)**  
  Errors are ideally symmetric and centered around zero. Heavy-tailed or skewed errors increase sensitivity to outliers.

---

## Failure Modes

Linear regression does not fail loudly. Its most dangerous behavior is *appearing to work* while producing misleading results.

- **Outlier sensitivity**  
  The squared error cost function disproportionately weights large errors, making the model fragile in the presence of outliers.

- **Silent assumption violations**  
  The model will still converge even when assumptions are violated, producing coefficients that look reasonable but generalize poorly.

- **Non-linear relationships**  
  Linear regression underfits when the true relationship is non-linear, even with strong signal present in the data.

- **Multicollinearity instability**  
  Coefficients may flip signs or vary widely across datasets while predictions remain similar, reducing interpretability.

- **Misuse for classification**  
  Outputs are unbounded and cannot represent probabilities, making linear regression unsuitable for classification tasks.

---

## Tradeoffs and Design Judgment

- **Why linear regression is used**  
  - Fast to train  
  - Highly interpretable  
  - Strong baseline for continuous targets  

- **Why it is rarely the final model**  
  - Limited expressive power  
  - High sensitivity to data issues  
  - Assumptions rarely hold fully in real systems  

- **Optimization choice**  
  Gradient descent is preferred over closed-form solutions when:
  - Dataset size is large  
  - Feature count is high  
  - Memory constraints make matrix inversion impractical  

---

## Professional Positioning

In practice, linear regression is treated as:
- A **diagnostic tool** to understand relationships  
- A **baseline model** for performance comparison  
- A **sanity check** before deploying more complex models  

It is not treated as a default production solution unless domain constraints explicitly justify its assumptions.

---

- **When it fails**  
  Linear regression fails when the underlying relationship is non-linear, when features are highly correlated, when outliers dominate the data, or when error variance is not constant.

- **Why it fails quietly**  
  The optimization process still converges and produces coefficients even when assumptions are violated. Because there are no built-in safeguards, the model can appear correct while producing unreliable or misleading results.

- **What I would check before trusting it**  
  I would analyze residual plots for non-linearity and heteroscedasticity, check feature correlations to detect multicollinearity, examine outliers and leverage points, and validate performance on unseen data.

- **What model I would consider next**  
  Based on the observed failure mode, I would move to polynomial features, regularized linear models such as Ridge or Lasso to address multicollinearity, or non-linear models like decision trees or gradient boosting when linear assumptions break down.

This framing demonstrates applied reasoning and decision-making, which interviewers value more than memorization.


