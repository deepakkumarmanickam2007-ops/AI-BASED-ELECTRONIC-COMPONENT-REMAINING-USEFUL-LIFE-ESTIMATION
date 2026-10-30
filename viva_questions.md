# Viva Questions and Short Answers

1. **What is RUL?** Remaining useful life is the estimated operating time or cycles until a component reaches a defined end-of-life condition.
2. **Why use machine learning?** It can learn nonlinear relationships between condition measurements and observed lifetime from examples.
3. **Why is this a regression problem?** The target `rul_cycles` is a continuous numerical quantity.
4. **What is the target variable?** `rul_cycles`, the simulated cycles remaining until EOL.
5. **Why are time-series features useful?** Trends and rolling means can reveal gradual drift that a single reading may miss.
6. **What is ESR?** Equivalent series resistance; an increase can indicate degradation in some capacitor technologies.
7. **Why compare Random Forest and Gradient Boosting?** They are tree-based nonlinear regressors with different ensemble strategies.
8. **What is MAE?** Mean absolute error, the average absolute difference between predictions and true values.
9. **What is RMSE?** Root mean squared error; it penalizes large errors more strongly than MAE.
10. **What is R²?** A metric comparing model residual variation with a mean-target baseline; it can be negative.
11. **Why split by component ID?** To test generalization to unseen components and avoid placing the same component in both sets.
12. **What is feature engineering?** Creating useful predictors from raw measurements, such as stress indices and trends.
13. **What does feature importance mean?** It describes how much the fitted tree model uses a feature; it does not prove causality.
14. **Why is the dataset synthetic?** It ensures the project runs reproducibly without needing special hardware or data access.
15. **Can this model be used for actual maintenance?** No—not without training and validation on representative real component data.
16. **What are the main limitations?** Synthetic data, simplified EOL assumptions, lack of uncertainty estimates, and limited temporal context for one-row inputs.
17. **What is predictive maintenance?** Maintenance guided by the estimated condition or failure risk of equipment.
18. **How can the project be improved?** Use measured aging data, cross-validation, uncertainty estimation, and domain-specific thresholds.
19. **Why is Gradient Boosting sequential?** Each new tree is fitted to improve the current ensemble by reducing its residual error.
20. **What does the dashboard do?** It shows data and estimates RUL for user-entered operating measurements using the saved model.
