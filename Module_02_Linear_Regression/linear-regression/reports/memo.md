# Two-Feature Linear Regression Extension: Analysis Memo

This memo evaluates extending our baseline linear regression model from a single predictor (BMI) to a bivariate specification incorporating age. Adding age significantly improved explanatory power, lifting R^2 from approximately 0.039 to roughly 0.117—an increase of nearly 7.8 percentage points. Concurrently, the Root Mean Squared Error decreased by about $500, demonstrating that patient age captures variance in medical expenses that body mass index alone leaves unexplained.

The parameter estimates derived via the closed-form Normal Equation and standardized Gradient Descent converged to identical values (w0 ≈ -6437.35, w1 ≈ 333.39, and w2 ≈ 241.90). Theoretically, these two methods should agree because the Ordinary Least Squares Mean Squared Error loss surface is strictly convex and smooth with a single global minimum. Given sufficient training epochs and proper feature scaling, gradient descent will inevitably converge to this analytical minimum.

For non-technical stakeholders, the age coefficient (w2 ≈ $241.90) indicates that, holding BMI constant, an individual's expected annual medical expenses increase by approximately $242 for each additional year of age. The positive sign directly confirms that aging elevates insurance claim risk across comparable body types.

Despite these improvements, this two-feature model is not ready for production deployment. Explaining only 11.7% of total variance leaves nearly 88% of cost fluctuation unaccounted for, resulting in large typical prediction errors (MAE ≈ $9,032). Furthermore, critical risk determinants like smoking status are omitted, which introduces substantial omitted-variable bias and misestimates individual risk.
