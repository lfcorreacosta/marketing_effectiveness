# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, acorr_breusch_godfrey
from statsmodels.stats.stattools import durbin_watson
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# %%
# Load the simulated marketing dataset
df = pd.read_csv("data/cruise_marketing_simulated.csv", parse_dates=["date"])

df.head(10)
# %%

# Target and predictors
target = "bookings_Fjords"
media_channels = ["saturated_TV", "saturated_Social", "saturated_Search",
                  "saturated_Display", "saturated_Email"]
promo_features = ["promo_Caribbean", "promo_discount_Caribbean"]
econ_feature = ["econ_index"]

features = media_channels + promo_features + econ_feature

print(features)
# %%

# Pre-modelling exploration
# -----------------------------
print(df[[target]+features].describe())
# %%

# Correlation matrix
plt.figure(figsize=(10,8))
sns.heatmap(df[[target]+features].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
# %%

# Time series plot
plt.figure(figsize=(14,5))
plt.plot(df['date'], df[target], label=target)
plt.title("Bookings Caribbean Over Time")
plt.xlabel("Date")
plt.ylabel("Bookings")
plt.grid(True)
plt.show()
# %%
# Missing values check
print("Missing values per column:\n", df[features+[target]].isnull().sum())

# %%
# Train-test split
# -----------------------------
X = df[features]
y = df[target]
X = sm.add_constant(X)  # add intercept

# 80%-20% split, no shuffling (time series)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# %%

# Multicollinearity check
# -----------------------------
vif_df = pd.DataFrame()
vif_df["feature"] = X_train.columns
vif_df["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
print("VIF per predictor:\n", vif_df)

# %%
#Model estimation
# -----------------------------
# OLS regression
# 5a. OLS regression
ols_model = sm.OLS(y_train, X_train).fit()
print(ols_model.summary())

# 5b. Poisson regression (count model)
poisson_model = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
print(poisson_model.summary())

# %%
# Econometric diagnostics
# -----------------------------
resid = ols_model.resid
exog = ols_model.model.exog

# Heteroscedasticity
bp_test = het_breuschpagan(resid, exog)
white_test = het_white(resid, exog)
print("Breusch-Pagan p-value:", bp_test[1])
print("White test p-value:", white_test[1])

# Autocorrelation
dw_stat = durbin_watson(resid)
bg_test = acorr_breusch_godfrey(ols_model, nlags=5)
print(f"Durbin-Watson: {dw_stat:.3f}")
print("Breusch-Godfrey p-value:", bg_test[1])

# Normality of residuals
sm.qqplot(resid, line='45')
plt.title("QQ Plot of Residuals")
plt.show()

# %%
# Robust standard errors 
# -----------------------------
ols_model_robust = ols_model.get_robustcov_results(cov_type='HC3')  # heteroscedasticity-robust
print(ols_model_robust.summary())

# %%
#  Model performance
# -----------------------------
# OLS predictions
y_pred_ols = ols_model.predict(X_test)
mse_ols = mean_squared_error(y_test, y_pred_ols)
r2_ols = r2_score(y_test, y_pred_ols)
print(f"OLS Test MSE: {mse_ols:.2f}, R^2: {r2_ols:.2f}")

# %%
# Poisson predictions
y_pred_poisson = poisson_model.predict(X_test)
mse_pois = mean_squared_error(y_test, y_pred_poisson)
print(f"Poisson Test MSE: {mse_pois:.2f}")

# %%

#Coefficient interpretation
# -----------------------------
coef_df = pd.DataFrame({
    "Feature": X_train.columns,
    "OLS_Coefficient": ols_model.params,
    "OLS_robust_SE": ols_model_robust.bse,
    "Poisson_Coefficient": poisson_model.params
})
print(coef_df)

# Media channel elasticities plot
plt.figure(figsize=(10,6))
sns.barplot(x="Feature", y="OLS_Coefficient", data=coef_df[coef_df["Feature"].str.contains("saturated")])
plt.title("OLS Estimated Media Elasticities")
plt.show()

# %%
#Residual analysis
# -----------------------------
plt.figure(figsize=(14,6))
plt.plot(df['date'].iloc[X_test.index], y_test - y_pred_ols, label='OLS Residuals', alpha=0.7)
plt.plot(df['date'].iloc[X_test.index], y_test - y_pred_poisson, label='Poisson Residuals', alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuals over Time")
plt.xlabel("Date")
plt.ylabel("Residuals")
plt.legend()
plt.show()

# %%
# Fitted vs Actual
# -----------------------------
plt.figure(figsize=(10,5))
plt.plot(df['date'].iloc[X_test.index], y_test, label='Actual (Test)')
plt.plot(df['date'].iloc[X_test.index], y_pred_ols, label='Predicted OLS (Test)', alpha=0.7)
plt.title("Actual vs Predicted Bookings (Caribbean) - Test Set")
plt.xlabel("Date")
plt.ylabel("Bookings")
plt.legend()
plt.show()

# %%
