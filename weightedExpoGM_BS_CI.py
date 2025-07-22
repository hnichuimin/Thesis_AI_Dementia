import numpy as np
import pandas as pd

# data
df = pd.DataFrame({
    'Year': [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'Publications': [114, 134, 206, 336, 532, 803, 1090, 1319, 1753, 2286]
})

# Calculate growth rates
df['Growth_Rate'] = df['Publications'].pct_change()

# Bootstrap confidence intervals
n_iterations = 1000
bootstrap_rates = []
rng = np.random.default_rng()

for _ in range(n_iterations):
    # Resample with replacement
    sample = rng.choice(df['Growth_Rate'].dropna(), size=len(df)-1, replace=True)
    # Weighted average
    recent = sample[-3:].mean() if len(sample) >=3 else sample.mean()
    historical = sample.mean()
    bootstrap_rates.append(0.7*recent + 0.3*historical)

# Get percentiles for CI
lower_ci = np.percentile(bootstrap_rates, 2.5)
upper_ci = np.percentile(bootstrap_rates, 97.5)
effective_growth = np.mean(bootstrap_rates)

print(f"Growth rate: {effective_growth:.1%} (95% CI: {lower_ci:.1%} to {upper_ci:.1%})")

# Forecasting function with CI
def predict_with_ci(base_count, years, growth, lower, upper):
    forecasts = {
        'year': [],
        'mean': [],
        'lower': [],
        'upper': []
    }
    for y in range(1, years+1):
        forecasts['year'].append(2024 + y)
        forecasts['mean'].append(int(base_count * (1 + growth)**y))
        forecasts['lower'].append(int(base_count * (1 + lower)**y))
        forecasts['upper'].append(int(base_count * (1 + upper)**y))
    return pd.DataFrame(forecasts)

# Generate predictions
forecasts = predict_with_ci(2286, 3, effective_growth, lower_ci, upper_ci)
print("\nForecasted Publications:")
print(forecasts.to_string(index=False))

# Validation
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# Prepare data
years = df['Year'].values
pubs = df['Publications'].values

# Define validation parameters
min_train_size = 5  # Minimum training set size (first 5 years)

# Store results - Initialize as lists
actuals = []
predictions = []
mae_values = []
mape_values = []

# Walk-forward validation
for i in range(min_train_size, len(df)):
    try:
        # Split data
        train_years, train_pubs = years[:i], pubs[:i]
        test_year, test_pub = years[i], pubs[i]

        # Calculate growth rates from training set
        growth_rates = np.diff(train_pubs) / train_pubs[:-1]

        # Weighted growth rate
        recent_growth = growth_rates[-3:].mean() if len(growth_rates) >= 3 else growth_rates.mean()
        historical_growth = growth_rates.mean()
        effective_growth = 0.7 * recent_growth + 0.3 * historical_growth

        # Make prediction
        pred_pub = train_pubs[-1] * (1 + effective_growth)

        # Store metrics
        actuals.append(test_pub)
        predictions.append(pred_pub)
        mae_values.append(mean_absolute_error([test_pub], [pred_pub]))
        mape_values.append(mean_absolute_percentage_error([test_pub], [pred_pub]))

    except Exception as e:
        print(f"Error at year {years[i] if i < len(years) else 'unknown'}: {str(e)}")
        continue

# Ensure all arrays have the same length
assert len(actuals) == len(predictions) == len(mae_values) == len(mape_values), "Array length mismatch"

# Convert to DataFrame
results = pd.DataFrame({
    'Year': years[min_train_size:min_train_size + len(actuals)],
    'Actual': actuals,
    'Predicted': predictions,
    'Absolute_Error': mae_values,
    'Percentage_Error': mape_values
})

# Print average performance
if len(mae_values) > 0:
    print(f"\nValidation Results:")
    print(f"Mean Absolute Error: {np.mean(mae_values):.1f} papers")
    print(f"Mean Absolute Percentage Error: {np.mean(mape_values):.1%}")
    print("\nDetailed Results:")
    print(results.to_string(index=False))
else:
    print("No validation results - check data or loop conditions")

# plot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.figure(figsize=(12, 7))

# Data Preparation
# Define forecast years (2025-2027)
forecast_years = [2025, 2026, 2027]
# Get validation years from your results DataFrame (2020-2024)
val_years = results['Year'].values

# Plotting
# 1. Historical Data (2015-2024)
plt.plot(df['Year'], df['Publications'],
        'o-', label='Historical Data (2015-2024)',
        color='#1f77b4', linewidth=2.5, markersize=8)

# 2. Connection Line (2024 to 2025)
plt.plot([df['Year'].iloc[-1], forecast_years[0]],
        [df['Publications'].iloc[-1], forecasts['mean'].iloc[0]],
        ':', color='#ff7f0e', alpha=0.7, linewidth=1.5,
        label='Model Projection')

# 3. Forecast (2025-2027)
plt.plot(forecast_years, forecasts['mean'],
        's--', label='Forecast (2025-2027)',
        color='#ff7f0e', linewidth=2.5, markersize=8)

# 4. Confidence Interval
plt.fill_between(forecast_years,
               forecasts['lower'],
               forecasts['upper'],
               color='#ff7f0e', alpha=0.15,
               label='95% Confidence Interval')

# 5. Validation Markers
plt.plot(val_years, results['Predicted'],
        'X', label='Backtest Validation',
        color='#2ca02c', markersize=10, markeredgewidth=1.5)

# Annotations
plt.title('AI in Dementia Research Publications Forecast\nExponential Growth Model Projection',
         fontsize=15, pad=20, fontweight='bold')
plt.xlabel('Year', fontsize=13, labelpad=10)
plt.ylabel('Number of Publications', fontsize=13, labelpad=10)
plt.xticks(np.arange(2015, 2028, 2), fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(0, 9500)

# Growth rate annotation
plt.annotate(f'Current Growth Rate: {effective_growth:.1%}',
            xy=(2023.5, df['Publications'].iloc[-2]),
            xytext=(30, 60), textcoords='offset points',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
            bbox=dict(boxstyle="round", fc="white", ec="#1f77b4", alpha=0.9),
            fontsize=11)

# Model performance annotation
plt.annotate('Model Validation\n(MAE: 117.8 papers)\n(MAPE: 9.1%)',
            xy=(2021, results['Predicted'].iloc[1]),
            xytext=(-80, 30), textcoords='offset points',
            arrowprops=dict(arrowstyle="->", color='#2ca02c'),
            bbox=dict(boxstyle="round", fc="white", ec="#2ca02c", alpha=0.9),
            fontsize=10, ha='center')

# Legend
handles, labels = plt.gca().get_legend_handles_labels()
order = [0, 4, 1, 2, 3] 
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
          loc='upper left', framealpha=1, fontsize=11)

plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

