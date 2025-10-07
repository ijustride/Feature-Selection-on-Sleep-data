import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import SplineTransformer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv("Feature-Selection-on-Sleep-data/data/updated_data_1.csv")

# Features and target
features = ['Occupation', 'Sleep Duration', 'Quality of Sleep', 
            'Physical Activity Level', 'Age']
X = data[features]
y = data['Stress Level']

# Configure OneHotEncoder to handle unknown categories
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ['Occupation']),
        ('num', StandardScaler(), ['Sleep Duration', 'Quality of Sleep', 
                                  'Physical Activity Level', 'Age'])
    ])

# Model pipelines
lr_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

svr_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', SVR(kernel='rbf', degree=2, C=1.0))
])

spline_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('spline', SplineTransformer(degree=3, n_knots=5)),
    ('regressor', LinearRegression())
])

# Bootstrapping setup
B = 500
n_features = len(features)

# Enhanced results storage with R²
results = {
    'LinearRegression': {
        'importance': np.zeros((B, n_features)),
        'MSE': np.zeros(B),
        'MAE': np.zeros(B),
        'R2': np.zeros(B)  # Added R² storage
    },
    'PolynomialSVR': {
        'importance': np.zeros((B, n_features)),
        'MSE': np.zeros(B),
        'MAE': np.zeros(B),
        'R2': np.zeros(B)  # Added R² storage
    },
    'SplineRegression': {
        'importance': np.zeros((B, n_features)),
        'MSE': np.zeros(B),
        'MAE': np.zeros(B),
        'R2': np.zeros(B)  # Added R² storage
    }
}

# Bootstrapping loop
for i in range(B):
    # Resample with replacement
    idx = np.random.choice(len(X), size=len(X), replace=True)
    X_boot = X.iloc[idx]
    y_boot = y.iloc[idx]
    
    # Train models
    lr_pipe.fit(X_boot, y_boot)
    svr_pipe.fit(X_boot, y_boot)
    spline_pipe.fit(X_boot, y_boot)
    
    # Get out-of-bag samples
    oob_idx = np.setdiff1d(np.arange(len(X)), np.unique(idx))
    
    # Skip iteration if too few OOB samples
    if len(oob_idx) < 10:
        continue
        
    X_oob = X.iloc[oob_idx]
    y_oob = y.iloc[oob_idx]
    
    # Evaluate each model
    for model_name, model in zip(
        ['LinearRegression', 'PolynomialSVR', 'SplineRegression'],
        [lr_pipe, svr_pipe, spline_pipe]
    ):
        # Get predictions
        y_pred = model.predict(X_oob)
        
        # Calculate all three metrics
        mse = mean_squared_error(y_oob, y_pred)
        mae = mean_absolute_error(y_oob, y_pred)
        r2 = r2_score(y_oob, y_pred)
        
        # Store results
        results[model_name]['MSE'][i] = mse
        results[model_name]['MAE'][i] = mae
        results[model_name]['R2'][i] = r2
        
        # Calculate permutation importance using R² for consistency
        try:
            r = permutation_importance(
                model, X_oob, y_oob, 
                n_repeats=10,
                scoring='r2',  # Now using R² for importance
                random_state=i,
                n_jobs=-1
            )
            results[model_name]['importance'][i] = r.importances_mean
        except Exception as e:
            print(f"Error in {model_name} iteration {i}: {str(e)}")
            results[model_name]['importance'][i] = np.nan

# =====================
# 1. Feature Importance
# =====================
importance_summary = []
for model_name, model_data in results.items():
    imp = model_data['importance']
    for j, feature in enumerate(features):
        # Remove NaN results
        clean_imp = imp[:, j][~np.isnan(imp[:, j])]
        if len(clean_imp) > 0:
            ci_lower = np.percentile(clean_imp, 2.5)
            ci_upper = np.percentile(clean_imp, 97.5)
            mean_val = np.mean(clean_imp)
            importance_summary.append({
                'Model': model_name,
                'Feature': feature,
                'Mean Importance': mean_val,
                'CI Lower': ci_lower,
                'CI Upper': ci_upper
            })

importance_df = pd.DataFrame(importance_summary)

# Plot feature importance
plt.figure(figsize=(12, 8))
for model in importance_df['Model'].unique():
    model_df = importance_df[importance_df['Model'] == model]
    plt.errorbar(
        model_df['Feature'], model_df['Mean Importance'],
        yerr=[model_df['Mean Importance'] - model_df['CI Lower'], 
              model_df['CI Upper'] - model_df['Mean Importance']],
        fmt='o', label=model, capsize=5
    )

plt.title('Stress Level: Feature Importance (95% CI)')
plt.ylabel('Permutation Importance (ΔR²)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('Feature-Selection-on-Sleep-data/pics/feature_importance.png')
plt.show()

# =========================================
# 2. Model Performance (R², MSE, and MAE)
# =========================================
# Prepare performance data
performance_data = []
metrics = ['R2', 'MSE', 'MAE']

for model_name, model_data in results.items():
    for metric in metrics:
        values = model_data[metric][model_data[metric] != 0]  # Remove unused iterations
        if len(values) > 0:
            mean_val = np.mean(values)
            ci_lower = np.percentile(values, 2.5)
            ci_upper = np.percentile(values, 97.5)
            performance_data.append({
                'Model': model_name,
                'Metric': metric,
                'Mean': mean_val,
                'CI Lower': ci_lower,
                'CI Upper': ci_upper
            })

performance_df = pd.DataFrame(performance_data)

# Create subplots for better visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot each metric separately
for i, metric in enumerate(metrics):
    ax = axes[i]
    metric_df = performance_df[performance_df['Metric'] == metric]
    
    # Determine bar positions
    models = metric_df['Model'].unique()
    x_pos = np.arange(len(models))
    
    # Plot bars
    bars = ax.bar(x_pos, metric_df['Mean'], yerr=[
        metric_df['Mean'] - metric_df['CI Lower'], 
        metric_df['CI Upper'] - metric_df['Mean']],
        capsize=5, alpha=0.7
    )
    
    # Add data labels
    for bar, (_, row) in zip(bars, metric_df.iterrows()):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, 
                f'{height:.3f}\n({row["CI Lower"]:.3f}-{row["CI Upper"]:.3f})',
                ha='center', va='bottom', fontsize=9)
    
    ax.set_title(f'{metric} Comparison')
    ax.set_ylabel(metric)
    ax.set_xlabel('Model')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=45)
    
    # Special handling for R²
    if metric == 'R2':
        ax.set_ylim(0, 1)  # R² typically between 0-1

plt.tight_layout()
plt.savefig('Feature-Selection-on-Sleep-data/pics/model_performance.png')
plt.show()

# ==============================
# 3. Print Summary Statistics
# ==============================
print("\n=== Model Performance Summary ===")
for model in performance_df['Model'].unique():
    print(f"\n{model}:")
    model_df = performance_df[performance_df['Model'] == model]
    for _, row in model_df.iterrows():
        print(f"  {row['Metric']}: {row['Mean']:.4f} (95% CI: {row['CI Lower']:.4f}-{row['CI Upper']:.4f})")

print("\n=== Top Features ===")
top_features = importance_df.sort_values('Mean Importance', ascending=False).head(5)
print(top_features[['Feature', 'Model', 'Mean Importance']].to_string(index=False))

# ==================================
# 4. Create Performance Comparison Table
# ==================================
# Create a publication-ready table
performance_table = performance_df.pivot_table(
    index='Model', 
    columns='Metric', 
    values=['Mean', 'CI Lower', 'CI Upper']
)

# Format table nicely
formatted_table = pd.DataFrame()
for metric in metrics:
    for model in performance_df['Model'].unique():
        mean_val = performance_table.loc[model, ('Mean', metric)]
        ci_lower = performance_table.loc[model, ('CI Lower', metric)]
        ci_upper = performance_table.loc[model, ('CI Upper', metric)]
        formatted_table.loc[model, metric] = f"{mean_val:.3f} ({ci_lower:.3f}-{ci_upper:.3f})"

print("\n=== Performance Comparison Table ===")
print(formatted_table)