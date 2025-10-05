import csv

import numpy as np
# from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
# from collections import Counter
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import train_test_split

import seaborn as sns
from distfit import distfit

from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def get_law(feature):
    data = np.array(feature)

    model = distfit()
    model.fit_transform(data)
    model.plot()
    print(model.summary)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"R²: {r2_score(y_test, y_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    # return r2_score(y_test, y_pred), mean_squared_error(y_test, y_pred), mean_absolute_error(y_test, y_pred)

def plot_cov_matrix(data):
    pearson_matrix = data.corr(method='pearson')
    # Set up the plot with LARGER FIGURE SIZE
    plt.figure(figsize=(10, 8))  # Increase width/height as needed

    # Custom colormap (Red -> White -> Blue)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    # Create heatmap with BIGGER SQUARES and CLEARER TEXT
    heatmap = sns.heatmap(
        round(pearson_matrix, 2),
        annot=True,                # Show values
        annot_kws={'size': 12},    # Larger annotation font
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        square=True,               # Ensures squares stay proportional
        cbar_kws={'shrink': 0.8},  # Adjust colorbar size
    )

    # Improve readability:
    plt.xticks(rotation=45, ha='right', fontsize=12)  # Rotate x-labels
    plt.yticks(rotation=0, fontsize=12)               # Align y-labels
    plt.title("Pearson Correlation Matrix", fontsize=16, pad=20)

    # Tight layout to prevent clipping
    plt.tight_layout()
    plt.show()

def plot_spline_importance(X, y, feature_names, n_knots=5, degree=3, n_repeats=30):
    # Standardize data
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = np.array(y)
    y = (y - y.mean()) / y.std()
    
    # Create spline pipeline
    spline = SplineTransformer(n_knots=n_knots, degree=degree, include_bias=False)
    model = make_pipeline(spline, LinearRegression())
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    # Compute permutation importance
    result = permutation_importance(
        model, X_test, y_test, 
        n_repeats=n_repeats,
        random_state=42,
        scoring='neg_mean_squared_error'
    )
    
    # Sort features by importance (descending order)
    sorted_idx = np.argsort(abs(result.importances_mean))[::-1]  # [::-1] reverses for descending
    sorted_names = np.array(feature_names)[sorted_idx]
    sorted_means = abs(result.importances_mean[sorted_idx])
    sorted_stds = abs(result.importances_std[sorted_idx])
    
    evaluate_model(model, X_test, y_test)
    # Plot (vertical bars)
    plt.figure(figsize=(10, 6))
    plt.bar(
        sorted_names,
        sorted_means,
        yerr=sorted_stds,
        color='skyblue',
        alpha=0.7
    )
    
    plt.title("Permutation Importance Spline Regression", pad=15)
    plt.ylabel("Absolute Mean MSE Increase")
    plt.xticks(rotation=45, ha='right')  # Rotate x-labels for readability
    plt.grid(True, axis='y', alpha=0.2)
    
    plt.tight_layout()
    plt.show()

def plot_spline_with_lines(X, y, feature_names):
    # Create figure with GridSpec
    n_features = X.shape[1]
    fig = plt.figure(figsize=(5*n_features + 1.5, 4))  # Extra width for colorbar
    
    # Adjust layout: 90% width for plots, 10% for colorbar
    gs = GridSpec(1, n_features + 1, width_ratios=[1]*n_features + [0.1])
    axes = [fig.add_subplot(gs[i]) for i in range(n_features)]
    cax = fig.add_subplot(gs[-1])  # Colorbar axis
    
    # Get counts for density coloring
    unique_values, counts = np.unique(y, return_counts=True)
    value_to_count = dict(zip(unique_values, counts))
    y_counts = np.array([value_to_count[val] for val in y])
    
    # Find global min/max for consistent color scaling
    count_min, count_max = min(y_counts), max(y_counts)
    
    for i, (ax, feature) in enumerate(zip(axes, feature_names)):
        # --- Scatter Plot with Density ---
        scatter = ax.scatter(
            X[:, i], 
            y, 
            c=y_counts, 
            cmap='viridis', 
            s=y_counts*10, 
            alpha=0.6,
            label='Data',
            vmin=count_min, 
            vmax=count_max  # Consistent color scaling
        )
        
        spline = SplineTransformer(n_knots=5, degree=3)
        model = make_pipeline(spline, LinearRegression())
        model.fit(X, y)
        
        x_vals = np.linspace(X[:, i].min(), X[:, i].max(), 100).reshape(-1, 1)
        X_temp = np.zeros((100, X.shape[1]))
        X_temp[:, i] = x_vals.flatten()
        y_pred = model.predict(X_temp)
        
        ax.plot(
            x_vals, 
            y_pred, 
            color='red', 
            linewidth=2
        )
        
        # --- Subplot Customization ---
        ax.set_title(feature)
        ax.set_xlabel(f"Standardized {feature}")
        if i == 0:
            ax.set_ylabel("Standardized Target")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    
    # --- Colorbar ---
    fig.colorbar(
        scatter, 
        cax=cax,
        label='Value Frequency',
        orientation='vertical'
    )
    
    plt.suptitle("Spline Regression with Value Density", y=1)
    plt.tight_layout()
    plt.show()

def plot_svr_scatter_with_lines(X, y, feature_names, kernel='rbf', C=10, epsilon=0.01):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y = np.array(y)
    y_scaled = (y - y.mean()) / y.std()
    
    # Create figure with GridSpec
    n_features = X.shape[1]
    fig = plt.figure(figsize=(5*n_features + 1.5, 4))  # Extra width for colorbar
    
    # Adjust layout: 90% width for plots, 10% for colorbar
    gs = GridSpec(1, n_features + 1, width_ratios=[1]*n_features + [0.1])
    axes = [fig.add_subplot(gs[i]) for i in range(n_features)]
    cax = fig.add_subplot(gs[-1])  # Colorbar axis
    
    # Get counts for density coloring
    unique_values, counts = np.unique(y, return_counts=True)
    value_to_count = dict(zip(unique_values, counts))
    y_counts = np.array([value_to_count[val] for val in y])
    
    # Find global min/max for consistent color scaling
    count_min, count_max = min(y_counts), max(y_counts)
    
    for i, (ax, feature) in enumerate(zip(axes, feature_names)):
        # --- Scatter Plot with Density ---
        scatter = ax.scatter(
            X_scaled[:, i], 
            y_scaled, 
            c=y_counts, 
            cmap='viridis', 
            s=y_counts*10, 
            alpha=0.6,
            label='Data',
            vmin=count_min, 
            vmax=count_max  # Consistent color scaling
        )
        
        
        # --- SVR Regression Line ---
        svr = SVR(kernel=kernel, C=C, epsilon=epsilon)
        svr.fit(X_scaled[:, i:i+1], y_scaled)
        
        x_vals = np.linspace(X_scaled[:, i].min(), X_scaled[:, i].max(), 100)
        y_pred = svr.predict(x_vals.reshape(-1, 1))
        
        
        ax.plot(
            x_vals, 
            y_pred, 
            color='red', 
            linewidth=2, 
            label=f'SVR ({kernel})'
        )
        
        # --- Subplot Customization ---
        ax.set_title(feature)
        ax.set_xlabel(f"Standardized {feature}")
        if i == 0:
            ax.set_ylabel("Standardized Target")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    
    # --- Colorbar ---
    fig.colorbar(
        scatter, 
        cax=cax,
        label='Value Frequency',
        orientation='vertical'
    )
    
    plt.suptitle("Polynomial Regression with Value Density", y=1)
    plt.tight_layout()
    plt.show()
    

def plot_polyreg_SVR(X, y):
    # Standardize features (critical for SVR)
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y = np.array(y)
    y_scaled = (y - y.mean()) / y.std()
    # Train/test split (shuffle=True by default)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # Initialize SVR (adjust hyperparameters)
    svr = SVR(kernel='rbf', C=10, epsilon=0.01)
    svr.fit(X_train, y_train)

    # Compute permutation importance (repeat multiple times)
    result = permutation_importance(
        svr, 
        X_test, 
        y_test, 
        n_repeats=30,  # More repetitions for stability
        random_state=42,
        scoring='neg_mean_squared_error'
    )

    # Get importance scores (absolute values)
    importance = np.abs(result.importances_mean)
    std = result.importances_std
    feature_names = ['job_activity', 'sleep_duration', 'Sleep Quality', 'physical_activity', 'age']

    # Sort features by importance (descending)
    sorted_idx = np.argsort(importance)[::-1]

    
    evaluate_model(svr, X_test, y_test)
    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(
        range(X.shape[1]), 
        importance[sorted_idx], 
        yerr=std[sorted_idx], 
        tick_label=np.array(feature_names)[sorted_idx],
        color='skyblue',
        capsize=5
    )
    plt.title("Polynomial Regression Feature Importance", fontsize=16)
    plt.xlabel("Features", fontsize=14)
    plt.ylabel("Permutation Importance (MSE Increase)", fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    
    plt.show()


def plot_regres_feature(X, y):
    print("################################################################")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Mean=0, Std=1

    y = np.array(y)
    y_scaled = (y - y.mean()) / y.std()
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    # --- Step 4: Fit Linear Regression ---
    model = LinearRegression()
    model.fit(X_train, y_train)

    # --- Step 5: Extract Coefficients ---
    feature_names = ['job_activity', 'sleep_duration', 'Sleep Quality', 'physical_activity', 'age']
    coefficients = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_,
        'Absolute_Coeff': np.abs(model.coef_)
    }).sort_values('Absolute_Coeff', ascending=False)

    print("Feature Importance Linear Regression:")
    print(coefficients)

    # --- Step 6: Check Statistical Significance (p-values) ---
    X_sm = sm.add_constant(X_scaled)  # Required by statsmodels
    model_sm = sm.OLS(y_scaled, X_sm).fit()
    print("\nStatistical Significance (p-values):")
    print(model_sm.summary())

    evaluate_model(model, X_test, y_test)
    # --- Step 7: Visualize Feature Importance ---
    plt.figure(figsize=(8, 4))
    plt.bar(coefficients['Feature'], coefficients['Absolute_Coeff'], color='skyblue')
    plt.title("Feature Importance Linear Regression")
    plt.xlabel("Features")
    plt.ylabel("Magnitude of Impact")

    plt.show()
    



def plot_regression_with_density(X, y):
    # Fit linear regression

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Mean=0, Std=1

    y = np.array(y)
    y_scaled = (y - y.mean()) / y.std()
    
    model = LinearRegression()
    model.fit(X_scaled, y_scaled)
    y_pred = model.predict(X_scaled)
    
    # Get counts for each y value
    unique_values, counts = np.unique(y_scaled, return_counts=True)
    
    # Create a mapping from value to count
    value_to_count = dict(zip(unique_values, counts))
    
    # Map each y to its count
    y_counts = np.array([value_to_count[val] for val in y_scaled])
    # Create plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X.flatten(), y_scaled, c=y_counts, cmap='viridis', 
                         s=y_counts*10, alpha=0.6)
    
    # Plot regression line
    plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')
    
    plt.colorbar(scatter, label='Number of occurrences')
    plt.xlabel('X values')
    plt.ylabel('y values')
    plt.title('Linear Regression with Value Frequency')
    plt.legend()
    plt.show()

def pearson(x, y, feature_names):
    x_ = np.mean(x)
    y_ = np.mean(y)
    
    sum1 = 0
    sum2 = 0
    sum3 = 0

    for i in range(len(x)):
        sum1 += (x[i] - x_)*(y[i] - y_)
    
    for i in range(len(x)):
        sum2 += (x[i] - x_)**2
        sum3 += (y[i] - y_)**2

    r = sum1 / np.sqrt(sum2 * sum3)

    print("Pearson Correlation between "+ f"{feature_names[0]}"+ ' and ' + f"{feature_names[1]}")
    print(r)
    return r

def rank_data(data):
    """Assign ranks to data, handling ties by averaging."""
    sorted_data = sorted(data)
    ranks = [sorted_data.index(x) + 1 for x in data]
    # Handle ties by averaging ranks
    unique_values = set(data)
    for value in unique_values:
        if data.count(value) > 1:
            indices = [i for i, x in enumerate(data) if x == value]
            avg_rank = np.mean([ranks[i] for i in indices])
            for i in indices:
                ranks[i] = avg_rank
    return ranks

def spearman_with_ties(x, y, feature_names):
    """Calculate Spearman's correlation coefficient with ties."""
    # Rank the data
    rank_x = rank_data(x)
    rank_y = rank_data(y)
    
    # Calculate covariance and standard deviations
    cov = np.cov(rank_x, rank_y, bias=True)[0, 1]
    std_x = np.std(rank_x, ddof=0)
    std_y = np.std(rank_y, ddof=0)
    
    # Compute Spearman's rho
    rho = cov / (std_x * std_y)
    print("Spearman Correlation between "+ f"{feature_names[0]}"+ ' and ' + f"{feature_names[1]}")
    print(rho)
    return rho


def read_data(file_path, gender, age, job_list, sleepDur, sleepQuality, physicalAct,
              stressLevel, bmi, systolic, diastolic, heartrate, steps, sleepDisorder):
    with open(file_path, mode ='r')as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
                gender.append(lines[1])
                age.append(lines[2])
                job_list.append(lines[3])
                sleepDur.append(lines[4])
                sleepQuality.append(lines[5])
                physicalAct.append(lines[6])
                stressLevel.append(lines[7])
                bmi.append(lines[8])
                systolic.append(lines[9])
                diastolic.append(lines[10])
                heartrate.append(lines[11])
                steps.append(lines[12])
                sleepDisorder.append(lines[13])

    age.pop(0)
    sleepDur.pop(0)
    sleepQuality.pop(0)
    stressLevel.pop(0)
    job_list.pop(0)
    physicalAct.pop(0)
    gender.pop(0)
    bmi.pop(0)
    systolic.pop(0)
    diastolic.pop(0)
    heartrate.pop(0)
    steps.pop(0)
    sleepDisorder.pop(0)    


    for i in range(len(sleepDur)):
        sleepDur[i] = float(sleepDur[i])
        sleepQuality[i] = float(sleepQuality[i])
        stressLevel[i] = float(stressLevel[i])
        age[i] = float(age[i])
        physicalAct[i] = float(physicalAct[i])
        systolic[i] = float(systolic[i])
        diastolic[i] = float(diastolic[i])
        heartrate[i] = float(heartrate[i])
        steps[i] = float(steps[i])

    for i in range(len(job_list)):
        if file_path == 'Feature-Selection-on-Sleep-data/updated_data_1.csv':
            if job_list[i] == 'Engineer':
                job_list[i] = 1
            if job_list[i] == 'Accountant':
                job_list[i] = 2
            if job_list[i] == 'Scientist':
                job_list[i] = 3
            if job_list[i] == 'Manager':
                job_list[i] = 4
            if job_list[i] == 'Teacher':
                job_list[i] = 5
            if job_list[i] == 'Lawyer':
                job_list[i] = 6
            if job_list[i] == 'Software Engineer':
                job_list[i] = 7
            if job_list[i] == 'Salesperson':
                job_list[i] = 8
            if job_list[i] == 'Doctor':
                job_list[i] = 9
            if job_list[i] == 'Nurse':
                job_list[i] = 10
            if job_list[i] == 'Sales Representative':
                job_list[i] = 11   
        elif file_path == 'Feature-Selection-on-Sleep-data/updated_data_2.csv':
            if job_list[i] == 'Manual Labor':
                job_list[i] = 1
            if job_list[i] == 'Office Worker':
                job_list[i] = 2
            if job_list[i] == 'Retired':
                job_list[i] = 3
            if job_list[i] == 'Student':
                job_list[i] = 4
    
        if bmi[i] == 'Normal':
            bmi[i] = 1
        if bmi[i] == 'Obese':
            bmi[i] = 2
        if bmi[i] == 'Overweight':
            bmi[i] = 3
        if bmi[i] == 'Underweight':
            bmi[i] = 4
        if bmi[i] == 'Normal Weight':
            bmi[i] = 5
        
        if sleepDisorder[i] == '':
            sleepDisorder[i] = 1
        if sleepDisorder[i] == 'Sleep Apnea':
            sleepDisorder[i] = 2
        if sleepDisorder[i] == 'Insomnia':
            sleepDisorder[i] = 3
        
        if gender[i] == 'Male':
            gender[i] = 1
        if gender[i] == 'Female':
            gender[i] = 2
