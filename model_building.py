import pandas as pd
from data_preprocessing import preprocess_data
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_random_variables(df):
    # Correlation of random variables with target
    corr_random1 = df['random_variable1'].corr(df['equipment_energy_consumption'])
    corr_random2 = df['random_variable2'].corr(df['equipment_energy_consumption'])
    print(f"Correlation of random_variable1 with target: {corr_random1:.4f}")
    print(f"Correlation of random_variable2 with target: {corr_random2:.4f}")
    
    # Plot distributions
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=df['random_variable1'], y=df['equipment_energy_consumption'])
    plt.title('random_variable1 vs Equipment Energy Consumption')
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=df['random_variable2'], y=df['equipment_energy_consumption'])
    plt.title('random_variable2 vs Equipment Energy Consumption')
    plt.tight_layout()
    plt.show()

def tune_and_evaluate_model(model, param_grid, X_train, y_train, X_test, y_test):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.2f}\n")
    return best_model, rmse, mae, r2

def build_and_evaluate_models(df, exclude_random_vars=False):
    # Define features and target
    if exclude_random_vars:
        X = df.drop(columns=['timestamp', 'equipment_energy_consumption', 'random_variable1', 'random_variable2'])
    else:
        X = df.drop(columns=['timestamp', 'equipment_energy_consumption'])
    y = df['equipment_energy_consumption']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {}
    
    # Linear Regression (no hyperparameters to tune)
    print("Tuning Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Linear Regression Performance (exclude_random_vars={exclude_random_vars}):")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.2f}\n")
    results['Linear Regression'] = {'model': lr, 'RMSE': rmse, 'MAE': mae, 'R2': r2}
    
    # Random Forest with hyperparameter tuning
    print("Tuning Random Forest...")
    rf = RandomForestRegressor(random_state=42)
    rf_param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    best_rf, rmse_rf, mae_rf, r2_rf = tune_and_evaluate_model(rf, rf_param_grid, X_train, y_train, X_test, y_test)
    results['Random Forest'] = {'model': best_rf, 'RMSE': rmse_rf, 'MAE': mae_rf, 'R2': r2_rf}
    
    # Gradient Boosting with hyperparameter tuning
    print("Tuning Gradient Boosting...")
    gb = GradientBoostingRegressor(random_state=42)
    gb_param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    }
    best_gb, rmse_gb, mae_gb, r2_gb = tune_and_evaluate_model(gb, gb_param_grid, X_train, y_train, X_test, y_test)
    results['Gradient Boosting'] = {'model': best_gb, 'RMSE': rmse_gb, 'MAE': mae_gb, 'R2': r2_gb}
    
    # Select best model based on RMSE
    best_model_name = min(results, key=lambda k: results[k]['RMSE'])
    best_model = results[best_model_name]['model']
    print(f"Best model: {best_model_name} (exclude_random_vars={exclude_random_vars})")
    
    # Feature importance for tree-based models
    if best_model_name in ['Random Forest', 'Gradient Boosting']:
        feature_importances = best_model.feature_importances_
        features = X.columns
        fi_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
        fi_df = fi_df.sort_values(by='Importance', ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=fi_df)
        plt.title(f'Feature Importance - {best_model_name} (exclude_random_vars={exclude_random_vars})')
        plt.tight_layout()
        plt.show()
    
    return best_model_name, results[best_model_name]

if __name__ == "__main__":
    data_file = "data/data.csv"
    df_clean = preprocess_data(data_file)
    print("Analyzing random variables:")
    analyze_random_variables(df_clean)
    print("\nModel performance including random variables:")
    build_and_evaluate_models(df_clean, exclude_random_vars=False)
    print("\nModel performance excluding random variables:")
    build_and_evaluate_models(df_clean, exclude_random_vars=True)
