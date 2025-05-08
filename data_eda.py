import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import preprocess_data

def perform_eda(df):
    # Correlation matrix
    corr = df.corr()
    plt.figure(figsize=(15, 12))
    sns.heatmap(corr, annot=False, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Matrix')
    plt.show()
    
    # Distribution of target variable
    plt.figure(figsize=(8, 6))
    sns.histplot(df['equipment_energy_consumption'], bins=50, kde=True)
    plt.title('Distribution of Equipment Energy Consumption')
    plt.xlabel('Energy Consumption (Wh)')
    plt.ylabel('Frequency')
    plt.show()
    
    # Scatter plots of some key features vs target
    features_to_plot = ['zone1_temperature', 'zone2_temperature', 'outdoor_temperature', 'lighting_energy']
    for feature in features_to_plot:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df[feature], y=df['equipment_energy_consumption'])
        plt.title(f'{feature} vs Equipment Energy Consumption')
        plt.xlabel(feature)
        plt.ylabel('Energy Consumption (Wh)')
        plt.show()

if __name__ == "__main__":
    data_file = "data/data.csv"
    df_clean = preprocess_data(data_file)
    perform_eda(df_clean)
