import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score
import joblib
import os
import json
import matplotlib.pyplot as plt  # Import matplotlib for plotting

def train_models(data_file):
    """Train regression models and save them to disk."""
    # Load the dataset
    data = pd.read_csv(data_file)
    
    # Drop unnecessary columns
    data.drop(columns=['Customer_Name'], inplace=True)

    # Check for missing values
    if data.isnull().sum().any():
        data.fillna(data.mean(), inplace=True)  # Fill missing values with the mean

    # Define the feature columns
    feature_columns = ['Cleanliness_Rating', 'Comfort_Rating', 'Staff_Performance', 
                       'Facilities_Rating', 'Location_Rating']
    
    # Use Overall Satisfaction Score as the target variable
    target_column = 'Overall_Satisfaction_Score'  # Corrected target column name
    
    # Map Overall Satisfaction Score to categories
    def label_satisfaction(score):
        if score >= 4.5:
            return 'Excellent'
        elif score >= 3.0:
            return 'Good'
        else:
            return 'Bad'

    # Create a new column for satisfaction labels
    data['Satisfaction_Label'] = data[target_column].apply(label_satisfaction)

    # Encode the categorical labels
    le = LabelEncoder()
    data['Satisfaction_Label'] = le.fit_transform(data['Satisfaction_Label'])

    # Split the data into features and target variable
    X = data[feature_columns]
    y = data['Satisfaction_Label']  # Change target variable to satisfaction label

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor()
    }

    # Store results
    results = {}

    # Train models and collect results
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Convert predictions to the nearest integers for accuracy calculation
        predictions_rounded = np.round(y_pred).astype(int)

        # Calculate metrics
        mse = mean_squared_error(y_test, predictions_rounded)
        r2 = r2_score(y_test, predictions_rounded)
        mae = mean_absolute_error(y_test, predictions_rounded)
        accuracy = accuracy_score(y_test, predictions_rounded) * 100

        results[name] = {
            'MSE': mse,
            'R2': r2,
            'MAE': mae,
            'Accuracy (%)': accuracy
        }

    # Create the models directory if it doesn't exist
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)

    # Save the models in the models directory
    for name, model in models.items():
        joblib.dump(model, os.path.join(models_dir, f'{name.lower().replace(" ", "_")}_model.joblib'))

    # Save the scaler for future use
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))

    # Call function to save visualizations
    save_visualizations(results, models_dir)

    return results

def save_visualizations(results, models_dir):
    """Save visualizations of model performance metrics."""
    # Create a bar plot for MSE
    plt.figure(figsize=(10, 5))
    plt.bar(results.keys(), [metrics['MSE'] for metrics in results.values()], color='skyblue')
    plt.title('Mean Squared Error for Each Model')
    plt.xlabel('Model')
    plt.ylabel('Mean Squared Error')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, 'mse_visualization.png'))  # Save the MSE plot
    plt.clf()  # Clear the current figure

    # Create a bar plot for R²
    plt.bar(results.keys(), [metrics['R2'] for metrics in results.values()], color='salmon')
    plt.title('R² Score for Each Model')
    plt.xlabel('Model')
    plt.ylabel('R² Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, 'r2_visualization.png'))  # Save the R² plot
    plt.clf()  # Clear the current figure

    # Create a bar plot for MAE
    plt.bar(results.keys(), [metrics['MAE'] for metrics in results.values()], color='lightgreen')
    plt.title('Mean Absolute Error for Each Model')
    plt.xlabel('Model')
    plt.ylabel('Mean Absolute Error')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, 'mae_visualization.png'))  # Save the MAE plot
    plt.clf()  # Clear the current figure

    # Create a bar plot for Accuracy
    plt.bar(results.keys(), [metrics['Accuracy (%)'] for metrics in results.values()], color='gold')
    plt.title('Accuracy for Each Model')
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, 'accuracy_visualization.png'))  # Save the Accuracy plot
    plt.clf()  # Clear the current figure

def save_results_to_json(results, file_path):
    """Save model training results to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    results = train_models('hotel_customer_data.csv')
    save_results_to_json(results, 'model_training_results.json')  # Save results to JSON
    print("Model Training Results:")
    for model, metrics in results.items():
        print(f"{model}:")
        print(f"  MSE: {metrics['MSE']:.4f}")
        print(f"  R²: {metrics['R2']:.4f}")
        print(f"  MAE: {metrics['MAE']:.4f}")
        print(f"  Accuracy: {metrics['Accuracy (%)']:.2f}%")
