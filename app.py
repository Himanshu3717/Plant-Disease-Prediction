from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for writing to files
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# Load the models
def load_models(models_dir):
    """Load the trained models and scaler from the specified directory."""
    models = {}
    model_names = ['linear_regression_model', 'decision_tree_model', 'random_forest_model', 'scaler']
    
    for name in model_names:
        models[name] = joblib.load(f'{models_dir}/{name}.joblib')
    
    return models

# Stats route
@app.route('/stats')
def stats():
    # Load model training results from the JSON file
    with open('model_training_results.json', 'r') as f:
        model_stats = json.load(f)

    # Load the dataset for analysis
    data = pd.read_csv('hotel_customer_data.csv')
    
    # Create visualizations
    plot_gender_distribution(data)
    plot_family_visit_distribution(data)
    plot_reason_distribution(data)
    plot_room_type_distribution(data)
    plot_ratings_distribution(data)

    return render_template('stats.html', model_stats=model_stats)

# Plotting functions
def plot_gender_distribution(data):
    gender_counts = data['Gender'].value_counts()

    plt.figure(figsize=(8, 6))
    plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
    plt.title('Gender Distribution')
    plt.axis('equal')  # Equal aspect ratio ensures the pie chart is a circle.
    plt.savefig('static/gender_distribution.png')
    plt.close()

def plot_family_visit_distribution(data):
    family_counts = data['Visited_With_Family'].value_counts()

    plt.figure(figsize=(8, 6))
    plt.bar(family_counts.index, family_counts.values, color=['lightgreen', 'lightpink'])
    plt.title('Visits With Family')
    plt.xlabel('Visited With Family')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.savefig('static/family_visit_distribution.png')
    plt.close()

def plot_reason_distribution(data):
    reason_counts = data['Reason_To_Visit'].value_counts()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=reason_counts.index, y=reason_counts.values, hue=reason_counts.index, palette='viridis', legend=False)
    plt.title('Reasons to Visit')
    plt.xlabel('Reason')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.savefig('static/reason_distribution.png')
    plt.close()

def plot_room_type_distribution(data):
    room_counts = data['Room_Type'].value_counts()

    plt.figure(figsize=(8, 6))
    sns.barplot(x=room_counts.index, y=room_counts.values, hue=room_counts.index, palette='magma', legend=False)
    plt.title('Room Type Distribution')
    plt.xlabel('Room Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.savefig('static/room_type_distribution.png')
    plt.close()

def plot_ratings_distribution(data):
    rating_columns = [
        'Cleanliness_Rating',
        'Comfort_Rating',
        'Staff_Performance',
        'Facilities_Rating',
        'Location_Rating',
        'Overall_Satisfaction_Score'
    ]
    
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(rating_columns, 1):
        plt.subplot(2, 3, i)
        sns.histplot(data[col], bins=5, kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('static/ratings_distribution.png')
    plt.close()

# Index route for predictions
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_results = None
    if request.method == 'POST':
        # Get user input from the form
        cleanliness_rating = float(request.form['cleanliness'])
        comfort_rating = float(request.form['comfort'])
        staff_performance = float(request.form['staff'])
        facilities_rating = float(request.form['facilities'])
        location_rating = float(request.form['location'])
        overall_score = float(request.form['overall'])  # Get overall score input

        # Load models
        models_dir = 'models'
        models = load_models(models_dir)

        # Prepare input for prediction excluding overall score for scaling
        user_input = {
            'Cleanliness_Rating': cleanliness_rating,
            'Comfort_Rating': comfort_rating,
            'Staff_Performance': staff_performance,
            'Facilities_Rating': facilities_rating,
            'Location_Rating': location_rating
        }
        
        scaler = models['scaler']
        
        # Convert user input to a DataFrame for scaling
        input_df = pd.DataFrame([user_input])
        input_scaled = scaler.transform(input_df)  # Scale the user input

        # Make predictions
        predictions = {}
        for model_name in ['linear_regression_model', 'decision_tree_model', 'random_forest_model']:
            model = models[model_name]
            prediction = model.predict(input_scaled)
            predictions[model_name] = np.round(prediction).astype(int)

        # Interpret predictions
        labels = {0: 'Bad', 1: 'Good', 2: 'Excellent'}
        prediction_results = {}
        
        for model, pred in predictions.items():
            pred_value = pred[0]
            
            # For linear regression, map the continuous output to categories
            if model == 'linear_regression_model':
                if pred_value < 1:
                    prediction_results[model] = 'Bad'
                elif pred_value < 3:
                    prediction_results[model] = 'Good'
                else:
                    prediction_results[model] = 'Excellent'
            else:
                if pred_value in labels:
                    prediction_results[model] = labels[pred_value]
                else:
                    prediction_results[model] = "Unknown prediction"

    return render_template('index.html', prediction_results=prediction_results)

if __name__ == '__main__':
    app.run(debug=True)
