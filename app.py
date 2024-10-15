from flask import Flask, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)

# Load dataset
df = pd.read_csv('dataset/Churn_Modelling.csv')

# Preprocess data (Example: Using Random Forest for simplicity)
def preprocess_data(df):
    # Selecting features and target
    X = df[['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
    y = df['Exited']

    # One-hot encoding categorical variables
    X = pd.get_dummies(X, columns=['Geography', 'Gender'], drop_first=True)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

# Train RandomForest Model and calculate confusion matrix
def train_model():
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    return model, scaler, accuracy, cm

# Load model, scaler, accuracy, and confusion matrix
model, scaler, accuracy, cm = train_model()

# Plot confusion matrix and save as image
def plot_confusion_matrix(cm):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=['Not Churn', 'Churn'], yticklabels=['Not Churn', 'Churn'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('static/confusion_matrix.png')  # Save the plot as a PNG image
    plt.close()

# Plot the confusion matrix
plot_confusion_matrix(cm)

@app.route('/')
def home():
    return render_template('index.html', accuracy=accuracy)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input
        input_data = request.form
        
        # Process input data
        CreditScore = float(input_data['CreditScore'])
        Geography = input_data['Geography']
        Gender = input_data['Gender']
        Age = int(input_data['Age'])
        Tenure = int(input_data['Tenure'])
        Balance = float(input_data['Balance'])
        NumOfProducts = int(input_data['NumOfProducts'])
        HasCrCard = int(input_data['HasCrCard'])
        IsActiveMember = int(input_data['IsActiveMember'])
        EstimatedSalary = float(input_data['EstimatedSalary'])

        # Create dataframe for input
        input_df = pd.DataFrame([[CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]],
                                columns=['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'])

        # One-hot encode and scale input data
        input_df = pd.get_dummies(input_df, columns=['Geography', 'Gender'], drop_first=True)
        input_df = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_df)[0]

        # Return result
        result = 'Customer will churn' if prediction == 1 else 'Customer will not churn'
        return render_template('index.html', prediction_text=f'Prediction: {result}', accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)
