import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# Load data from both files
usda_data = pd.read_csv(r"C:\Users\Aditya\Downloads\USDA.csv")  # Replace with the actual path
classification_data = pd.read_excel(r"C:\Users\Aditya\OneDrive\Desktop\python\calorie_tracker\Classificationmodel.xlsx")  # Replace with the actual path

# Select columns of interest (adjust as needed)
# Here I assume we're using only one of the files; otherwise, merge as needed
data = classification_data[['Calories (kcal)', 'Carbohydrates (g)', 'Protein (g)', 'Fat (g)', 'Fiber (g)', 'Sugar (g)', 'Sodium (mg)']]

# Simple rule-based classification (modify as per your requirements)
def classify_food(row):
    if row['Protein (g)'] > 15:
        return 'High-Protein'
    elif row['Carbohydrates (g)'] > 20:
        return 'High-Carb'
    elif row['Fat (g)'] < 5:
        return 'Low-Fat'
    else:
        return 'Balanced'

data['Category'] = data.apply(classify_food, axis=1)


# Define feature columns and target column
X = data.drop('Category', axis=1)
y = data['Category']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the training data, then transform the test data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the model
joblib.dump(model, 'food_classification_model.pkl')

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Load model and scaler
model = joblib.load('food_classification_model.pkl')
scaler = joblib.load('scaler.pkl')

# Example of using the model on new data
def classify_new_food(data_row):
    data_row_scaled = scaler.transform([data_row])
    return model.predict(data_row_scaled)

# Example usage
new_food_data = [200, 30, 5, 2, 4, 6, 50]  # Replace with actual data
category = classify_new_food(new_food_data)
print("Predicted Category:", category[0])

