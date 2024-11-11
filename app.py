import sqlite3
from flask import Flask, request, render_template, session, redirect, url_for, flash
import requests
import joblib
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'supersecretkey'
API_KEY = '99b4aba459a54a33a820659f752a868f'
DATABASE = 'users.db'  # SQLite database file

# Load the trained model and scaler
model = joblib.load(r'C:\Users\Aditya\OneDrive\Desktop\python\calorie_tracker\food_classification_model.pkl')
scaler = joblib.load(r'C:\Users\Aditya\OneDrive\Desktop\python\calorie_tracker\scaler.pkl')

# Function to get a database connection
def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

# Function to initialize the database and create the users table
def init_db():
    with app.app_context():
        db = get_db()
        db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        """)
        db.commit()

# Initialize the database
init_db()

# Function to classify food using the model and scaler
def classify_food_with_model(basic_nutrients):
    features = np.array([
        basic_nutrients.get("Calories", 0),
        basic_nutrients.get("Carbohydrates", 0),
        basic_nutrients.get("Protein", 0),
        basic_nutrients.get("Fat", 0),
        basic_nutrients.get("Fiber", 0),
        basic_nutrients.get("Sugar", 0),
        basic_nutrients.get("Sodium", 0)
    ]).reshape(1, -1)

    # Scale and classify
    features = scaler.transform(features)
    classification = model.predict(features)[0]
    return classification

# Food search function
def search_food(food_name):
    api_url = "https://api.spoonacular.com/food/ingredients/search"
    params = {"query": food_name, "apiKey": API_KEY, "number": 10}
    response = requests.get(api_url, params=params)
    if response.status_code == 200 and response.json()['results']:
        return response.json()['results'][0]['id']
    return None

# Recipe search function
def search_recipe(food_name):
    api_url = "https://api.spoonacular.com/recipes/complexSearch"
    params = {
        "query": food_name,
        "apiKey": API_KEY,
        "number": 1,
        "addRecipeInformation": True,
        "addRecipeNutrition": True
    }
    response = requests.get(api_url, params=params)
    if response.status_code == 200 and response.json()['results']:
        return response.json()['results'][0]
    return None

# Nutrition information retrieval
def get_nutrition_info(ingredient_id, amount=1, unit='cup'):
    api_url = f"https://api.spoonacular.com/food/ingredients/{ingredient_id}/information"
    params = {"apiKey": API_KEY, "amount": amount, "unit": unit}
    response = requests.get(api_url, params=params)
    if response.status_code == 200:
        data = response.json()
        nutrients = data.get('nutrition', {}).get('nutrients', [])
        
        # Macronutrients
        basic_nutrients = {"Calories": 0, "Protein": 0, "Fat": 0, "Carbohydrates": 0}
        # Micronutrients
        micronutrients = {"Fiber": 0, "Sugar": 0, "Calcium": 0, "Iron": 0, "Potassium": 0}
        
        for nutrient in nutrients:
            if nutrient['name'] in basic_nutrients:
                basic_nutrients[nutrient['name']] = nutrient['amount']
            elif nutrient['name'] in micronutrients:
                micronutrients[nutrient['name']] = nutrient['amount']
        
        classification = classify_food_with_model(basic_nutrients)
        return basic_nutrients, micronutrients, classification
    return None, None, None

# Main route
@app.route("/", methods=["GET", "POST"])
def index():
    if 'username' not in session:
        return redirect(url_for('login'))

    session.setdefault('calorie_goal', 2000)
    session.setdefault('food_items', [])
    session.setdefault('meal_nutrients', {"breakfast": [], "lunch": [], "dinner": [], "snacks": []})

    calorie_goal = int(session['calorie_goal'])
    total_calories = sum(item['calories'] for item in session['food_items'])
    total_protein = sum(item['protein'] for item in session['food_items'])
    total_fat = sum(item['fat'] for item in session['food_items'])
    total_carbs = sum(item['carbohydrates'] for item in session['food_items'])
    total_micronutrients = aggregate_micronutrients(session['food_items'])
    
    # Calculate progress percentage
    progress_percentage = min((total_calories / calorie_goal) * 100, 100) if calorie_goal else 0

    if request.method == "POST":
        food_name = request.form.get("food_name")
        amount = float(request.form.get("amount", 1))
        unit = request.form.get("unit", "cup")
        meal_type = request.form.get("meal_type", "snacks")
        
        ingredient_id = search_food(food_name)
        if ingredient_id:
            basic_nutrients, micronutrients, classification = get_nutrition_info(ingredient_id, amount, unit)
            if basic_nutrients:
                food_item = {
                    "food_name": food_name,
                    "amount": amount,
                    "unit": unit,
                    "calories": basic_nutrients["Calories"],
                    "protein": basic_nutrients["Protein"],
                    "fat": basic_nutrients["Fat"],
                    "carbohydrates": basic_nutrients["Carbohydrates"],
                    "micronutrients": micronutrients,
                    "classification": classification,
                }
            else:
                return render_template("index.html", error="Nutrition information not available.")
        else:
            recipe_data = search_recipe(food_name)
            if recipe_data:
                food_item = {
                    "food_name": recipe_data['title'],
                    "amount": amount,
                    "unit": unit,
                    "calories": recipe_data['nutrition']['nutrients'][0]['amount'],
                    "protein": recipe_data['nutrition']['nutrients'][1]['amount'],
                    "fat": recipe_data['nutrition']['nutrients'][2]['amount'],
                    "carbohydrates": recipe_data['nutrition']['nutrients'][3]['amount'],
                    "micronutrients": {
                        "Fiber": recipe_data['nutrition']['nutrients'][4]['amount'] if len(recipe_data['nutrition']['nutrients']) > 4 else 0,
                        "Sugar": 0, "Calcium": 0, "Iron": 0, "Potassium": 0
                    },
                    "classification": "complex recipe",
                }
            else:
                return render_template("index.html", error="Food item not found.")
        
        session['food_items'].append(food_item)
        session['meal_nutrients'][meal_type].append(food_item)
        session.modified = True

    return render_template(
        "index.html",
        total_calories=total_calories,
        total_protein=total_protein,
        total_fat=total_fat,
        total_carbs=total_carbs,
        total_micronutrients=total_micronutrients,
        calorie_goal=calorie_goal,
        progress_percentage=progress_percentage,
        food_items=session['food_items'],
        meal_nutrients=session['meal_nutrients'],
    )

# Route to set calorie goal
@app.route("/set_calories", methods=["POST"])
def set_calories():
    try:
        calorie_goal = int(request.form.get("calorie_goal"))
        session['calorie_goal'] = calorie_goal
        session.modified = True
    except (TypeError, ValueError):
        session['calorie_goal'] = 2000
    return redirect(url_for("index"))

# Route to remove a food item by index
@app.route("/remove/<int:index>", methods=["GET"])
def remove_item(index):
    try:
        session['food_items'].pop(index)
        session.modified = True
    except IndexError:
        pass
    return redirect(url_for("index"))

# Route to register a user
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        db = get_db()
        existing_user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        
        if existing_user:
            flash("Username already exists.")
            return redirect(url_for("register"))
        
        hashed_password = generate_password_hash(password)
        db.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        db.commit()
        flash("Registration successful. Please log in.")
        return redirect(url_for("login"))
    
    return render_template("register.html")

# Route to login a user
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        db = get_db()
        user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        
        if not user or not check_password_hash(user['password'], password):
            flash("Invalid credentials.")
            return redirect(url_for("login"))
        
        session['username'] = username
        return redirect(url_for("index"))
    
    return render_template("login.html")

# Helper function to aggregate micronutrients
def aggregate_micronutrients(food_items):
    micronutrients = {"Fiber": 0, "Sugar": 0, "Calcium": 0, "Iron": 0, "Potassium": 0}
    for item in food_items:
        for nutrient in micronutrients:
            micronutrients[nutrient] += item["micronutrients"].get(nutrient, 0)
    return micronutrients

# Route to view all users
@app.route("/view_users", methods=["GET"])
def view_users():
    db = get_db()
    users = db.execute("SELECT * FROM users").fetchall()  # Fetch all users from the database
    return render_template("view_users.html", users=users)

if __name__ == "__main__":
    app.run(debug=True)
