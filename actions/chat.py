import pandas as pd
import sqlite3
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report


conn = sqlite3.connect('food.db')

# Preview the contents of both tables
categories_df = pd.read_sql_query("SELECT * FROM categories", conn)
food_items_df = pd.read_sql_query("SELECT * FROM food_items", conn)

print("Categories Preview:")
print(categories_df.head())
print("\nFood Items Preview:")
print(food_items_df.head())

# Copy the original food_items_df to add labels
df = food_items_df.copy()

# Apply heuristic rules to assign goals
def assign_goal(row):
    if row['energy_kcal'] < 150 and row['fat_g'] < 5 and row['fiber_g'] >= 2:
        return 'lose_weight'
    elif row['energy_kcal'] > 250 and row['protein_g'] > 10:
        return 'gain_muscle'
    elif row['energy_kcal'] > 300:
        return 'gain_weight'
    else:
        return 'maintain_weight'

df['goal'] = df.apply(assign_goal, axis=1)

# Features and target
features = ['carbohydrate_g', 'energy_kcal', 'protein_g', 'fat_g', 'fiber_g']
X = df[features]
y = df['goal']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Build and train model pipeline
model = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42))
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)

report_df = pd.DataFrame(report).transpose()
print("\nModel Classification Report:")
print(report_df)

# Function to print food items by goal
def print_foods_by_goal(dataframe, goal, num_items=10):
    """
    Print food items for a specific goal
    
    Parameters:
    dataframe: DataFrame containing food items with goals
    goal: str - one of 'lose_weight', 'gain_muscle', 'gain_weight', 'maintain_weight'
    num_items: int - number of items to display (default: 10)
    """
    goal_foods = dataframe[dataframe['goal'] == goal]
    
    if goal_foods.empty:
        print(f"No food items found for goal: {goal}")
        return
    
    print(f"\n=== FOOD ITEMS FOR {goal.upper().replace('_', ' ')} ===")
    print(f"Total items: {len(goal_foods)}")
    print("-" * 60)
    
    # Display top items (you can modify sorting criteria)
    display_cols = ['name', 'energy_kcal', 'protein_g', 'fat_g', 'carbohydrate_g', 'fiber_g']
    
    for i, (idx, row) in enumerate(goal_foods.head(num_items).iterrows()):
        print(f"{i+1}. {row['name']}")
        print(f"   Energy: {row['energy_kcal']:.1f} kcal | "
              f"Protein: {row['protein_g']:.1f}g | "
              f"Fat: {row['fat_g']:.1f}g | "
              f"Carbs: {row['carbohydrate_g']:.1f}g | "
              f"Fiber: {row['fiber_g']:.1f}g")
        print()

# Print food items for each goal
goals = ['lose_weight', 'gain_muscle', 'gain_weight', 'maintain_weight']

for goal in goals:
    print_foods_by_goal(df, goal, num_items=10)

# Summary of goal distribution
print("\n=== GOAL DISTRIBUTION SUMMARY ===")
goal_counts = df['goal'].value_counts()
print(goal_counts)
print(f"\nTotal food items: {len(df)}")

# Optional: Print foods by goal with better nutritional profile
print("\n=== TOP RECOMMENDATIONS BY GOAL ===")

# For weight loss: lowest calorie, high fiber
lose_weight_foods = df[df['goal'] == 'lose_weight'].sort_values(['energy_kcal', 'fiber_g'], ascending=[True, False])
print("\nBest for WEIGHT LOSS (lowest calories, highest fiber):")
print_foods_by_goal(lose_weight_foods, 'lose_weight', 5)

# For muscle gain: highest protein
gain_muscle_foods = df[df['goal'] == 'gain_muscle'].sort_values('protein_g', ascending=False)
print("\nBest for MUSCLE GAIN (highest protein):")
print_foods_by_goal(gain_muscle_foods, 'gain_muscle', 5)

# For weight gain: highest calories
gain_weight_foods = df[df['goal'] == 'gain_weight'].sort_values('energy_kcal', ascending=False)
print("\nBest for WEIGHT GAIN (highest calories):")
print_foods_by_goal(gain_weight_foods, 'gain_weight', 5)

# For maintenance: balanced nutrition
maintain_foods = df[df['goal'] == 'maintain_weight'].sort_values(['protein_g', 'fiber_g'], ascending=[False, False])
print("\nBest for WEIGHT MAINTENANCE (balanced nutrition):")
print_foods_by_goal(maintain_foods, 'maintain_weight', 5)

# Close database connection
conn.close()
