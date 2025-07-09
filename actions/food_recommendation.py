import sqlite3
from pathlib import Path
from typing import Text, List, Dict, Any

import pandas as pd
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

# --- Database Setup ---
db_path = str(Path(__file__).parent / "food.db")
print(f"Database path: {db_path}")

# Load data and assign clusters
def load_and_cluster_foods():
    conn = sqlite3.connect(db_path)
    food_items_df = pd.read_sql_query("SELECT * FROM food_items", conn)
    conn.close()

# Assign each food to a goal cluster (heuristic or model-based)
    def assign_goal(row):
        if row['energy_kcal'] < 150 and row['fat_g'] < 5 and row['fiber_g'] >= 2:
            return 'lose_weight'
        elif row['energy_kcal'] > 250 and row['protein_g'] > 10:
            return 'gain_muscle'
        elif row['energy_kcal'] > 300:
            return 'gain_weight'
        else:
            return 'maintain_weight'

    food_items_df['goal'] = food_items_df.apply(assign_goal, axis=1)
    return food_items_df

try:
    df = load_and_cluster_foods()
    print(f"Loaded {len(df)} foods. Goals distribution:\n{df['goal'].value_counts()}")
except Exception as e:
    print(f"Error loading data: {e}")
    df = pd.DataFrame()

# --- Rasa Action ---
class FoodRecommendation(Action):
    def name(self) -> Text:
        return "action_recommend_food"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        if df.empty:
            dispatcher.utter_message(text="Sorry, the food database is not available.")
            return []

        # Map Rasa intent to goal cluster
        intent_to_goal = {
            "goal_lose_weight": "lose_weight",
            "goal_gain_weight": "gain_weight",
            "goal_gain_muscle": "gain_muscle",
            "goal_maintain_weight": "maintain_weight"
        }

        intent = tracker.latest_message.get("intent", {}).get("name")
        goal = intent_to_goal.get(intent)

        if not goal:
            dispatcher.utter_message(text="Please specify a valid goal like 'lose_weight', 'gain_muscle', etc.")
            return []

        # Filter foods for the goal and pick one randomly
        goal_foods = df[df['goal'] == goal]
        if goal_foods.empty:
            dispatcher.utter_message(text=f"No foods found for {goal.replace('_', ' ')}.")
            return []

        random_food = goal_foods.sample(1).iloc[0]  # Random selection

        # Format response
        response = (
            f"Here's a recommendation for {goal.replace('_', ' ')}: **{random_food['name']}**\n"
            f"Nutrition per 100g:\n"
            f"• Calories: {random_food['energy_kcal']} kcal\n"
            f"• Protein: {random_food['protein_g']}g\n"
            f"• Fat: {random_food['fat_g']}g\n"
            f"• Carbs: {random_food['carbohydrate_g']}g\n"
            f"• Fiber: {random_food['fiber_g']}g"
        )

        dispatcher.utter_message(text=response)
        return []