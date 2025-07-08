
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from .api_client import USDAAPIClient

GOAL_PROFILES = {
    'weight gain': {'calories': 'high', 'protein': 'high'},
    'weight loss': {'calories': 'low', 'fat': 'low'},
    'muscle gain': {'protein': 'high', 'calories': 'moderate'},
    'low sugar': {'sugars': 'low'},
    'low sodium': {'sodium': 'low'}
}

class SmartDietRecommender:
    def __init__(self):
        self.api = USDAAPIClient()
        self.features = ['calories', 'protein', 'fat', 'carbohydrate', 'fiber',
                         'sugars', 'calcium', 'iron', 'potassium', 'sodium', 'vitaminC']
        self.scaler = StandardScaler()
        self.knn_model = None
        self.kmeans_model = None
        self.data = None

    def load_data(self, query="apple", size=100)-> bool:
        foods=self.api.search_foods(query, size)
        if not foods:
            return False
        
        processed=[]
        for food in foods:
            item={
                'fdc_id': food['fdcId'],
                'description': food['description'],
                'brand':food.get('brandOwner', ''),
                'category': food.get('foodCategory', '')
            }

            nutrients={f: 0 for f in self.features}
            for n in food.get('foodNutrients', []):
                name=n['nutrientName'].lower()
                for f in self.features:
                    if f in name:
                        nutrients[f]=n['value']
            processed.append({**item, **nutrients})

        self.data=pd.DataFrame(processed).fillna(0)
        return True

    def train_models(self):
        if self.data is None:
            raise ValueError("No data loaded.")
        scaled = self.scaler.fit_transform(self.data[self.features])
        self.knn_model = NearestNeighbors(n_neighbors=5, metric='euclidean')
        self.knn_model.fit(scaled)
        self.kmeans_model = KMeans(n_clusters=5, random_state=42)
        self.data['cluster'] = self.kmeans_model.fit_predict(scaled)

    def filter_by_goal(self, goal: str, allergy_keywords: list = []):
        if self.data is None:
            return pd.DataFrame()
        goal = goal.lower()
        goal_profile = GOAL_PROFILES.get(goal, {})
        if not goal_profile:
            return pd.DataFrame()

        df = self.data.copy()
        for nutrient, target in goal_profile.items():
            if nutrient in df.columns:
                if target == 'high':
                    df = df.sort_values(by=nutrient, ascending=False)
                elif target == 'low':
                    df = df.sort_values(by=nutrient, ascending=True)
        for allergy in allergy_keywords:
            df = df[~df['description'].str.contains(allergy, case=False)]
        return df[['description', 'brand', 'category'] + self.features].head(10)

    def find_similar(self, food_name: str, n: int = 5) -> pd.DataFrame:
        matches = self.data[self.data['description'].str.contains(food_name, case=False)]
        if matches.empty:
            return pd.DataFrame()
        idx = matches.index[0]
        distances, indices = self.knn_model.kneighbors(
            self.scaler.transform([self.data.loc[idx, self.features]]),
            n_neighbors=n+1
        )
        results = self.data.iloc[indices[0][1:]].copy()
        results['distance'] = distances[0][1:]
        return results[['description', 'brand', 'category', 'distance']]

    def get_cluster_foods(self, food_name: str) -> pd.DataFrame:
        matches = self.data[self.data['description'].str.contains(food_name, case=False)]
        if matches.empty:
            return pd.DataFrame()
        cluster = matches.iloc[0]['cluster']
        return self.data[self.data['cluster'] == cluster][['description', 'brand', 'category']]
