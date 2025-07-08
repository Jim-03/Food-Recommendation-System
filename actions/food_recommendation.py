from typing import Text, List, Dict, Any

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict


class FoodRecommendation(Action):
    def name(self) -> Text:
        return "action_recommend_food"

    def run(self, dispatcher: "CollectingDispatcher", tracker: Tracker, domain: "DomainDict") -> List[Dict[Text, Any]]:
        intent = tracker.latest_message.get("intent", {}).get("name")
        dispatcher.utter_message(text=f"Identified intent: {intent}")
        return []