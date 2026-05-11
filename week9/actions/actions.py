
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import re

class MockDatabase:
    @staticmethod
    def get_order_status(order_id: str) -> str:
        db = {
            "12345": "Shipped - arriving on Friday.",
            "67890": "Processing - expected to ship tomorrow."
        }
        return db.get(order_id, "Order not found. Please check your order ID.")

class ActionTrackOrder(Action):
    def name(self) -> Text:
        return "action_track_order"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        order_id = tracker.get_slot("order_id")
        
        if not order_id:
            dispatcher.utter_message(text="I couldn't find an order ID to track.")
            return []

        status = MockDatabase.get_order_status(order_id)
        dispatcher.utter_message(text=f"Status for order {order_id}: {status}")

        return []

class ValidateSlotOrder(Action):
    def name(self) -> Text:
        return "validate_track_order_form"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
            
        order_id = tracker.get_slot("order_id")
        
        # Validation: Ensure order ID is numeric
        if order_id and not re.match(r"^[0-9]+$", order_id):
            dispatcher.utter_message(text="Order IDs should only contain numbers. Please try again.")
            return [SlotSet("order_id", None)]
            
        return []
