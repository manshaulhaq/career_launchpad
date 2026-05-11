import yaml
import pandas as pd

df = pd.read_csv('dataset/bitext_cutomer_support.csv')

all_rules = [
    {
        "rule": "Say goodbye",
        "steps": [{"intent": "goodbye"}, {"action": "utter_goodbye"}]
    },
    {
        "rule": "Say greet", 
        "steps": [{"intent": "greet"}, {"action": "utter_greet"}]
    }
]

for intent in df['intent'].unique():
    all_rules.append({
        "rule": f"Handle {intent}",
        "steps": [
            {"intent": intent},
            {"action": f"utter_{intent}"}
        ]
    })

rules_data = {"version": "3.1", "rules": all_rules}

with open('data/rules.yml', 'w') as f:
    yaml.dump(rules_data, f, default_flow_style=False, sort_keys=False)

print(f"Done — {len(all_rules)} rules written")