# Patch nlu.yml to add missing intents
with open('data/nlu.yml', 'r') as f:
    nlu_data = yaml.safe_load(f)

nlu_data["nlu"] += [
    {"intent": "greet", "examples": "- hi\n- hello\n- hey\n- good morning\n- good evening"},
    {"intent": "goodbye", "examples": "- bye\n- goodbye\n- see you\n- take care"},
    {"intent": "inform_order_id", "examples": "- my order id is [12345](order_id)\n- it's [67890](order_id)\n- order [11111](order_id)\n- [99999](order_id)"}
]

with open('data/nlu.yml', 'w') as f:
    yaml.dump(nlu_data, f, default_flow_style=False, sort_keys=False)

print("NLU patched")
