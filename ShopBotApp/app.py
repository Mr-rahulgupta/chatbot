import json, random, numpy as np, tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Load JSON
with open("intents.json") as file:
    data = json.load(file)

training_sentences, training_labels, responses = [], [], {}
menu_data = {}

# Extract data from JSON
for intent in data['intents']:
    for pattern in intent.get("patterns", []):
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    responses[intent['tag']] = intent.get("responses", [])
    
    if intent["tag"] == "greeting" and "menu" in intent:
        menu_data = intent["menu"]

# Encode labels
lbl_encoder = LabelEncoder()
labels = lbl_encoder.fit_transform(training_labels)

# Tokenization and padding
tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(training_sentences)
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(1000, 16, input_length=padded.shape[1]),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(len(set(labels)), activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(padded, np.array(labels), epochs=500, verbose=0)

# Predict intent tag
def predict_tag(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=padded.shape[1], padding='post')
    return lbl_encoder.inverse_transform([np.argmax(model.predict(pad, verbose=0))])[0]

# Display menus
def show_menu(menu_dict):
    print("\n=== Main Menu ===")
    for key, val in menu_dict.items():
        print(f"{key}. {val['name']}")
    print("Type number, 'back', 'quit', or '*' to show main menu anytime.")

def show_sub_menu(options):
    print("\n--- Sub Menu ---")
    for key, val in options.items():
        print(f"{key}. {val}")
    print("Type option or 'back'.")

# Map menu names to numbers
menu_aliases = {
    "order related": "1",
    "delivery": "2",
    "payment": "3",
    "offers": "4",
    "account": "5",
    "contact": "6",
    "contact support": "6",
    "support": "6"
}

# Chatbot loop
def chatbot():
    stack = []
    menu_shown = False
    print("🤖 Hi! I am ShopBot. Type 'quit' to exit, 'back' to go back, or '*' to show main menu anytime.")

    while True:
        user_input = input("\nYou: ").strip().lower()

        if user_input == "quit":
            print("Bot: 👋 Goodbye!")
            break

        # Show main menu on greeting first time or '*' input
        if not menu_shown or user_input == "*":
            show_menu(menu_data)
            menu_shown = True
            continue

        if user_input == "back":
            if stack:
                stack.pop()
            else:
                print("Bot: Already at main menu. Type '*' to view menu.")
            continue

        # Convert text menu names to number keys
        if not stack:
            if user_input in menu_aliases:
                user_input = menu_aliases[user_input]

            if user_input in menu_data:
                stack.append(menu_data[user_input]["options"])
                show_sub_menu(stack[-1])
                continue

        # Sub-menu input handling
        if stack:
            current_options = stack[-1]
            if user_input in current_options:
                print(f"Bot: You selected '{current_options[user_input]}'")
                stack.pop()  # after selection, return to normal conversation
                continue
            match=None
            for key,val in current_options.items():
                if user_input == val.lower():
                    match=val
                    break
            if match:
                print(f"Bot: You selected '{match}'")
                stack.pop()
                continue
            print("Bot: Invalid option. Please choose a sub-menu option.")
            continue

        # Free text ML intent
        tag = predict_tag(user_input)
        if tag in responses and responses[tag]:
            print("Bot:", random.choice(responses[tag]))
        else:
            print("Bot: I did not understand. Type '*' to view main menu or ask a question")
chatbot()
