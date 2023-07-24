import pandas as pd
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('injuries.csv')

# Select relevant features
features = ['MIN', 'TOUCHES', 'USG_PCT', 'DRIVES']
target = 'INJURED_TYPE'

# Split the data into training and testing sets
x = data[features]
y = data[target]
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Create and train the model
model = RandomForestClassifier()
model.fit(x_train_scaled, y_train)

# Predict for a new player
new_player = pd.DataFrame([[input("Enter minutes per game: "), input("Enter touches per game: "),
                            input("Enter usage percentage: "), input("Enter drives per game: ")]], columns=features)
new_player_scaled = scaler.transform(new_player)
prediction = model.predict(new_player_scaled)
prediction_proba = model.predict_proba(new_player_scaled)[:, 1]  # Probability of injury

# Print the prediction
print("Probability of injury: {:.2f}%".format(prediction_proba[0] * 100))

# Evaluate the model
y_pred = model.predict(x_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Classification Report:\n", report)
