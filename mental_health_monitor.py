# mental_health_monitor.py

# 📦 Import required libraries
import pandas as pd
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 🧠 Sample data
data = {
    "text": [
        "I'm so tired and feel anxious all the time.",
        "I had a great day at college and everything feels fine.",
        "I don't want to attend classes anymore.",
        "Looking forward to exams and improving myself.",
        "I'm feeling really stressed and can't sleep well.",
        "I'm completely fine, enjoying my college life.",
        "I'm having breakdowns and feeling worthless.",
        "Today was productive, I completed all assignments.",
        "Nobody understands me. I feel alone.",
        "I've made good progress this week."
    ],
    "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}

# 📄 Create DataFrame
df = pd.DataFrame(data)

# 🔍 Extract sentiment score using TextBlob
def extract_sentiment(text):
    return TextBlob(text).sentiment.polarity

df['sentiment_score'] = df['text'].apply(extract_sentiment)

# 🎯 Define features and labels
X = df[['sentiment_score']]
y = df['label']

# 🔀 Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔧 Train model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# 📈 Evaluate model
y_pred = clf.predict(X_test)
print("\n📝 Classification Report:\n", classification_report(y_test, y_pred))

# 📊 Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', xticklabels=["Normal", "At-Risk"], yticklabels=["Normal", "At-Risk"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 🤖 Risk prediction function
def predict_risk(text):
    score = TextBlob(text).sentiment.polarity
    return "At-Risk" if clf.predict([[score]])[0] == 1 else "Normal"

# 🧪 Example usage
user_input = "I feel hopeless and don't want to talk to anyone."
print(f"\nInput: {user_input}")
print("Prediction:", predict_risk(user_input))
