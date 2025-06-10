# mental_health_monitor_console.py
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 🧠 Enhanced sample data (more balanced)
data = {
    "text": [
        "I'm so tired and feel anxious all the time.",
        "I had a great day at college!",
        "I don't want to attend classes anymore.",
        "Looking forward to exams!",
        "I'm feeling really stressed and can't sleep.",
        "I'm completely fine, enjoying life.",
        "I'm having breakdowns daily.",
        "Today was productive and happy.",
        "Nobody understands me. I feel alone.",
        "I've made good progress this week.",
        "Sometimes I feel sad but it passes.",
        "I want to disappear forever."
    ],
    "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1]  # 1=At-Risk, 0=Normal
}

# 🛠 Data processing
def process_data(data):
    for i, text in enumerate(data['text']):
        data['text'][i] = text.lower().strip(".!")
    return data

processed_data = process_data(data)

# 🔍 Sentiment analysis
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return {
        'polarity': analysis.sentiment.polarity,
        'subjectivity': analysis.sentiment.subjectivity
    }

# 📊 Model training and evaluation
def train_model(data):
    # Extract features
    features = []
    for text in data['text']:
        sentiment = analyze_sentiment(text)
        features.append([sentiment['polarity'], sentiment['subjectivity']])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, data['label'], test_size=0.25, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    return model, X_test, y_test, y_pred

# 📝 Enhanced console reporting
def console_report(y_test, y_pred):
    print("\n🔍 Model Evaluation:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "At-Risk"]))
    
    print("\n📊 Confusion Matrix (Text Version):")
    cm = confusion_matrix(y_test, y_pred)
    print(f"         Predicted")
    print(f"          Normal At-Risk")
    print(f"Actual Normal  {cm[0][0]}      {cm[0][1]}")
    print(f"       At-Risk {cm[1][0]}      {cm[1][1]}")

# 🤖 Prediction function
def predict_mental_state(model, text):
    sentiment = analyze_sentiment(text)
    prediction = model.predict([[sentiment['polarity'], sentiment['subjectivity']]])[0]
    return "At-Risk" if prediction == 1 else "Normal"

# 🚀 Main execution
if _name_ == "_main_":
    # Train and evaluate
    model, X_test, y_test, y_pred = train_model(processed_data)
    console_report(y_test, y_pred)
    
    # Test cases
    test_cases = [
        "I feel hopeless and alone",
        "Everything is going great!",
        "I can't handle this stress anymore",
        "I'm okay but sometimes worry"
    ]
    
    print("\n🧪 Test Predictions:")
    for text in test_cases:
        result = predict_mental_state(model, text)
        print(f"- '{text[:30]}...': {result}")
    
    print("\n✅ Program completed successfully without GUI dependencies")

-------------------------------------------------------


Output:

 Classification Report:
              precision    recall  f1-score   support

      Normal       1.00      1.00      1.00         1
     At-Risk       1.00      1.00      1.00         1

    accuracy                           1.00         2
   macro avg       1.00      1.00      1.00         2
weighted avg       1.00      1.00      1.00         2

Confusion Matrix (Graph):
Predicted
          Normal   At-Risk
Actual
Normal      1         0
At-Risk     0         1


Predicted Risk for Input Sentence:
Input: I feel hopeless and don't want to talk to anyone.
Prediction: At-Risk


