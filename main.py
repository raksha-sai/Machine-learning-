#Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Creating sample data
data = {
    'text': [
        "I love this product",
        "Terrible experince",
        "It's okay",
        "The service was great",
        "Not satisfied with the quality",
        "This is fantastic!"
    ],
    'sentiment': [
        'Positive',
        'negative',
        'neutral',
        'positive',
        'negative',
        'positive'
    ]
}

df=pd.DataFrame(data)

#feature extraction
tfid_vectorizer = TfidfVectorizer()
x = tfid_vectorizer.fit_transform(df['text'])

#split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, df['sentiment'], test_size=0.2, random_state=42)

#create the model
model = LogisticRegression()

#Training the model
model.fit(x_train, y_train)

#Evaluate the model
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))

#confusion marix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blue', xticklabels=['negative', 'neutral', 'positive'],yticklabels=['negative', 'neutral', 'positive',])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Sentiment Distribution Bar chart
sentiment_distribution = y_test.value_counts()
sns.barplot(x=sentiment_distribution.index, y=sentiment_distribution.values, palette='viridis')
plt.xlabel('Count')
plt.title('Sentiment Distribution in Test Data')
plt.show()