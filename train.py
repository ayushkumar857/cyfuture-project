import warnings
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import pyttsx3
warnings.filterwarnings("ignore")

df = pd.read_csv("dataset.csv")
df = df.dropna(subset=["content", "sentiment"])

X_train, X_test, y_train, y_test = train_test_split(
    df["content"], df["sentiment"], test_size=0.2, random_state=42
)

model = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', MultinomialNB())
])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(model, "emotion_model.joblib")

engine = pyttsx3.init()
engine.save_to_file("I am feeling really happy and excited today.", "audio.wav")
engine.runAndWait()
print("Model trained and audio.wav generated.")