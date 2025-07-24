import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

def train_model(df):
    df = df.copy()

    # Preprocessing
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])

    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    X = df[features]
    y = df['Survived']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def predict_survival(model, passenger_dict):
    # Encode 'Sex' to 0 or 1
    sex_encoded = 1 if passenger_dict['Sex'] == 'male' else 0

    features = [
        passenger_dict['Pclass'],
        sex_encoded,
        passenger_dict['Age'],
        passenger_dict['SibSp'],
        passenger_dict['Parch'],
        passenger_dict['Fare'],
    ]

    prediction = model.predict([features])[0]
    return prediction
