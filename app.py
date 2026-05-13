import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

# title
st.title("Titanic Survival Prediction using SVM")

st.write("Predict whether a passenger survived or not")

# load dataset
df = pd.read_csv("train.csv")

# preprocessing
label_encoder = LabelEncoder()

df['Sex'] = label_encoder.fit_transform(df['Sex'])

# fill missing values
df['Age'] = df['Age'].fillna(df['Age'].mean())

df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

# features
X = df[['Pclass', 'Sex', 'Age', 'Fare']]

# target
y = df['Survived']

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# model
model = SVC(
    kernel='rbf',
    C=10,
    gamma='scale'
)

# train model
model.fit(X_train, y_train)

# user inputs
st.subheader("Enter Passenger Details")

pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

sex = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.slider(
    "Age",
    1,
    80,
    25
)

fare = st.slider(
    "Fare",
    0,
    600,
    50
)

# convert gender
sex_value = 1 if sex == "Male" else 0

# prediction button
predict_button = st.button("Predict Survival")

# prediction
if predict_button:

    input_data = [[
        pclass,
        sex_value,
        age,
        fare
    ]]

    # scale input
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Passenger Survived")
    else:
        st.error("Passenger Did Not Survive")