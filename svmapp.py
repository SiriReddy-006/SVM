import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVR

# title
st.title("Titanic Fare Prediction using SVR")

st.write("Predict Passenger Fare using Support Vector Regression")

# load dataset
df = pd.read_csv("train.csv")

# preprocessing
label_encoder = LabelEncoder()

df['Sex'] = label_encoder.fit_transform(df['Sex'])

# fill missing values
df['Age'] = df['Age'].fillna(df['Age'].mean())

df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

# features
X = df[['Pclass', 'Sex', 'Age']]

# target
y = df['Fare']

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

# create model
model = SVR(
    kernel='rbf',
    C=100,
    gamma=0.1,
    epsilon=0.1
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
    min_value=1,
    max_value=80,
    value=25,
    step=1
)

# convert gender
sex_value = 1 if sex == "Male" else 0

# prediction button
predict_button = st.button("Predict Fare")

# prediction only after button click
if predict_button:

    input_data = [[
        pclass,
        sex_value,
        age
    ]]

    # scale input
    input_data = scaler.transform(input_data)

    # predict fare
    prediction = model.predict(input_data)

    # remove decimal values
    predicted_fare = int(prediction[0])

    st.success(
        f"Predicted Fare: ${predicted_fare}"
    )