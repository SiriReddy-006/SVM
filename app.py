import streamlit as st

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# title
st.title("SVM Iris Flower Prediction App")

st.write("Support Vector Machine Classification")

# load dataset
data = load_iris()

X = data.data
y = data.target

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

# train model
model = SVC(kernel='linear')

model.fit(X_train, y_train)

# accuracy
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")

st.write(accuracy)

# user inputs
st.subheader("Enter Flower Measurements")

sepal_length = st.number_input(
    "Sepal Length",
    step=1,
    format="%d"
)

sepal_width = st.number_input(
    "Sepal Width",
    step=1,
    format="%d"
)

petal_length = st.number_input(
    "Petal Length",
    step=1,
    format="%d"
)

petal_width = st.number_input(
    "Petal Width",
    step=1,
    format="%d"
)
# prediction
if st.button("Predict Flower"):

    input_data = scaler.transform([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    prediction = model.predict(input_data)

    flower_names = [
        "Setosa",
        "Versicolor",
        "Virginica"
    ]

    st.success(
        f"Predicted Flower: {flower_names[prediction[0]]}"
    )