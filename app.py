import streamlit as st

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# page title
st.title("SVM Iris Flower Prediction App")

st.write("Support Vector Machine Classification")

# load iris dataset
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

# feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# create and train model
model = SVC(kernel='linear')

model.fit(X_train, y_train)

# user inputs
st.subheader("Enter Flower Measurements")

sepal_length = st.slider(
    "Sepal Length",
    4.0,
    8.0,
    5.1
)

sepal_width = st.slider(
    "Sepal Width",
    2.0,
    5.0,
    3.5
)

petal_length = st.slider(
    "Petal Length",
    1.0,
    7.0,
    1.4
)

petal_width = st.slider(
    "Petal Width",
    0.1,
    3.0,
    0.2
)

# prediction button
if st.button("Predict Flower"):

    # prepare input
    input_data = scaler.transform([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    # prediction
    prediction = model.predict(input_data)

    # flower names
    flower_names = [
        "Setosa",
        "Versicolor",
        "Virginica"
    ]

    # result
    st.success(
        f"Predicted Flower: {flower_names[prediction[0]]}"
    )

# sample values
st.subheader("Sample Test Values")

st.write("Setosa → 5.1, 3.5, 1.4, 0.2")

st.write("Versicolor → 6.0, 2.9, 4.5, 1.5")

st.write("Virginica → 6.9, 3.1, 5.4, 2.1")
