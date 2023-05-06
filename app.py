import streamlit as st
import pickle
import numpy as np

# Laden Sie das gespeicherte Modell
with open('gs_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Erstellen Sie eine Funktion, um Vorhersagen zu treffen
def predict_survival(pclass, sex, age):
    input_data = np.array([[pclass, sex, age]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][prediction]
    if prediction == 1:
        return "You would have survived the Titanic with a probability of {:.2f}%".format(probability * 100)
    else:
        return "Unfortunately, you would not have survived the Titanic with a probability of {:.2f}%".format(probability * 100)

# Erstellen Sie die Streamlit-App
st.title("Titanic Survival Predictor")
st.write("Welcome to the Titanic Survival Predictor app! With this app, you can predict your chances of surviving the infamous Titanic disaster based on your passenger details. We used machine learning to create a model that predicts whether or not a passenger survived based on factors like their age, sex, and passenger class.")
st.write("Our model uses a Gradient Boosting Classifier, which is a powerful machine learning algorithm that builds an ensemble of weak learners and combines their predictions to make accurate classifications. The model has been trained on a dataset of real passenger data from the Titanic disaster, so it can give you a good estimate of your chances of survival.")
st.write("To use the app, simply enter your passenger details into the input fields and click \"Predict\". The app will then use the model to calculate your chances of surviving the disaster. We hope you find this app informative and interesting!")

pclass = st.slider("Class", 1, 3, 2)
sex = st.selectbox("Gender", ["female", "male"])
if sex == "female":
    sex = 0
else:
    sex = 1
age = st.slider("Age", 0, 100, 25)

if st.button("Calculate"):
    result = predict_survival(pclass, sex, age)
    st.write(result)
