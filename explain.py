import shap
import joblib

model = joblib.load("models/crop_model.pkl")

def explain_prediction(input_data):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    return shap_values