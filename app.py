
import streamlit as st
import pandas as pd
import joblib
import os

st.title("Conversie Voorspeller")

model_path = "final_model.pkl"
if not os.path.exists(model_path):
    st.error("❌ Modelbestand ontbreekt.")
    st.stop()

try:
    model, feature_names, label_encoder = joblib.load(model_path)
except Exception as e:
    st.error(f"❌ Fout bij laden van model: {e}")
    st.stop()

st.markdown("Voer de gegevens in om een voorspelling te krijgen:")

input_data = {}
for feature in feature_names:
    input_data[feature] = st.number_input(f"{feature}", step=1.0)

if st.button("Voorspel"):
    try:
        df = pd.DataFrame([input_data])
        prediction = model.predict(df)
        resultaat = label_encoder.inverse_transform(prediction)[0]
        st.success(f"→ Voorspelling: **{resultaat}**")
    except Exception as e:
        st.error(f"❌ Fout tijdens voorspellen: {e}")
