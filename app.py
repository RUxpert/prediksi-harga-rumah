import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "tuning_v1.1-copy"

@st.cache_resource
def load_artifacts():
    with open(MODEL_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    model.set_params(device="cpu")
    with open(MODEL_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(MODEL_DIR / "metadata.json", "r") as f:
        metadata = json.load(f)
    return model, scaler, metadata

def format_rupiah(value: float) -> str:
    return f"Rp {value:,.0f}".replace(",", ".")

model, scaler, metadata = load_artifacts()

st.set_page_config(page_title="Prediksi Harga Rumah", page_icon="🏠", layout="centered")

st.title("🏠 Prediksi Harga Rumah")
st.caption(f"Model: XGBoost · R² = {metadata['metrics']['r2']:.4f} · MAPE = {metadata['metrics']['mape']:.2f}%")

st.divider()
st.subheader("Masukkan Detail Properti")

col1, col2 = st.columns(2)

with col1:
    kamar_tidur = st.number_input(
        "Kamar Tidur",
        min_value=1,
        max_value=20,
        value=4,
        step=1,
        help="Jumlah kamar tidur"
    )
    kamar_mandi = st.number_input(
        "Kamar Mandi",
        min_value=1,
        max_value=20,
        value=3,
        step=1,
        help="Jumlah kamar mandi"
    )
    garasi = st.number_input(
        "Garasi",
        min_value=0,
        max_value=20,
        value=2,
        step=1,
        help="Kapasitas garasi (jumlah kendaraan)"
    )

with col2:
    luas_tanah = st.number_input(
        "Luas Tanah (m²)",
        min_value=10.0,
        max_value=10000.0,
        value=438.0,
        step=1.0,
        format="%.1f",
        help="Luas tanah dalam meter persegi"
    )
    luas_bangunan = st.number_input(
        "Luas Bangunan (m²)",
        min_value=10.0,
        max_value=10000.0,
        value=110.0,
        step=1.0,
        format="%.1f",
        help="Luas bangunan dalam meter persegi"
    )

st.divider()

if st.button("Prediksi Harga", type="primary", use_container_width=True):
    input_data = pd.DataFrame(
        [[kamar_tidur, kamar_mandi, garasi, luas_tanah, luas_bangunan]],
        columns=metadata["features"]
    )

    input_scaled = scaler.transform(input_data)
    predicted_price = np.expm1(model.predict(input_scaled)[0])

    st.success("Estimasi Harga Properti")
    st.metric(
        label="Harga Prediksi",
        value=format_rupiah(predicted_price)
    )

    margin = predicted_price * (metadata["metrics"]["mape"] / 100)
    low = predicted_price - margin
    high = predicted_price + margin

    st.caption(
        f"Rentang estimasi (±MAPE {metadata['metrics']['mape']:.1f}%): "
        f"{format_rupiah(low)} – {format_rupiah(high)}"
    )

    with st.expander("Detail Input"):
        st.dataframe(
            input_data.T.rename(columns={0: "Nilai"}),
            width="stretch"
        )
