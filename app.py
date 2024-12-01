import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Data Dummy untuk Pelatihan Model
X_dummy = [[80, 1.01, 3, 2, 1, 120, 40, 1.2, 135, 4.5], 
           [70, 1.02, 0, 0, 0, 110, 35, 1.0, 140, 4.0]]
y_dummy = [1, 0]  # 1: CKD, 0: Tidak CKD

# Model Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_dummy, y_dummy)

# Model Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_dummy, y_dummy)

# Fungsi untuk melakukan prediksi
def predict(model, inputs):
    try:
        # Prediksi berdasarkan model yang dipilih
        prediction = model.predict([inputs])[0]
        return prediction
    except Exception as e:
        return f"Error: {e}"

# Struktur Aplikasi Streamlit
st.set_page_config(
    page_title="Prediksi Penyakit Gagal Ginjal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Navigasi
st.sidebar.title("Navigasi")
tabs = st.sidebar.radio("Pages", ["Home", "Prediction", "Visualisation"])

if tabs == "Home":
    # Halaman Utama
    st.title("Home")
    st.write("""
    Selamat datang di aplikasi prediksi penyakit gagal ginjal kronis!
    Aplikasi ini menggunakan model machine learning untuk memprediksi
    kemungkinan seseorang menderita gagal ginjal kronis (Chronic Kidney Disease - CKD)
    berdasarkan data medis yang Anda masukkan.
    """)

elif tabs == "Prediction":
    # Halaman Prediction
    st.title("Halaman Prediksi")

    st.subheader("Masukkan Data untuk Prediksi")
    st.markdown("""
    **Catatan:** Semua nilai harus diisi untuk melakukan prediksi.
    """)

    # Input Form untuk Prediksi
    col1, col2 = st.columns(2)
    with col1:
        bp = st.number_input("Tekanan Darah (bp)", min_value=0, max_value=200, value=80, step=1)
        sg = st.number_input("Kerapatan Urin (sg)", min_value=1.0, max_value=1.05, value=1.02, step=0.01)
        al = st.number_input("Albumin Urin (al)", min_value=0, max_value=10, value=1, step=1)
        su = st.number_input("Gula Urin (su)", min_value=0, max_value=10, value=0, step=1)
        rbc = st.selectbox("Sel Darah Merah (rbc)", options=[0, 1], format_func=lambda x: "Abnormal" if x == 1 else "Normal")
    with col2:
        bgr = st.number_input("Glukosa Darah Acak (bgr)", min_value=50, max_value=500, value=150, step=1)
        bu = st.number_input("Urea Darah (bu)", min_value=0, max_value=200, value=30, step=1)
        sc = st.number_input("Serum Kreatinin (sc)", min_value=0.0, max_value=15.0, value=1.0, step=0.1)
        sod = st.number_input("Sodium (sod)", min_value=100, max_value=200, value=140, step=1)
        pot = st.number_input("Potassium (pot)", min_value=1.0, max_value=10.0, value=4.5, step=0.1)

    # Pilihan Model
    st.subheader("Pilih Model")
    model_choice = st.radio("Model untuk Prediksi", ["Decision Tree", "Naive Bayes"])

    # Tombol Prediksi
    if st.button("Prediksi"):
        # Data Input
        inputs = [bp, sg, al, su, rbc, bgr, bu, sc, sod, pot]

        # Pilih Model
        if model_choice == "Decision Tree":
            selected_model = dt_model
        elif model_choice == "Naive Bayes":
            selected_model = nb_model

        # Lakukan Prediksi
        hasil_prediksi = predict(selected_model, inputs)

        # Tampilkan Hasil
        if hasil_prediksi == 1:
            st.success("Hasil Prediksi: Pasien Menderita CKD (Chronic Kidney Disease)")
        elif hasil_prediksi == 0:
            st.success("Hasil Prediksi: Pasien Tidak Menderita CKD")
        else:
            st.error(hasil_prediksi)

elif tabs == "Visualisation":
    # Placeholder untuk halaman visualisasi
    st.title("Halaman Visualisasi")
    st.write("""
    Halaman ini akan menampilkan visualisasi data jika tersedia.
    Anda dapat menambahkan grafik atau diagram untuk memahami distribusi data.
    """)
