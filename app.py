import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Fungsi Utama
def main():
    st.set_page_config(
        page_title="Klasifikasi Penyakit Gagal Ginjal Kronis",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
        <style>
            .main { background-color: #f0f2f6; }
            h1, h2, h3 { color: #4CAF50; }
            .stButton>button { background-color: #4CAF50; color: white; }
        </style>
    """, unsafe_allow_html=True)

    st.title("Klasifikasi Penyakit Gagal Ginjal Kronis")

    # Global variables for data sharing
    global new_df, Imp_features, X, y, model
    new_df, Imp_features, X, y, model = None, None, None, None, None

    try:
        data = pd.read_csv('data.csv')
        st.success("Dataset berhasil dimuat!")
    except FileNotFoundError:
        st.error("File 'data.csv' tidak ditemukan. Harap unggah dataset terlebih dahulu.")
        return

    # Sidebar Navigation
    page = st.sidebar.selectbox("Navigasi", ["Data", "Preprocessing", "Modeling", "Evaluasi"])

    # Halaman Data
    if page == "Data":
        st.header("Data Gagal Ginjal Kronis")

        st.subheader("Data Asli")
        st.dataframe(data, height=400)

        st.subheader("Dimensi Data")
        st.write(f"Jumlah Baris dan Kolom: {data.shape}")

        st.subheader("Informasi Data")
        buffer = StringIO()
        data.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

    # Halaman Preprocessing
    elif page == "Preprocessing":
        st.header("Preprocessing Data")

        col1, col2 = st.columns([3, 2])

        with col1:
            st.subheader("Data Asli")
            st.dataframe(data)

        with col2:
            st.subheader("Jumlah Missing Values per Kolom")
            missing_values = data.isnull().sum()
            st.dataframe(missing_values.rename("Missing Values"))

        # Handle Missing Values
        imputer_mean = SimpleImputer(strategy='mean')
        imputer_mode = SimpleImputer(strategy='most_frequent')

        for col in data.columns:
            if data[col].isnull().sum() > 0:
                if data[col].dtype in ['float64', 'int64']:
                    data[col] = imputer_mean.fit_transform(data[[col]])
                else:
                    data[col] = imputer_mode.fit_transform(data[[col]])

        # One-hot Encoding
        new_df = pd.get_dummies(data, drop_first=True)
        st.subheader("Data Setelah Preprocessing")
        st.dataframe(new_df)

        # Korelasi dan Fitur Penting
        st.subheader("Fitur Penting Berdasarkan Korelasi")
        corr_matrix = new_df.corr()
        target_corr = corr_matrix.get('classification_notckd', pd.Series())
        Imp_features = target_corr[target_corr.abs() > 0.4].index.tolist()

        if 'id' in Imp_features:
            Imp_features.remove('id')

        st.write(Imp_features if Imp_features else "Tidak ada fitur penting yang ditemukan.")

        # Split Features dan Target
        if Imp_features:
            X = new_df[Imp_features]
            y = new_df['classification_notckd']
            st.write("Contoh Data Fitur (X):")
            st.dataframe(X.head())
            st.write("Contoh Data Target (y):")
            st.dataframe(y.head())
        else:
            st.error("Fitur penting tidak ditemukan. Proses berhenti di sini.")

    # Halaman Modeling
    elif page == "Modeling":
        st.header("Modeling")

        if X is None or y is None:
            st.error("Harap lakukan preprocessing terlebih dahulu!")
            return

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Pilih Model
        model_type = st.selectbox("Pilih Model", ["Naive Bayes", "Decision Tree"])

        if model_type == "Naive Bayes":
            model = GaussianNB()
        elif model_type == "Decision Tree":
            model = DecisionTreeClassifier()

        # Latih Model
        model.fit(X_train, y_train)
        st.success(f"Model {model_type} berhasil dilatih!")

        # Simpan Model untuk Evaluasi
        st.session_state['model'] = model
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test

    # Halaman Evaluasi
    elif page == "Evaluasi":
        st.header("Evaluasi Model")

        if 'model' not in st.session_state:
            st.error("Harap lakukan modeling terlebih dahulu!")
            return

        model = st.session_state['model']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']

        # Prediksi
        y_pred = model.predict(X_test)

        # Evaluasi
        st.subheader("Hasil Evaluasi")
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"**Akurasi:** {accuracy:.2f}")

        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).T)

        st.subheader("Confusion Matrix")
        conf_matrix = confusion_matrix(y_test, y_pred)
        st.write(conf_matrix)

        # Visualisasi Confusion Matrix
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

if __name__ == '__main__':
    main()
