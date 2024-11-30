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
from sklearn.preprocessing import LabelEncoder

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

    elif page == "Preprocessing":
        st.header("Penanganan Missing Value")

        col1, col2 = st.columns([3, 2])

        with col1:
            st.subheader("Data Asli")
            st.dataframe(data, width=1000, height=400)

        with col2:
            st.subheader("Total Missing Value per Kolom")
            missing_values = data.isnull().sum()
            st.dataframe(missing_values.rename("Jumlah Missing Value"))

        missing_proportions = data.isnull().mean()
        high_missing_cols = missing_proportions[missing_proportions > 0.2]
        low_missing_cols = missing_proportions[(missing_proportions > 0) & (missing_proportions <= 0.2)]

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Kolom dengan Missing Value Tinggi (>20%)")
            st.write(high_missing_cols if not high_missing_cols.empty else "Tidak ada.")

        with col4:
            st.subheader("Kolom dengan Missing Value Rendah (<=20%)")
            st.write(low_missing_cols if not low_missing_cols.empty else "Tidak ada.")
        
        # Menampilkan tipe data setiap kolom
        st.write("Tipe data setiap kolom:")
        st.write(data.dtypes)

        # Menampilkan data sebelum preprocessing
        st.write("Data sebelum preprocessing:")
        st.dataframe(data.head())

        if 'rbc' in data.columns:
            data = data.drop('rbc', axis=1)

        imputer_mean = SimpleImputer(strategy='mean')
        imputer_mode = SimpleImputer(strategy='most_frequent')

        # Memperbaiki imputasi berdasarkan tipe data
        for col in high_missing_cols.index:
            if col in data.columns:
                if data[col].dtype in ['float64', 'int64']:  # Periksa tipe data numerik
                    sample_values = data[col].dropna().sample(data[col].isnull().sum(), replace=True).values
                    data.loc[data[col].isnull(), col] = sample_values
                else:
                    st.warning(f"Kolom {col} tidak numerik, dan tidak dapat diisi dengan imputasi numerik.")

        for col in low_missing_cols.index:
            if col in data.columns:
                if data[col].dtype in ['float64', 'int64']:
                    data[col] = imputer_mean.fit_transform(data[[col]]).flatten()
                else:
                    data[col] = imputer_mode.fit_transform(data[[col]]).flatten()
                
        # Pastikan untuk menangani kategori dengan benar
        for col in data.select_dtypes(include=['object']).columns:
            if col in ['wc', 'rc']:  # Tambahkan kolom ini pada pengecekan
                data[col] = imputer_mode.fit_transform(data[[col]]).flatten()  # Gunakan imputasi modus
            else:
                data[col] = data[col].astype(str)  # Ubah kolom kategori menjadi string

        # Kolom-kolom kategorikal yang ingin diencoding
        categorical_columns = ['rbc', 'pc', 'pcc', 'ba', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'classification']

        # Memeriksa kolom mana yang ada dalam dataset
        valid_categorical_columns = [col for col in categorical_columns if col in data.columns]

        # Melakukan One-Hot Encoding hanya pada kolom yang valid
        if valid_categorical_columns:
            new_df = pd.get_dummies(data, columns=valid_categorical_columns, drop_first=True)
        else:
            st.error("Tidak ada kolom kategorikal yang valid ditemukan untuk encoding.")

        # Mengonversi kolom kategorikal menjadi numerik menggunakan LabelEncoder jika diperlukan
        label_encoder = LabelEncoder()
        binary_columns = ['htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'classification']
        
        # Pastikan hanya kolom yang valid ada di dataset
        valid_binary_columns = [col for col in binary_columns if col in new_df.columns]

        for col in valid_binary_columns:
            new_df[col] = label_encoder.fit_transform(new_df[col])

        # Kolom 'wc' dan 'rc' sudah ada di dalam dataset dan perlu imputasi menggunakan modus
        for col in ['wc', 'rc']:
            if col in new_df.columns:
                new_df[col] = imputer_mode.fit_transform(new_df[[col]]).flatten()

        # Pastikan data tidak memiliki nilai NaN setelah imputasi
        numerical_columns = new_df.select_dtypes(include=['float64']).columns
        new_df[numerical_columns] = new_df[numerical_columns].fillna(0)

        # Convert all columns to appropriate types for Arrow compatibility
        numerical_cols = new_df.select_dtypes(include=['object', 'int64', 'float64']).columns
        for col in numerical_cols:
            if new_df[col].dtype == 'object':
                new_df[col] = pd.to_numeric(new_df[col], errors='coerce')

        # Ensure any categorical columns are of type 'category'
        categorical_cols = new_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            new_df[col] = new_df[col].astype(str)  # Convert to string if not already

        # Double-check if any columns contain mixed types that still can't be converted
        new_df = new_df.apply(pd.to_numeric, errors='coerce')

        # Fill NaN values after conversion if necessary
        new_df.fillna(0, inplace=True)

        st.subheader("Data Setelah Perbaikan Tipe dan Imputasi Missing Value")
        st.dataframe(new_df)

        corr_matrix = new_df.corr()
        Dependent_corr = corr_matrix.get('classification', pd.Series())
        Imp_features = Dependent_corr[Dependent_corr.abs() < 0.4].index.tolist()
        if 'id' in Imp_features:
            Imp_features.remove('id')

        st.subheader("Fitur yang Dipilih Berdasarkan Korelasi")
        st.write(Imp_features if Imp_features else "Tidak ada fitur yang memenuhi syarat korelasi.")

    # Halaman Modeling
    elif page == "Modeling":
        st.header("Pelatihan Model")

        if new_df is None or Imp_features is None:
            st.error("Silakan lakukan preprocessing terlebih dahulu!")
            return
        
        X = new_df[Imp_features]
        y = new_df['classification']  # Ensure this is the target column
        
        st.subheader("Pilih Model")
        model_type = st.selectbox("Model", ["Naive Bayes", "Decision Tree"])

        if model_type == "Naive Bayes":
            model = GaussianNB()
        elif model_type == "Decision Tree":
            model = DecisionTreeClassifier()

        model.fit(X, y)

        st.session_state['model'] = model
        st.session_state['X_test'], st.session_state['y_test'] = train_test_split(X, y, test_size=0.2, random_state=42)

        st.success(f"Model {model_type} berhasil dilatih!")

    # Halaman Evaluasi
    elif page == "Evaluasi":
        st.header("Evaluasi Model")

        if 'model' not in st.session_state:
            st.error("Model belum dilatih!")
            return

        model = st.session_state['model']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        st.subheader("Akurasi Model")
        st.write(f"Akurasi: {accuracy * 100:.2f}%")

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
