import streamlit as st
import pandas as pd
import joblib
from model.preprocessing import preprocess_text
from utils.visualisasi import show_pie_chart, generate_wordcloud
import chardet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load model dan vectorizer
try:
    model = joblib.load("model/naive_bayes_model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
except FileNotFoundError:
    st.error("Model or vectorizer files not found. Please ensure they are saved in the 'model' directory.")
    st.stop()

st.set_page_config(page_title="Klasifikasi Sentimen IKN", layout="wide")
st.title("📊 Aplikasi Klasifikasi Sentimen Relokasi Ibu Kota Negara (IKN)")

uploaded_file = st.file_uploader("Upload file CSV komentar", type=["csv"])

if uploaded_file:
    raw_data = uploaded_file.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    uploaded_file.seek(0)

    try:
        df = pd.read_csv(uploaded_file, encoding=encoding)
    except Exception as e:
        st.error(f"Error reading CSV file with encoding {encoding}: {e}")
        st.info("Pastikan file CSV disimpan dalam encoding UTF-8. Jika berasal dari Excel, pilih 'Save As' → 'CSV UTF-8'.")
        st.stop()

    if 'komentar' not in df.columns:
        st.error("Kolom 'komentar' tidak ditemukan. Pastikan file CSV kamu memiliki kolom 'komentar' (huruf kecil semua, tanpa spasi).")
    else:
        try:
            df['cleaned'] = df['komentar'].astype(str).apply(preprocess_text)
        except Exception as e:
            st.error(f"Error during text preprocessing: {e}")
            st.stop()

        try:
            X = vectorizer.transform(df['cleaned'])
            df['sentimen'] = model.predict(X)
        except Exception as e:
            st.error(f"Error during vectorization or prediction: {e}")
            st.stop()

        st.success("Prediksi berhasil!")
        st.dataframe(df[['komentar', 'sentimen']])

        try:
            show_pie_chart(df)
            generate_wordcloud(df)
        except Exception as e:
            st.warning(f"Could not generate visualizations: {e}")

        # ✅ Evaluasi model (jika ada kolom label)
        if 'label' in df.columns:
            st.subheader("🧪 Evaluasi Model (dibandingkan dengan label asli)")
            try:
                y_true = df['label']
                y_pred = df['sentimen']

                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

                st.write(f"**Akurasi:** {acc:.2f}")
                st.write(f"**Precision:** {prec:.2f}")
                st.write(f"**Recall:** {rec:.2f}")
                st.write(f"**F1-score:** {f1:.2f}")

                # Confusion Matrix
                st.markdown("#### Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred, labels=model.classes_)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
                ax.set_xlabel("Prediksi")
                ax.set_ylabel("Label Asli")
                st.pyplot(fig)

                # Classification report (expandable)
                with st.expander("📄 Lihat Classification Report Lengkap"):
                    report = classification_report(y_true, y_pred)
                    st.text(report)

            except Exception as e:
                st.warning(f"Tidak dapat mengevaluasi model: {e}")
        else:
            st.info("Tambahkan kolom 'label' dalam file CSV jika ingin mengevaluasi performa model.")

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Hasil", csv, "hasil_klasifikasi.csv", "text/csv")
