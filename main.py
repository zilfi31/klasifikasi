import streamlit as st
import joblib
import pandas as pd
from source import *

# preparing model
one_hot_encoder = joblib.load("/Volumes/Data/deployment/klasifikasi/one_hot_encoder.joblib")
model = joblib.load("/Volumes/Data/deployment/klasifikasi/model.joblib")

# prepare result
result = 0


st.title("Klasifikasi Bank")
st.write("Klasifikasi pelanggan sudah berlangganan deposito berjangka untuk memprediksi apakah nasabah akan berlangganan deposito berjangka berdasarkan beberapa parameter.")


col_1, col_2 = st.columns(2)

with col_1:
   opt_age= st.selectbox("Umur", age)
   opt_job = st.selectbox("Pekerjaan", job)
   opt_marital = st.selectbox("Perkawinan ", marital)
   opt_education = st.selectbox("Pendidikan ", education)

with col_2:
   opt_default = st.selectbox("Kredit default", default)
   opt_balance = st.selectbox("Saldo", balance)
   opt_housing = st.selectbox("Perumahan", housing)
   opt_loan = st.selectbox("Pinjaman", loan)

with col_1:
   button_predict = st.button("Prediksi")

with st.container(border=True):
   if(button_predict):

      # inisilisasi dataframe baru 
      df = pd.DataFrame({
         "age": [opt_age],
         "job": [opt_job],
         "marital": [opt_marital],
         "education": [opt_education],
         "default": [opt_default],
         "balance": [opt_balance],
         "housing": [opt_housing],
         "loan": [opt_loan]
         
      })

      # menentukan column categorical dan numerical
      categories = df.select_dtypes(include=['object']).columns.to_list()
      numeric = df.select_dtypes(include=['int64']).columns.to_list()

      # transform categorical dengan one hot encoding
      encoded = one_hot_encoder.transform(df[categories])

      # ubah hasil transform ke dataframe
      one_hot_df = pd.DataFrame(encoded, columns=one_hot_encoder.get_feature_names_out(categories))

      # gabung dataframe numerik dan categorik
      new_df = pd.concat([df[numeric], one_hot_df], axis=1)

      # prediksi dengan model SVM
      predict = model.predict(new_df)[0]
      predict_proba = model.predict_proba(new_df)[0]
      predict_proba = [round(x*100, 2) for x in predict_proba]

      # membuat dataframe metrik probabilitas hasil perhitungan
      probability_metrics = pd.DataFrame({
         "Berlangganan": [f"{predict_proba[0]}%"],
         "Tidak Berlangganan": [f"{predict_proba[1]}%"],
         
      })

      # penentuan kelas
      labels = ['Berlangganan',  'Tidak Berlangganan']

      # tampilkan di web
      st.write(f"Prediksi : {labels[predict]}")
      st.write("Probabilitas Setiap Kelas :")
      st.table(probability_metrics)

