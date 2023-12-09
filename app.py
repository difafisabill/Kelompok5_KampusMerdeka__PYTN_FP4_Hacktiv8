import streamlit as st
import pandas as pd
import requests  
import pickle
from PIL import Image
from io import BytesIO
import os
import shutil



url = 'https://img.freepik.com/free-photo/credit-card-payment-buy-sell-products-service_1150-16379.jpg?w=1060&t=st=1702163338~exp=1702163938~hmac=5fceda1f8326941656494324903ec38383066acd66195b7120c8d19b2f0fa6bd'
response = requests.get(url)
if response.status_code == 200:
    image = Image.open(BytesIO(response.content))
    st.image(image, use_column_width=True)
else:
    st.error(f"Failed to download image. Status code: {response.status_code}")

csv_url = 'https://github.com/difafisabill/Kelompok5_KampusMerdeka__PYTN_FP4_Hacktiv8/raw/main/Dataset/credit_card.csv'

def download_model_from_url(model_url, save_path):
    if model_url.startswith('http'):
        response = requests.get(model_url)
        with open(save_path, 'wb') as file:
            file.write(response.content)
    else:
        shutil.copy(model_url, save_path)

st.markdown("<h1 style='text-align: center;'>Clustering Credit Card</h1>",
            unsafe_allow_html=True)
st.markdown("Data ini sudah melalui proses data cleaning dan preprosessing sehingga siap untuk dilakukan pemodelan")

tab1, tab2, tab3 = st.tabs(["Dataset", "Dog", "Owl"])


def main():
    

    @st.cache_resource
    def load_data():
        data = pd.read_csv(csv_url)
        return data
    
    with tab1:
        st.header("Dataset")
        data = load_data()
        check_box = st.checkbox("Show Dataset")
        if (check_box):
            st.markdown("#### Credit Card Dataset")
            st.write(data)

    with tab2:
        st.header("A dog")
        st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

    with tab3:
        st.header("An owl")
        st.image("https://static.streamlit.io/examples/owl.jpg", width=200)



    

if __name__ == '__main__':
    main()
