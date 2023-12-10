import streamlit as st
import pandas as pd
import requests  
from PIL import Image
from io import BytesIO
import shutil
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



url = 'https://github.com/difafisabill/Kelompok5_KampusMerdeka__PYTN_FP4_Hacktiv8/blob/main/credit-card-payment-buy-sell-products-service.jpg?raw=true'
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

tab1, tab2, tab3 = st.tabs(["Dataset", "Model", "Cluster"])


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
        st.markdown("#### Saldo Desil")
        df=data.copy()
        df['saldo_decile'] = pd.qcut(df['saldo'], q=10).astype(str)

        data_grp   = df.groupby('saldo_decile', as_index=False).mean()
        data_grp   = data_grp[['saldo_decile', 'pembelian', 'pembelian_sekaligus', 'pembelian_angsuran']]
        data_grp_t = pd.melt(data_grp, id_vars = 'saldo_decile')
        fig = px.bar(
            data_grp_t, 
            x='saldo_decile', 
            y='value', 
            color='variable', 
            barmode='group', 
            labels={'value': 'Jumlah Pembelian Rata-rata', 'saldo_decile': 'Saldo Groups'},
        )

        fig.update_layout(
            width=800, 
            height=500, 
            xaxis_title='Saldo Groups',
            yaxis_title='Jumlah Pembelian Rata-rata',
        )
        st.plotly_chart(fig)
        with st.expander("Lihat Penjelasan"):
            st.write("""
                     Pelanggan dengan saldo akun yang yang lebih rendah memiliki frekuensi pembelian yang lebih tinggi dibandingkan dengan pelanggan dengan saldo lebih tinggi dalam rentang persentil 50–75.
           
                    """)
            st.markdown("1. **Saldo di kisaran menengah:**  Merujuk kepada pelanggan yang memiliki saldo akun dalam kisaran pertengahan dari dataset.")
            st.markdown("2. **Memiliki lebih banyak pembelian:** Menunjukkan bahwa pelanggan dengan saldo di kisaran menengah cenderung melakukan lebih banyak pembelian.") 
            st.markdown("3. **Persentil 50–75:** Pelanggan dengan saldo menengah melakukan lebih banyak pembelian dibandingkan dengan mereka yang memiliki saldo lebih tinggi dalam rentang persentil yang disebutkan.") 
            
        sns.pairplot(data=data, x_vars='saldo', y_vars='pembelian',
             height=6, aspect=1.5).map(sns.kdeplot, levels=1, color='red')
        plt.title("Hubungan antara saldo dan pembelian", fontweight='bold', fontsize=15)
        st.write("""
                     Refresh sekalilagi jika plot tidak muncul
           
                    """)
        st.pyplot(plt) 
        with st.expander("Lihat Penjelasan"):
            st.write("""
                     Berdasarkan plot yang ditampilkan dapat disimpulkan bahwa mayoritas pengguna melakukan pembelian di bawah angka 10.000.
                    """)

        with tab2:
            st.header("Model Building using KMeans")
            scalar=StandardScaler()
            scaled_df = scalar.fit_transform(data)
            clusters = []
            for i in range(1, 11):
                km = KMeans(n_clusters=i).fit(scaled_df)
                clusters.append(km.inertia_)

            fig, ax = plt.subplots(figsize=(12, 8), sharex=True)
            sns.lineplot(x=list(range(1, 11)), y=clusters, ax=ax)
            ax.set_title("Mencari elbow")
            ax.set_xlabel('clusters')
            ax.set_ylabel('inertia')
            st.pyplot(fig)
            with st.expander("Lihat Penjelasan"):
                st.write("""Berdasarkan hasil metode elbow  dapat disimpulkan bahwa jumlah clustering terbaik untuk algoritma K-Means adalah 4 cluster. Sehingga dataset ini akan dibagi menjadi 4 cluster, Kemudian untuk membuat  dataset lebih sederhana agar dapat dengan mudah dilakukan pemodelan K-Mean digunakan metode PCA unutk mereduksi dimensi""")
                pca = PCA(n_components=2)
                principal_components = pca.fit_transform(scaled_df)
                pca_df = pd.DataFrame(data=principal_components ,columns=["PCA1","PCA2"])
                st.dataframe(pca_df)
                st.write("""Dalam model ini PCA yang diambil hanya 2 komponen utama. Data PCA dapat  dilihat diatas""")

            kmeans_model=KMeans(4)
            kmeans_model.fit_predict(scaled_df)
            pca_df_kmeans= pd.concat([pca_df,pd.DataFrame({'cluster':kmeans_model.labels_})],axis=1)

            plt.figure(figsize=(8,8))
            ax=sns.scatterplot(x="PCA1",y="PCA2",hue="cluster",data=pca_df_kmeans,palette=['red','green','blue','black'])
            plt.title("Clustering using K-Means Algorithm")
            st.pyplot(plt)
            with st.expander("Lihat Penjelasan"):
                st.write("""Dibandingkan cluster lainnya, cluster 1 dan 3 memiliki viskositas yang lebih tinggi. Hal ini karena sebagian besar titik data terletak di sudut kiri bawah plot sebar. Selain itu, algoritma K-Means mengasumsikan bahwa data outlier merupakan bagian dari cluster 0 dan 2, dimana outlier pada sumbu x merupakan bagian dari cluster 0, dan outlier pada sumbu y merupakan bagian dari cluster 2. """)
        

    with tab3:
        st.header("Clustering")
        st.markdown("#### Centroid")
        cluster_centers = pd.DataFrame(data=kmeans_model.cluster_centers_,columns=[data.columns])
        cluster_centers = scalar.inverse_transform(cluster_centers)
        cluster_centers = pd.DataFrame(data=cluster_centers,columns=[data.columns])
        st.dataframe(cluster_centers)
        with st.expander("Lihat Penjelasan"):
            st.write("""Cluster center atau centroid ini merupakan representasi titik tengah atau rata-rata dari seluruh titik data dalam tiap cluster, untuk menetapkan cluster yang mewakili tiap data sesuai dengan centroid terdekat. """)

        st.markdown("#### Data Tiap Cluster")
        cluster_df = pd.concat([df,pd.DataFrame({'Cluster':kmeans_model.labels_})],axis=1)
        st.dataframe(cluster_df)

        fig = px.histogram(cluster_df, x='Cluster', color='Cluster', title='Cluster Distribution',
                   labels={'Cluster': 'Cluster'}, color_discrete_map={0: 'red', 1: 'green', 2: 'blue', 3: 'black'})
        st.plotly_chart(fig)
        with st.expander("Lihat Penjelasan"):
            st.write("""Disini dapat dilihat bahwa data cluster 1 memiliki data paling banyak yyang kemudian diikuti dengan cluster 3. Dan cluster yang memiliki data paling sedikit adalah cluster 0 """)

       

       


    

if __name__ == '__main__':
    main()
