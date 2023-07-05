import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import pandas as pd
import seaborn as sns



st.title("Web Klasterisasi Pengunjung Mall\nKlastering dengan K-Means\n")

df = pd.read_csv('data-mall.csv')

df.rename(index=str, columns={
    'Annual Income (k$)' : 'Income',
    'Spending Score (1-100)' : 'Score'
}, inplace = True)

X = df.drop(['CustomerID', 'Gender'], axis=1)

print(X)

st.header("isi dataset")
st.write(X)

#menampilkan elbow
from matplotlib.patches import ArrowStyle
clusters=[]
for i in range(1,11):
  km = KMeans(n_clusters=i).fit(X)
  clusters.append(km.inertia_)

fig, ax = plt.subplots(figsize=(12,8))
sns.lineplot(x=list(range(1,11)),y=clusters, ax=ax)
ax.set_title('pencarian elbow')
ax.set_xlabel('cluster')
ax.set_ylabel('inertia')

#panah elbow
ax.annotate('Possible elbow point', xy=(2, 1900000), xytext=(2, 50000), xycoords='data',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))

ax.annotate('Possible elbow point', xy=(3, 80000), xytext=(2, 2500000), xycoords='data',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))

#menampilkan panah elbow di streamlit
st.set_option('deprecation.showPyplotGlobalUse', False)
elbo_plot = st.pyplot()

#membuat side bar untuk menentukan nilai k klastering
st.sidebar.subheader('Nilai jumlah k')
clust = st.sidebar.slider("Pilih Jumlah Klaster (k) : ", 2,10,3,1)

#fungsi menentukan nilai k klastering
def k_means(n_clust):
    kmean = KMeans(n_clusters=n_clust).fit(X)
    X['Labels'] = kmean.labels_

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X['Income'], y=X['Score'], hue=X['Labels'], markers=True, size=X['Labels'], palette=sns.color_palette('hls', n_clust))

#keterangan
    for label in X['Labels']:
        plt.annotate(label,
            (X[X['Labels']==label]['Income'].mean(),
            X[X['Labels']==label]['Score'].mean()),
            horizontalalignment ='center',
            verticalalignment='center',
            size=20, weight='bold',
            color='black')
    
#menampilkan di streamlit
    st.header('Cluster Plot')
    st.pyplot()
    st.write(X)

#yg diambil dari slider
k_means(clust)

