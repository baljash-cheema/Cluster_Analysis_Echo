import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

global file
file = 'echo.csv'  # choose appropriate csv

def data_prep():
    '''
    Function to prep data.
    Output: (train,test)
    '''

    df = pd.read_csv(file)
    df.drop(['dataset', 'gender1'], inplace=True, axis=1) #
    # df['gender1'].replace(to_replace=['MALE', 'FEMALE',], value=[0, 1], inplace=True) # gender to numeric (male=0,female=1)

    # 80/20 train/test split
    train = df.sample(frac=0.8, random_state=2) #can remove random state if desired
    test = df.drop(train.index)

    # code for missing values, correlations
    # df.isnull().sum()/len(df) * 100 # % missing values by column
    # print(train.corr().abs())

    train_ids = list(train.pop('idno'))  # pull out ids to add back later as index
    test_ids = list(test.pop('idno'))
    df.drop('idno', inplace=True, axis=1)

    simp_imp = SimpleImputer(strategy='mean').fit(train) #impute missing values as mean of training set
    trans_train = simp_imp.transform(train)
    trans_test = simp_imp.transform(test)

    scaler = MinMaxScaler().fit(trans_train) #scale based on training data, with mean 0 and var 1
    scaltrans_train = scaler.transform(trans_train)
    scaltrans_test = scaler.transform(trans_test)

    # make df again
    clean_train = pd.DataFrame(scaltrans_train, columns=df.columns)
    clean_train.insert(0, 'idno', train_ids)
    clean_train.set_index('idno', inplace=True)

    clean_test = pd.DataFrame(scaltrans_test, columns=df.columns)
    clean_test.insert(0, 'idno', test_ids)
    clean_test.set_index('idno', inplace=True)

    return clean_train, clean_test

def pca_variance():
    '''
    Plots explained variance ratio over all the variables.
    '''

    train = data_prep()[0]
    pca = PCA()
    pca.fit(train)
    cumsum = np.cumsum(pca.explained_variance_ratio_)

    fig = plt.figure(figsize=(8, 8))
    plt.plot(cumsum)
    plt.xlabel('PCA features')
    plt.ylabel('Cumulative Variance Explained')
    plt.show()

    fig = plt.figure(figsize=(8, 8))
    features = range(pca.n_components_)
    plt.bar(features, pca.explained_variance_ratio_, color='black')
    plt.xlabel('PCA features')
    plt.ylabel('Variance Explained')
    plt.show()

def dim_red(n_components):
    '''
    Returns train, test dfs after dimensionality reduction.
    Need this since we have many features and are using Euclidean distance.
    '''

    train,test = data_prep()
    train_ids = list(train.index.values)
    test_ids = list(test.index.values)

    pca = PCA(n_components=n_components)
    train_red = pca.fit_transform(train)
    test_red = pca.transform(test)

    train_red = pd.DataFrame(train_red)
    test_red = pd.DataFrame(test_red)

    train_red.insert(0, 'idno', train_ids)
    train_red.set_index('idno', inplace=True)

    test_red.insert(0, 'idno', test_ids)
    test_red.set_index('idno', inplace=True)

    return train_red, test_red

def plots(data, range_):
    '''
    Output silhouette and inertia plots, for number of k specified by range.
    '''

    k_list, inertia, sil = [], [], []

    for n in range(2,range_):
        kmeans = KMeans(n_clusters=n, random_state=2, max_iter=300, n_init=10).fit(data)
        k_list.append(n)
        inertia.append(kmeans.inertia_)
        labels = kmeans.labels_
        sil.append(silhouette_score(data, labels, metric='cosine'))

    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.plot(k_list, sil, 'ro-', color='blue')
    plt.show()

    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.xlim(1, 14)
    plt.plot(k_list, inertia, 'ro-', color='blue')
    plt.show()

def sil_viz(data):
    '''
    Silhouette visualizer plot.
    Dashed line is average silhouette score.
    Find clusters that are of similar size (size represents heterogeneity of data within cluster).
    '''
    fig, ax = plt.subplots(2, 2, figsize=(15, 8))

    for i in [2, 3, 4, 5]:
        plt.xlim(0, 1)
        km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=2)
        q, mod = divmod(i, 2)
        visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q - 1][mod])
        visualizer.fit(data)

    visualizer.show()

def kmeans(train, test, n_clusters,plot, random_obs = None, pca_=False):
    '''
    Uses data prep function to generate data.
    Plot with 2 dimensions using PCA.
    Outputs test labels to original data set and provides means per cluster.
    '''

    if pca_ == True:
        qualifier = 'red'
    else:
        qualifier = ''

    df_orig = pd.read_csv(file)
    train_orig = df_orig.sample(frac=0.8, random_state=2)
    test_orig = df_orig.drop(train_orig.index)

    kmeans = KMeans(n_clusters=n_clusters, random_state=2, max_iter=300, n_init=10).fit(train)
    predict_test = kmeans.predict(test)
    test_orig.insert(0, 'label', predict_test)

    test_orig.to_csv(f'output_{qualifier}_{file}')
    meandf = test_orig.groupby(by='label').mean()
    meandf.to_csv(f'mean_{qualifier}_{file}')

    k_out = []

    if random_obs != None:
        for each in random_obs:
            k_out.append(test['label'].iloc[each])
        print(k_out)

    if plot == True:
        pca = PCA(n_components=2)
        pca.fit(train)
        pca_data = pca.transform(test)

        pca1 = list()
        pca2 = list()

        for each in pca_data:
            pca1.append(each[0])
            pca2.append(each[1])

        pca1 = np.array(pca1)
        pca2 = np.array(pca2)

        test.insert(1, 'pca1', pca1)
        test.insert(2, 'pca2', pca2)

        fig = plt.figure(figsize=(8, 8))
        plt.scatter(
            x=pca1,
            y=pca2,
            c=predict_test,
            cmap='viridis')
        plt.xlabel('PCA1')
        plt.ylabel('PCA2')
        plt.title('Clusters with Dimensionality Reduction')
        plt.show()

def metrics_(data, k_range):
    '''
    Calculates Calinksi-Harabasz index (higher score is means more dense and well-separated clusters)
    and Davies-Bouldin index (lower score means better separation between clusters).
    '''

    cal = []
    dav = []

    for n in np.arange(2, k_range+1):
        kmeans_model = KMeans(n_clusters=n, random_state=2).fit(data)
        labels = kmeans_model.labels_
        cal.append((n,calinski_harabasz_score(data, labels)))
        dav.append((n,davies_bouldin_score(data, labels)))

    print(f'Calinski-Harabasz Index: {cal}')
    print(f'Davies-Bouldin Scores: {dav}')

def corr_matrix(threshold, data):
    '''
    Prints pairs of variables with a correlation coefficient greater than threshold.
    '''

    corr_matrix = data.corr().abs()
    high_corr_var = np.where(corr_matrix > threshold)
    high_corr_var = [(corr_matrix.columns[x], corr_matrix.columns[y]) for x, y in zip(*high_corr_var) if
                     x != y and x < y]

    df = pd.DataFrame(high_corr_var)
    print(f'Total pairs: {df.shape[0]}')
    print(df.head(50))

def pca_analysis(train,n_components):

    pca = PCA(n_components=n_components)
    fit = pca.fit_transform(train)
    pca_df = pd.DataFrame(pca.components_, columns=list(train.columns))

    print(pca_df)

if __name__ == '__main__':

    # df = pd.read_csv('clin_echo_outcomes.csv')
    # print(df.groupby(by='dataset').count())

    # # Data is highly correlated
    # train, test = data_prep()
    # pca_analysis(train,2)
    # corr_matrix(threshold=0.9,data=train)
    # pca_variance()

    # # Without dimensionality reduction
    # train, test = data_prep()
    # print(train.shape)
    # plots(train,range_=20)
    # sil_viz(train)
    # metrics_(train,k_range=10)
    # kmeans(train=train, test=test, n_clusters=2, plot=True, random_obs=None, pca_ = False)
    # train, test = data_prep()
    # kmeans(train=train, test=test, n_clusters=3, plot=True, random_obs=None, pca_ = False)
    # train, test = data_prep()
    # kmeans(train=train, test=test, n_clusters=4, plot=True, random_obs=None, pca_ = False)

    # # With dimensionality reduction
    # pca_variance()
    # train_red, test_red = dim_red(n_components=0.6)
    # print(train_red.shape)
    # plots(train_red,range_=20)
    # sil_viz(train_red)
    # metrics_(train_red,k_range=10)
    # kmeans(train=train_red, test=test_red, n_clusters=2,plot=True,random_obs=None, pca_=True)
    # train_red, test_red = dim_red(n_components=10)
    # kmeans(train=train_red, test=test_red, n_clusters=3,plot=True,random_obs=None, pca_=True)
    # train_red, test_red = dim_red(n_components=0.8)
    # kmeans(train=train_red, test=test_red, n_clusters=4,plot=True,random_obs=None, pca_=True)

    '''
    - Optimal K -> 3
    - Explain most important features -> SHAP library
    - Run mixed k means with categorical variables (gender) 
    - Or adjust for height/gender through regression 
    - What variables are correlated with PCA 1/2?
    '''
