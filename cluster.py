import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

'''
Used simple imputation to fill missing values in one run of K means.
Used multiple imputation with 5 datasets for another run. 
'''

def data_simp():
    '''
    Function to prep data for kmeans with simple imputation. Several subsequent functions call this.
    Dropped: dataset (will split manually), gender (categorical), tr_peak and echopasp (identical to RVSP).
    '''

    file = 'clin_pro_echo.csv'
    df = pd.read_csv(file)

    df.drop(['dataset', 'gender1', 'tr_peak', 'echopasp'], inplace=True, axis=1)
    # df['gender1'].replace(to_replace=['MALE', 'FEMALE',], value=[0, 1], inplace=True) # gender to numeric (male=0,female=1)

    # df = df[['echorap','idno','echo_hr']]

    # 80/20 train/test split
    train = df.sample(frac=0.8) # previously set random_state =2
    test = df.drop(train.index)

    train_ids = list(train.pop('idno'))  # pull out ids to add back later as index
    test_ids = list(test.pop('idno'))
    df.drop('idno', inplace=True, axis=1)

    # impute mean, scale data
    simp_imp = SimpleImputer(strategy='mean').fit(train)
    trans_train = simp_imp.transform(train)
    trans_test = simp_imp.transform(test)

    scaler = StandardScaler().fit(trans_train)
    scaltrans_train = scaler.transform(trans_train)
    scaltrans_test = scaler.transform(trans_test)

    # make df again
    clean_train = pd.DataFrame(scaltrans_train, columns=df.columns)
    clean_train.insert(0, 'idno', train_ids)
    clean_train.set_index('idno', inplace=True)

    clean_test = pd.DataFrame(scaltrans_test, columns=df.columns)
    clean_test.insert(0, 'idno', test_ids)
    clean_test.set_index('idno', inplace=True)

    # code for missing values, correlations
    # df.isnull().sum()/len(df) * 100 # % missing values by column
    # train.corr().abs()

    return clean_train, clean_test

def plots(data, range_, type_):
    '''
    Produce plot of x = cluster number, y = inertia for given number of clusters.
    Goal is to find the elbow.
    '''

    k_list, inertia, sil = [], [], []

    for n in range(2,range_):
        kmeans = KMeans(n_clusters=n, random_state=0, max_iter=300, n_init=10).fit(data)
        k_list.append(n)
        inertia.append(kmeans.inertia_)
        labels = kmeans.labels_
        sil.append(silhouette_score(data, labels, metric='euclidean'))

    if type_ == 'sil':
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Silhouette Score")
        plt.plot(k_list, sil, 'ro-', color='blue')
        plt.show()

    elif type_ == 'inertia':
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Inertia")
        plt.xlim(1, 14)
        # plt.ylim(140000, 210000)
        plt.plot(k_list, inertia, 'ro-', color='blue')
        plt.show()

def sil_viz(data):
    fig, ax = plt.subplots(2, 2, figsize=(15, 8))

    for i in [2, 3, 4, 5]:
        plt.xlim(0, 1)
        km = KMeans(n_clusters=i, init='k-means++', n_init=100, max_iter=1000, random_state=42)
        q, mod = divmod(i, 2)
        visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q - 1][mod])
        visualizer.fit(data)

    visualizer.show()

def kmeans_simp(n_clusters,plot, random_obs = None):
    '''
    Simple imputation.
    Pick number of clusters.
    Train on train set, predict test set, output labels.
    Switch for PCA plot (dim = 2).
    '''

    train, test = data_simp()

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, max_iter=300, n_init=10).fit(train)
    predict_test = kmeans.predict(test)
    test.insert(0, 'label', predict_test)

    # test.to_csv('kmeans_simple_testoutput.csv')

    if random_obs != None:
        print(test['label'].iloc[random_obs])

    if plot == True:
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(test)

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

def data_mimp():
    '''
    Takes 5 datasets from multiple imputations.
    Creates 2 dictionaries with data. Keys are mimp1 - mimp5, value is corresponding df.
    Train_dict, test_dict.
    Creates 1 dictionary with test set ids and empty list to later populate k values from each of 5 mimp sets.
    '''

    file = 'multimp/mimp.xlsx'
    file2 = 'MESA-echo.csv'

    xls = pd.ExcelFile(file)
    all_sheets = xls.sheet_names

    data = dict()

    for sheet in all_sheets:
        data[f'{sheet}'] = pd.read_excel(xls, sheet_name=sheet)

    # remove gender (cat), tr_peak/echopasp (highly correlated w/ rvsp)
    orig = pd.read_csv(file2)
    orig.drop(['dataset', 'gender1', 'idno', 'tr_peak', 'echopasp'], inplace=True, axis=1)

    # split into training data, test data, and create dict that will hold labels for test set
    train_dict = dict()
    test_dict = dict()
    test_labels = dict()

    random_state = random.randint(2,200) # different random state each time, but same for each df in dictionary

    # prep all dfs in train/test
    for df in data:
        data[df].drop(['dataset', 'gender1', 'tr_peak', 'echopasp'], inplace=True, axis=1)
        ids = data[df].pop('idno')
        scaled_data = StandardScaler().fit_transform(data[df])
        clean_df = pd.DataFrame(scaled_data, columns=orig.columns)
        clean_df.insert(0, 'idno', ids)  #
        clean_df.set_index('idno', inplace=True)
        train = clean_df.sample(frac=0.8, random_state=random_state)
        test = clean_df.drop(train.index)
        train_dict[df] = train
        test_dict[df] = test

    for i,row in test_dict['mimp1'].iterrows(): # filling the dict with test indices and empty list
        test_labels[i] = list()

    return train_dict, test_dict, test_labels

def kmeans_mimp(train_dict, test_dict, test_labels, n_clusters):
    '''
    For each of 5 datasets:
        Train on 80%.
        Make prediction on remaining 20%.
        Store prediction in dictionary with idno as key, values as list with all 5 k values.
        Will pick max k value as the predicted k value for given test id.
    '''

    for df in train_dict:
        kmeans = KMeans(n_clusters=n_clusters, init = 'random', random_state=2, n_init = 100, max_iter=1000).fit(train_dict[df])
        # kmeans = GaussianMixture(n_components=3, random_state=2, max_iter=10000, tol=1e-4, init_params='kmeans').fit(train_dict[df])
        predict_test = kmeans.predict(test_dict[df])
        test_dict[df].insert(0, 'label', predict_test)

        for i,row in test_dict[df].iterrows():
            test_labels[i].append(row['label'])

    return test_dict, test_labels # returns

def mimp_count(test_labels):
    '''
    Counts number of times the predicted class label for any given observation has two classes with 2 values, which
    would result in us needing to make a random assignment (ie [2,2,1,1,0], need to randomly choose 2 or 1.
    Suggestive of instability and unclusterable data.
    '''

    count = 0
    for each in test_labels:
        labels = test_labels[each]
        max_ = max(labels, key=labels.count)
        if labels.count(max_) < 3:
            # print(labels)
            count = count + 1
    print(count)

def pca_plot():
    train, test = data_simp()
    print(train.shape)
    pca = PCA()
    pca.fit(train)
    cumsum = np.cumsum(pca.explained_variance_ratio_)

    fig = plt.figure(figsize=(8, 8))
    plt.plot(cumsum)
    plt.show()

def dim_red(n_components,plot):
    train = data_simp()[0]
    pca = PCA(n_components=n_components)
    train_red = pca.fit_transform(train)

    if plot == True:
        plots(train_red, range_=20, type_ = 'inertia')
        plots(train_red, range_=20, type_ = 'sil')

    return train_red

def affin(random_obs = None):
    train, test = data_simp()

    cluster = AffinityPropagation(random_state = 5).fit(train)
    predict_test = cluster.predict(test)
    test.insert(0, 'label', predict_test)

    if random_obs != None:
        print(test['label'].iloc[random_obs])

def aggom(random_obs = None):
    train, test = data_simp()

    cluster = AgglomerativeClustering(n_clusters=4, affinity='cosine', linkage='complete').fit(test)
    test.insert(0, 'label', cluster.labels_)

    if random_obs != None:
        print(test['label'].iloc[random_obs])

if __name__ == '__main__':
    # plot inertia, silhouette score for simple imputation -> inflection point is 3 or 4?
    # train_simp = data_simp()[0]
    # plots(train_simp, range_=10, type_ = 'inertia')
    # plots(train_simp, range_=10, type_ = 'sil')

    # data = dim_red(n_components=0.8,plot=False)
    # plots(data, range_=10, type_ = 'inertia')
    # plots(data, range_=10, type_ = 'sil')
    # sil_viz(data)

    '''
    Points for future discussion: 
    - What to do with RAP/PASP/RVSP -> 21% missing 
    - What to do with RAP and gender -> categorical in general for clustering 
    - Finite mixture model to accommodate categorical variables? How can we address this with a clustering algorithm?
    '''




