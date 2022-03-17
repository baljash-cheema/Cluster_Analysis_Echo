import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from cluster_simp import plots
from cluster_simp import sil_viz
from cluster_simp import corr_matrix
from cluster_simp import metrics_
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from kmodes.kprototypes import KPrototypes

def data_split(file, outcomes, dropped):
    '''
    Takes dataset, removes outcomes col and other cols desired to drop.
    :param file: csv file with data
    :param outcomes: outcome columns from csv file
    :param dropped: columns to drop
    :return: outcomes_df, dropped_df, train_df, val_df, test_df
    '''

    # full dataframe, keep index separated for later
    orig_df = pd.read_csv(file, index_col='idno')
    index = orig_df.index

    # keep those cols as separate df in case needed
    outcomes_df = pd.DataFrame(orig_df[outcomes + ['dataset']], index=index)
    outcomes_df.chf.replace(('Yes','No'),(1,0),inplace=True)
    outcomes_df.chdh.replace(('Yes','No'),(1,0),inplace=True)
    outcomes_df.chda.replace(('Yes','No'),(1,0),inplace=True)

    dropped_df = pd.DataFrame(orig_df[dropped + ['dataset']], index=index)
    dropped_df.gender1.replace(('FEMALE','MALE'),(1,0),inplace=True) # FEMALE = 1

    # drop outcomes and drop cols from original df
    orig_df.drop(outcomes+dropped, axis=1, inplace=True)

    # group by dataset, then get group for each train, val, test
    train_df = orig_df.groupby('dataset').get_group(0)
    val_df = orig_df.groupby('dataset').get_group(1)
    test_df = orig_df.groupby('dataset').get_group(2)

    # loop through to drop dataset col from each
    for each in train_df, val_df, test_df:
        each.drop('dataset', axis=1, inplace=True)

    return outcomes_df, dropped_df, train_df, val_df, test_df

def data_prep(df, imputer, scaler):
    '''
    Takes df, imputer object, scaler object, and returns prepped df.
    :param df: train, val, or test
    :param imputer: imputer object (must be fit on training set)
    :param scaler: scaler object (must be fit on training set)
    :return: prepped dataframe
    '''

    df_ = df.copy()
    ids = df_.index

    df_ = imputer.transform(df_)
    df_ = scaler.transform(df_)

    prepped_df = pd.DataFrame(df_,columns=df.columns)
    prepped_df.insert(0, 'idno', ids)
    prepped_df.set_index('idno', inplace=True)

    return prepped_df

def pca_variance(df):
    '''
    Takes a df and shows what percent of variation is explained by given number of principal components.
    :param df: dataframe
    :return: None
    '''

    pca = PCA()
    pca.fit(df)
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

    return None

def dim_red(df, pca):
    '''
    Takes a df and returns a version of it after PCA.
    :param df: input df
    :param pca: pca object
    :return: df after dimensionality reduction
    '''

    index = df.index
    dr = pca.transform(df)

    df_dr = pd.DataFrame(dr, index=index)

    return df_dr

def kmeans(train, val_test, n_clusters, plot):
    '''
    Trains kmeans classifier on training data, predicts on val/test data.
    :param train: training data
    :param val: validation or test data
    :param n_clusters: number clusters
    :param plot: if true will make plot in 2D (after PCA)
    :return: predict_val_test, n_clusters, predict_train
    '''

    kmeans = KMeans(n_clusters=n_clusters, random_state=2, max_iter=300, n_init=10).fit(train)
    predict_val_test = kmeans.predict(val_test)
    predict_train = kmeans.predict(train)

    if plot == True:
        pca = PCA(n_components=2)
        pca.fit(train)
        pca_data = pca.transform(val_test)

        pca1 = list()
        pca2 = list()

        for each in pca_data:
            pca1.append(each[0])
            pca2.append(each[1])

        pca1 = np.array(pca1)
        pca2 = np.array(pca2)

        val_test.insert(1, 'pca1', pca1)
        val_test.insert(2, 'pca2', pca2)

        fig = plt.figure(figsize=(8, 8))
        plt.scatter(
            x=pca1,
            y=pca2,
            c=predict_val_test,
            cmap='viridis')
        plt.xlabel('PCA1')
        plt.ylabel('PCA2')
        plt.title('K-Means Clustering with Dimensionality Reduction for Visualization')
        plt.show()

    return (predict_val_test, n_clusters, predict_train)

def output(predicts, train_data, val_test_data, outcomes, drop, test):
    '''
    Takes predictions and appends them to orignal data.
    Does mean by group.
    Outputs to CSV.
    :param predicts: tuple -> 0 = val_test cluster predicts, 1 = num clusters, 2 = train cluster predicts
    :param train_data: original training data
    :param val_data: original val/test data
    :param test: if True, file names will say test, otherwise will say validation
    :return: None
    '''

    outs_cont = ['total6','kccq126c','label']
    outs_cat = ['chf', 'chdh','chda','afib_status']

    val_test_data.insert(0,'label',predicts[0])
    train_data.insert(0,'label',predicts[2])

    outcomes.drop('dataset', axis=1, inplace=True)

    add = pd.concat([outcomes,drop],axis=1)

    add_train = add.groupby('dataset').get_group(0)
    add_val = add.groupby('dataset').get_group(1)
    add_test = add.groupby('dataset').get_group(2)

    train_data = pd.concat([train_data,add_train],axis=1)

    if test==True:
        val_test_data = pd.concat([val_test_data,add_test],axis=1)
    else:
        val_test_data = pd.concat([val_test_data,add_val],axis=1)

    if test==False:
        name = f'output/validation_output_k{predicts[1]}.csv'
    else:
        name = f'output/test_output_k{predicts[1]}.csv'

    outs = list(outcomes.columns)
    outs.append('label')
    outs.append('gender1')

    train_data.to_csv(f'output/train_output_k{predicts[1]}.csv')
    train_data[outs].groupby('label').agg(['count', 'sum', 'mean']).T.to_csv(f'output/train_outcomes_k{predicts[1]}.csv')

    val_test_data.to_csv(name)
    val_test_data[outs].groupby('label').agg(['count', 'sum', 'mean']).T.to_csv(f'output/valtest_outcomes_k{predicts[1]}.csv')

    return train_data, val_test_data

def pca_analysis(data, n_components, n_most_important):
    '''
    Takes data, num principal components, num most important features interested in exploring.
    Output PCA loadings and top most important variables in each principal component.
    :param data: data df
    :param n_components: number of principal components
    :param n_most_important: number of most important features desired
    :return: None
    '''

    pca = PCA(n_components=n_components)
    pca.fit(data)
    pca_df = pd.DataFrame(pca.components_, columns=list(data.columns))

    imp_df = pd.DataFrame(pca_df.columns.values[np.argsort(-pca_df.values, axis=1)[:, :n_most_important]],
                          index=pca_df.index).reset_index()

    variance = list(pca.explained_variance_ratio_)
    imp_df.insert(1,'variance explained',variance)
    imp_df.rename(columns={'index':'principal component'},inplace=True)

    imp_df.to_csv('output/pca_most_important.csv',index=False)
    pca_df.to_csv('output/pca_loadings.csv')

def kproto(train, cat_pos, val_test, n_clusters, plot):
    '''
    Trains kprototypes classifier on training data, predicts on val/test data.
    :param train: training data with cat variable
    :param cat_pos: location of categorical variables as list
    :param val: validation or test data
    :param n_clusters: number clusters
    :param plot: if true will make plot in 2D (after PCA)
    :return: predict_val_test, n_clusters, predict_train
    '''

    kproto = KPrototypes(n_clusters=n_clusters, init='Cao').fit(train, categorical=cat_pos)

    predict_val_test = kproto.predict(val_test, categorical=cat_pos)
    predict_train = kproto.predict(train, categorical=cat_pos)

    train = train.iloc[:, :-6]
    val_test = val_test.iloc[:, :-6]

    if plot == True:
        pca = PCA(n_components=2)
        pca.fit(train)
        pca_data = pca.transform(val_test)

        pca1 = list()
        pca2 = list()

        for each in pca_data:
            pca1.append(each[0])
            pca2.append(each[1])

        pca1 = np.array(pca1)
        pca2 = np.array(pca2)

        val_test.insert(1, 'pca1', pca1)
        val_test.insert(2, 'pca2', pca2)

        fig = plt.figure(figsize=(8, 8))
        plt.scatter(
            x=pca1,
            y=pca2,
            c=predict_val_test,
            cmap='viridis')
        plt.xlabel('PCA1')
        plt.ylabel('PCA2')
        plt.title('K-Prototypes Clustering with Dimensionality Reduction for Visualization')
        plt.show()

    return (predict_val_test, n_clusters, predict_train)

def kproto_plots(train, cat_pos, range_):
    '''
    Cost plot.
    '''

    k_list, cost = [], []

    for n in range(2,range_):
        kproto = KPrototypes(n_clusters=n, init='Cao').fit(train, categorical=cat_pos)
        k_list.append(n)
        cost.append(kproto.cost_)

    plt.xlabel("Number of Clusters")
    plt.ylabel("Cost")
    plt.xlim(1, range_)
    plt.plot(k_list, cost, 'ro-', color='blue')
    plt.show()

def kproto_metrics_(train, cat_pos, k_range):
    '''
    Calculates Calinksi-Harabasz index (higher score means more dense and well-separated clusters)
    and Davies-Bouldin index (lower score means better separation between clusters).
    '''

    cal = []
    dav = []

    for n in np.arange(2, k_range+1):
        labels = KPrototypes(n_clusters=n, init='Cao').fit_predict(train, categorical=cat_pos)
        train_ = train.drop('gender1',axis=1)
        cal.append((n,calinski_harabasz_score(train_, labels)))
        dav.append((n,davies_bouldin_score(train_, labels)))

    print(f'Calinski-Harabasz Index: {cal}')
    print(f'Davies-Bouldin Scores: {dav}')

def kproto_pipeline():
    # file = 'clin_echo_outcomes.csv'
    file = 'full_data.csv'

    # Define outcome cols and cols to drop, remove those, then split train/val/test.
    outcomes = ['chf', 'chdh', 'chda', 'afib_status', 'total6', 'kccq126c']
    dropped = ['race','gender1']

    outcomes, drop, train, val, test = data_split(file=file,outcomes=outcomes,dropped=dropped)

    # Impute missing values as training set mean, scale data.
    imputer = SimpleImputer(strategy='mean').fit(train)
    scaler = MinMaxScaler().fit(train)

    train_p = data_prep(df=train, imputer=imputer, scaler=scaler)
    val_p = data_prep(df=val, imputer=imputer, scaler=scaler)
    test_p = data_prep(df=test, imputer=imputer, scaler=scaler)

    # Use PCA for dimensionality reduction.
    # Set n_components to int for number or decimal for fraction.
    pca = PCA(n_components=0.8)
    pca.fit(train_p)

    train_dr = dim_red(df=train_p, pca=pca)
    val_dr = dim_red(df=val_p, pca=pca)
    test_dr = dim_red(df=test_p, pca=pca)

    # Add back gender and race for K-prototypes
    gender_train = drop.groupby('dataset').get_group(0)['gender1']
    gender_val = drop.groupby('dataset').get_group(1)['gender1']
    gender_test = drop.groupby('dataset').get_group(2)['gender1']

    race_train = drop.groupby('dataset').get_group(0)['race']
    race_val = drop.groupby('dataset').get_group(1)['race']
    race_test = drop.groupby('dataset').get_group(2)['race']

    train_proto = train_dr.copy()
    val_proto = val_dr.copy()
    test_proto = test_dr.copy()

    train_proto.insert(0,'gender1',gender_train)
    val_proto.insert(0,'gender1',gender_val)
    test_proto.insert(0,'gender1',gender_test)

    train_proto.insert(1,'race',race_train)
    val_proto.insert(1,'race',race_val)
    test_proto.insert(1,'race',race_test)

    train_proto = pd.get_dummies(train_proto, columns = ['gender1','race'])
    val_proto = pd.get_dummies(val_proto, columns = ['gender1','race'])
    test_proto = pd.get_dummies(test_proto, columns = ['gender1','race'])

    # print(train_proto.columns)

    # Can change cluster number here
    predicts = kproto(train=train_proto, cat_pos=[26,27,28,29,30,31], val_test=val_proto, n_clusters=3, plot=False)
    # predicts = kproto(train=train_proto, cat_pos=[21,22,23,24,25,26], val_test=val_proto, n_clusters=6, plot=True)
    # predicts = kproto(train=train_proto, cat_pos=[21,22,23,24,25,26], val_test=val_proto, n_clusters=7, plot=True)

    # Output data
    train_data, val_test_data = output(predicts=predicts, train_data=train, val_test_data=val, outcomes=outcomes, drop=drop, test=False)

    # kproto_plots(train=train_proto, cat_pos=[21,22,23,24,25,26], range_=14)

    # kproto_metrics_(train=train_proto, cat_pos=[0], k_range=10)

    return None

def kmeans_pipeline():
    file = 'clin_echo_outcomes.csv'

    # Define outcome cols and cols to drop, remove those, then split train/val/test.
    outcomes = ['chf', 'chdh', 'chda', 'afib_status', 'total6', 'kccq126c']
    dropped = ['race','gender1']

    outcomes, drop, train, val, test = data_split(file=file,outcomes=outcomes,dropped=dropped)

    # train = train[list]
    # val = val[list]
    # test = test[list]

    # Impute missing values as training set mean, scale data.
    imputer = SimpleImputer(strategy='mean').fit(train)
    scaler = MinMaxScaler().fit(train)

    train_p = data_prep(df=train, imputer=imputer, scaler=scaler)
    val_p = data_prep(df=val, imputer=imputer, scaler=scaler)
    test_p = data_prep(df=test, imputer=imputer, scaler=scaler)

    # Use PCA for dimensionality reduction.
    # Set n_components to int for number or decimal for fraction.
    pca = PCA(n_components=0.8)
    pca.fit(train_p)

    train_dr = dim_red(df=train_p, pca=pca)
    val_dr = dim_red(df=val_p, pca=pca)
    test_dr = dim_red(df=test_p, pca=pca)

    # Kmeans predictions
    predicts = kmeans(train=train_dr,val_test=val_dr,n_clusters=4,plot=True)

    # Outputs
    train_data, val_test_data = output(predicts=predicts, train_data=train, val_test_data=val, outcomes=outcomes, drop=drop, test=False)

    # Inertia, silhouette plots from given data for given range
    # Silhouette visualizer for given data
    plots(data=train_dr,range_=20)
    sil_viz(data=train_dr)

    # PCA analysis
    pca_analysis(data=train_p,n_components=2,n_most_important=5)

    # Other functions
    metrics_(train_p,10)
    pca_variance(train_p)
    corr_matrix(threshold=0.9,data=val_p) # displays variable pairs with correlation > threshold

    return None

if __name__ == '__main__':

    # file = 'clin_echo_outcomes.csv'
    file = 'full_data.csv'

    # Define outcome cols and cols to drop, remove those, then split train/val/test.
    outcomes = ['chf', 'chdh', 'chda', 'afib_status', 'total6', 'kccq126c']
    dropped = ['race','gender1']

    outcomes, drop, train, val, test = data_split(file=file,outcomes=outcomes,dropped=dropped)

    # filter = ['gls','rvfw','ra_res_final', 'lu_la_strain_reservoir', 'la_reservoir_final']

    strain_list =  ['gls',
                    'rvfw',
                    'ra_res_final',
                    'la_reservoir_final',
                    'lu_la_strain_reservoir',
                    'lv2c_global_strain',
                    'lv3c_global_strain',
                    'a2c_la_strain_peak_pos',
                    'a2c_la_strain_peak_neg',
                    'a2c_la_reservoir_strain',
                    'ap_rv_strain',
                    'mid_rv_strain',
                    'bas_rv_strain',
                    'ra_strain_peak_pos',
                    'ra_strain_peak_neg',
                    'lu_lv4c_global_strain',
                    'lv_circ_strain',
                    'a2c_la_reservoir_strain_d',
                    'ra_reservoir_strain_d',
                    'lv4c_global_strain',
                    'pos_a4c_la_strain',
                    'neg_a4c_la_strain',
                    'reservoir_a4c_la_strain',
                    'lu_la_strain_peak_pos',
                    'lu_la_strain_peak_neg'
                    ]

    train = train[strain_list]
    val = val[strain_list]
    test = test[strain_list]

    # Impute missing values as training set mean, scale data.
    imputer = SimpleImputer(strategy='mean').fit(train)
    scaler = MinMaxScaler().fit(train)

    train_p = data_prep(df=train, imputer=imputer, scaler=scaler)
    val_p = data_prep(df=val, imputer=imputer, scaler=scaler)
    test_p = data_prep(df=test, imputer=imputer, scaler=scaler)

    # Use PCA for dimensionality reduction.
    # Set n_components to int for number or decimal for fraction.
    # pca = PCA(n_components=0.8)
    # pca.fit(train_p)
    #
    # train_dr = dim_red(df=train_p, pca=pca)
    # val_dr = dim_red(df=val_p, pca=pca)
    # test_dr = dim_red(df=test_p, pca=pca)

    # Kmeans predictions
    predicts = kmeans(train=train_p,val_test=val_p,n_clusters=3,plot=True)

    # Outputs
    train_data, val_test_data = output(predicts=predicts, train_data=train, val_test_data=val, outcomes=outcomes, drop=drop, test=False)

    # Inertia, silhouette plots from given data for given range
    # Silhouette visualizer for given data
    plots(data=train_p,range_=20)
    sil_viz(data=train_p)

    # PCA analysis
    # pca_analysis(data=train_p,n_components=2,n_most_important=5)

    # Other functions
    # metrics_(train_p,10)
    pca_variance(train_p)
    corr_matrix(threshold=0.9,data=val_p) # displays variable pairs with correlation > threshold
