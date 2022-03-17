import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

def data_prep():
    '''
    Takes 5 datasets from multiple imputations.
    Creates 2 dictionaries with data. Keys are mimp1 - mimp5, value is corresponding df.
    Train_dict, test_dict.
    Creates 1 dictionary with test set ids and empty list to later populate k values from each of 5 mimp sets.
    '''

    file = 'multimp/mimp.xlsx'
    file2 = 'echo.csv'

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

    # prep all dfs in train/test
    for df in data:
        data[df].drop(['dataset', 'gender1', 'tr_peak', 'echopasp'], inplace=True, axis=1)
        ids = data[df].pop('idno')
        scaled_data = MinMaxScaler().fit_transform(data[df])
        clean_df = pd.DataFrame(scaled_data, columns=orig.columns)
        clean_df.insert(0, 'idno', ids)  #
        clean_df.set_index('idno', inplace=True)
        train = clean_df.sample(frac=0.8, random_state=2)
        test = clean_df.drop(train.index)
        train_dict[df] = train
        test_dict[df] = test

    for i,row in test_dict['mimp1'].iterrows(): # filling the dict with test indices and empty list
        test_labels[i] = list()

    return train_dict, test_dict, test_labels

def kmeans(train_dict, test_dict, test_labels, n_clusters):
    '''
    For each of 5 datasets:
        Train on 80%.
        Make prediction on remaining 20%.
        Store prediction in dictionary with idno as key, values as list with all 5 k values.
        Will pick max k value as the predicted k value for given test id.
    '''

    for df in train_dict:
        kmeans = KMeans(n_clusters=n_clusters, init = 'random', random_state=2, n_init = 10, max_iter=300).fit(train_dict[df])
        predict_test = kmeans.predict(test_dict[df])
        test_dict[df].insert(0, 'label', predict_test)

        for i,row in test_dict[df].iterrows():
            test_labels[i].append(row['label'])

    return test_dict, test_labels

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
            count = count + 1
    print(count)

if __name__ == '__main__':

    train, test, test_labels = data_prep()
    test_dict, test_labels = kmeans(train,test,test_labels,4)

    mimp_count(test_labels)