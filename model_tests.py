import numpy as np
from pandas import read_csv, concat, unique, DataFrame
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import libs.ds_charts as ds
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from matplotlib.pyplot import figure, savefig, show
from libs.ds_charts import plot_evaluation_results, bar_chart
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.neighbors import KNeighborsClassifier

def train_test_split_d2(positive,negative,target,file_name,file_path):
    data: DataFrame = read_csv((file_path+'.csv'))
    values = {'Original': [len(data[data[target] == positive]), len(data[data[target] == negative])]}
    y: np.ndarray = data.pop(target).values
    X: np.ndarray = data.values
    labels: np.ndarray = unique(y)
    labels.sort()

    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

    train = concat([DataFrame(trnX, columns=data.columns), DataFrame(trnY,columns=[target])], axis=1)
    train.to_csv(file_path+'_train.csv', index=False)

    test = concat([DataFrame(tstX, columns=data.columns), DataFrame(tstY,columns=[target])], axis=1)
    test.to_csv(file_path+'_test.csv', index=False)
    values['Train'] = [len(np.delete(trnY, np.argwhere(trnY==negative))), len(np.delete(trnY, np.argwhere(trnY==positive)))]
    values['Test'] = [len(np.delete(tstY, np.argwhere(tstY==negative))), len(np.delete(tstY, np.argwhere(tstY==positive)))]

    plt.figure(figsize=(12,4))
    ds.multiple_bar_chart([positive, negative], values, title='Data distribution for '+file_name)
    plt.show()


def naive_bayes(train_path,test_path,target,pos_label):
    train: DataFrame = read_csv(train_path)
    test: DataFrame = read_csv(test_path)

    trnY: np.ndarray = train.pop(target).values
    trnX: np.ndarray = train.values
    labels = unique(trnY)
    labels.sort()

    tstY: np.ndarray = test.pop(target).values
    tstX: np.ndarray = test.values

    clf = MultinomialNB()
    clf.fit(trnX, trnY)
    prd_trn = clf.predict(trnX)
    prd_tst = clf.predict(tstX)
    
    print("F1: " + str(f1_score(y_true=tstY, y_pred=prd_tst, pos_label=pos_label)))
    print("Recall: " + str(recall_score(tstY, prd_tst, pos_label=pos_label)))
    print("Precision: " + str(precision_score(tstY, prd_tst, pos_label=pos_label)))

    #plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
    #savefig(f'lab4_images/dataset_1/{file_tag_zscore}_nb_best_dummified_finalversion.png')
    #show()
    
def knn(train_path,test_path,target,pos_label,n,metric):
    train: DataFrame = read_csv(train_path)
    test: DataFrame = read_csv(test_path)
    
    trnY: np.ndarray = train.pop(target).values
    trnX: np.ndarray = train.values
    labels = unique(trnY)
    labels.sort()

    tstY: np.ndarray = test.pop(target).values
    tstX: np.ndarray = test.values

    clf = knn = KNeighborsClassifier(n_neighbors=n, metric=metric)
    clf.fit(trnX, trnY)
    prd_trn = clf.predict(trnX)
    prd_tst = clf.predict(tstX)

    print("F1: " + str(f1_score(tstY, prd_tst, pos_label=pos_label)))
    print("Recall: " + str(recall_score(tstY, prd_tst, pos_label=pos_label)))
    print("Precision: " + str(precision_score(tstY, prd_tst, pos_label=pos_label)))