from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, title, savefig, show
import model_tests as mt
from seaborn import heatmap
from matplotlib.pyplot import figure, savefig, show
from libs.ds_charts import bar_chart, get_variable_types, multiple_line_chart
import numpy as np
from pandas import read_csv, concat, unique, DataFrame
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from pandas import Series
from imblearn.over_sampling import SMOTE

filename = 'data/air_quality_dummified.csv'
data = read_csv(filename, na_values='?')
data.shape


THRESHOLD = 0.9

print(data)
print("Dataframe has " + str(len(data.columns)))

def select_redundant(corr_mtx, threshold: float) -> tuple[dict, DataFrame]:
    if corr_mtx.empty:
        return {}

    corr_mtx = abs(corr_mtx)
    vars_2drop = {}
    for el in corr_mtx.columns:
        el_corr = (corr_mtx[el]).loc[corr_mtx[el] >= threshold]
        if len(el_corr) == 1:
            corr_mtx.drop(labels=el, axis=1, inplace=True)
            corr_mtx.drop(labels=el, axis=0, inplace=True)
        else:
            vars_2drop[el] = el_corr.index
    return vars_2drop, corr_mtx

drop1, corr_mtx1 = select_redundant(data.corr(), 0.9)
print(len(drop1))
drop2, corr_mtx2 = select_redundant(data.corr(), 0.7)
print(len(drop2))

drop3, corr_mtx3 = select_redundant(data.corr(), 0.5)

if corr_mtx1.empty:
    raise ValueError('Matrix is empty.')

figure(figsize=[10, 10])
heatmap(corr_mtx1, xticklabels=corr_mtx1.columns, yticklabels=corr_mtx1.columns, annot=False, cmap='Blues')
title('Filtered Correlation Analysis')
savefig(f'lab6_images/dataset_2/filtered_correlation_analysis_d2_{THRESHOLD}.png')
#show()

def drop_redundant(data: DataFrame, vars_2drop: dict) -> DataFrame:
    sel_2drop = []
    print(vars_2drop.keys())
    for key in vars_2drop.keys():
        if key not in sel_2drop:
            for r in vars_2drop[key]:
                if r != key and r not in sel_2drop:
                    sel_2drop.append(r)
    print('Variables to drop', sel_2drop)
    df = data.copy()
    for var in sel_2drop:
        df.drop(labels=var, axis=1, inplace=True)
    return df
df1 = drop_redundant(data, drop1)
df2 = drop_redundant(data, drop2)
df3 = drop_redundant(data, drop3)

def select_low_variance(data: DataFrame, threshold: float) -> list:
    lst_variables = []
    lst_variances = []
    for el in data.columns:
        value = data[el].var()
        if value >= threshold:
            lst_variables.append(el)
            lst_variances.append(value)

    print(len(lst_variables), lst_variables)
    figure(figsize=[10, 4])
    #bar_chart(lst_variables, lst_variances, title='Variance analysis', xlabel='variables', ylabel='variance')
    #savefig('lab6_images/dataset_2/filtered_variance_analysis_ds2.png')
    return lst_variables

numeric = get_variable_types(data)['Numeric']
vars_2drop = select_low_variance(data[numeric], 0.05)
for var in vars_2drop:
    if var in list(df1.columns):
        df1.drop(var,inplace=True,axis=1)
    if var in list(df2.columns):
        df2.drop(var,inplace=True,axis=1)
    if var in list(df3.columns):
        df3.drop(var,inplace=True,axis=1)
print(vars_2drop)
def split(df):
    target = 'ALARM'
    positive = 'Danger'
    negative = 'Safe'
    values = {'Original': [len(df[df[target] == positive]), len(df[df[target] == negative])]}

    y: np.ndarray = df.pop(target).values
    X: np.ndarray = df.values
    labels: np.ndarray = unique(y)
    labels.sort()

    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

    return concat([DataFrame(trnX, columns=df.columns), DataFrame(trnY,columns=[target])], axis=1) , concat([DataFrame(tstX, columns=df.columns), DataFrame(tstY,columns=[target])], axis=1)

# split train and test
trn0,tst0 = split(data.copy())
trn1,tst1 = split(df1.copy())
trn2,tst2 = split(df2.copy())
trn3,tst3 = split(df3.copy())

def nb(train_new,test_new):
    target = 'ALARM'
    trnY_new: np.ndarray = train_new.pop(target).values
    trnX_new: np.ndarray = train_new.values
    labels = unique(trnY_new)
    labels.sort()

    tstY_new: np.ndarray = test_new.pop(target).values
    tstX_new: np.ndarray = test_new.values

    clf = BernoulliNB()
    clf.fit(trnX_new, trnY_new)
    prd_trn = clf.predict(trnX_new)
    prd_tst = clf.predict(tstX_new)
    return [f1_score(tstY_new, prd_tst, pos_label='Danger'), recall_score(tstY_new, prd_tst, pos_label='Danger'), precision_score(tstY_new, prd_tst, pos_label='Danger'), accuracy_score(tstY_new, prd_tst)]

# Naive Bayes!!!
trn0
a = nb(trn0.copy(), tst0.copy())
b = nb(trn1.copy(), tst1.copy())
c = nb(trn2.copy(), tst2.copy())
d = nb(trn3.copy(), tst3.copy())

multiple_line_chart([0.5,0.7,0.9,1],{"F1":[b[0],c[0],d[0],a[0]],"Recall":[b[1],c[1],d[1],a[1]],"Precision":[b[2],c[2],d[2],a[2]],"Accuracy":[b[3],c[3],d[3],a[3]]},title="Feature Impact (Naive Bayes)",xlabel="Correlation threshold (1 is the original)",ylabel="Score")
show()
exit(1)

# KNN impact 

from sklearn.neighbors import KNeighborsClassifier
#BEFORE
clf = knn = KNeighborsClassifier(n_neighbors=1, metric='manhattan')
clf.fit(trnX_old, trnY_old)
prd_trn = clf.predict(trnX_old)
prd_tst = clf.predict(tstX_old)
knn_old = f1_score(tstY_old, prd_tst, pos_label='Danger')

#AFTER
clf = knn = KNeighborsClassifier(n_neighbors=1, metric='manhattan')
clf.fit(trnX_new, trnY_new)
prd_trn = clf.predict(trnX_new)
prd_tst = clf.predict(tstX_new)
knn_new = f1_score(tstY_new, prd_tst, pos_label='Danger')

#bar_chart(xvalues=['Before','After'],yvalues=[knn_old,knn_new],title="Feature selection KNN")
#savefig('lab6_images/dataset_2/knn_before_after_ds2.png')

objects = ('Before', 'After')
y_pos = np.arange(len(objects))
performance = [knn_old, knn_new]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('F1_Score')
plt.title('Feature selection KNN')
#plt.show()
#plt.savefig('lab6_images/dataset_2/knn_before_after_ds2.png')
#show()

# Random Forests Impact 

#BEFORE
clf = RandomForestClassifier(n_estimators=125,max_depth=5,max_features=0.9)
clf.fit(trnX_old, trnY_old)
prd_trn = clf.predict(trnX_old)
prd_tst = clf.predict(tstX_old)
rf_old = f1_score(tstY_old, prd_tst, pos_label='Danger')

#AFTER
clf = RandomForestClassifier(n_estimators=125,max_depth=5,max_features=0.9)
clf.fit(trnX_new, trnY_new)
prd_trn = clf.predict(trnX_new)
prd_tst = clf.predict(tstX_new)
rf_new = f1_score(tstY_new, prd_tst, pos_label='Danger')

objects = ('Before', 'After')
y_pos = np.arange(len(objects))
performance = [rf_old, rf_new]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('F1_Score')
plt.title('Feature selection RF')
plt.show()
plt.savefig('lab6_images/dataset_2/knn_before_after_ds2.png')
#bar_chart(xvalues=['Before','After'],yvalues=[rf_old,rf_new],title="Feature selection RF")
#savefig('lab6_images/dataset_2/rf_before_after_ds2.png')
#show()









