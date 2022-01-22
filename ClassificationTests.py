from numpy import ndarray
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, savefig, show
from sklearn.neighbors import KNeighborsClassifier
from libs.ds_charts import plot_evaluation_results, multiple_line_chart
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

target = 'PERSON_INJURY'
filename = 'nyc_collisions_scaled_zscore'
train: DataFrame = read_csv(f'data/{filename}_train.csv')
trnY: ndarray = train.pop(target).values
trnX: ndarray = train.values
labels = unique(trnY)
labels.sort()

test: DataFrame = read_csv(f'data/{filename}_test.csv')
tstY: ndarray = test.pop(target).values
tstX: ndarray = test.values
