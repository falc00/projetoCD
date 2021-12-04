from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart

register_matplotlib_converters()

filename = 'dataset_1/NYC_collisions_tabular.csv'
data = read_csv(filename, na_values='na')

print(data.shape)