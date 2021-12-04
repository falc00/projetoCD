from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig, show
from libs.ds_charts import bar_chart

register_matplotlib_converters()

filename = 'dataset_1/NYC_collisions_tabular.csv'
data = read_csv(filename, na_values='na')

print(data.shape)

figure(figsize=(4,2))
values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
bar_chart(list(values.keys()), list(values.values()), title='Nr of records vs nr variables')
savefig('images/records_variables.png')
show()