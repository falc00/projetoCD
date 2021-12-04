from pandas import read_csv
from pandas import DataFrame
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig, show
from libs.ds_charts import bar_chart
import re
from datetime import datetime

register_matplotlib_converters()

filename = 'dataset_1/NYC_collisions_tabular.csv'
data = read_csv(filename, na_values='na')

print(data.shape)
figure(figsize=(4,2))
values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
bar_chart(list(values.keys()), list(values.values()), title='Nr of records vs nr variables')
savefig('images/records_variables.png')
#show()

cat_vars = data.select_dtypes(include='object')
data[cat_vars.columns] = data.select_dtypes(['object']).apply(lambda x: x.astype('category'))
print(data.dtypes)
print("CRASH TIME: ", type(data['CRASH_TIME'][0]))
def get_variable_types(df: DataFrame) -> dict:
    variable_types: dict = {
        'Numeric': [],
        'Binary': [],
        'Date': [],
        'Symbolic': []
    }
    for c in df.columns:
        uniques = df[c].dropna(inplace=False).unique()
        if len(uniques) == 2:
            variable_types['Binary'].append(c)
            df[c].astype('bool')
        elif df[c].dtype == 'datetime64':
            variable_types['Date'].append(c)
        ##QUESTION: WHAT IS DATE TYPE ATTRIBUTE
        elif c == "CRASH_DATE":
            variable_types['Date'].append(c)
        elif c == "CRASH_TIME":
            variable_types['Date'].append(c)
        elif df[c].dtype == 'int':
            variable_types['Numeric'].append(c)
        elif df[c].dtype == 'float':
            variable_types['Numeric'].append(c)
        else:
            df[c].astype('category')
            variable_types['Symbolic'].append(c)

    return variable_types

variable_types = get_variable_types(data)
print(variable_types)
print()
counts = {}
for tp in variable_types.keys():
    counts[tp] = len(variable_types[tp])
figure(figsize=(4,2))
bar_chart(list(counts.keys()), list(counts.values()), title='Nr of variables per type')
savefig('images/variable_types.png')

mv = {}
for var in data:
    nr = data[var].isna().sum()
    if nr > 0:
        mv[var] = nr

figure()
bar_chart(list(mv.keys()), list(mv.values()), title='Nr of missing values per variable',
            xlabel='variables', ylabel='nr missing values', rotation=True)
savefig('images/mv.png')

summary5 = data.describe()
print(summary5)

data.boxplot()
savefig('images/global_boxplot.png')
show()