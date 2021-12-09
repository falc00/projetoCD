from pandas import read_csv
from pandas import DataFrame
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig, show, bar, subplots, Axes
from libs.ds_charts import bar_chart, choose_grid, HEIGHT, multiple_bar_chart, get_variable_types, multiple_line_chart
from seaborn import distplot
from numpy import log
from pandas import Series
from scipy.stats import norm, expon, lognorm
from matplotlib.pyplot import figure, savefig, show, title
from seaborn import heatmap


import re
from datetime import datetime
'''
Data Dimensionality
'''
register_matplotlib_converters()

filename = 'dataset_2/air_quality_tabular.csv'
data = read_csv(filename, na_values='na')

print(data.shape)
figure(figsize=(20,10))
values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
bar_chart(list(values.keys()), list(values.values()), title='Nr of records vs nr variables', percentage=True)
savefig('images/1.records_variables_dataset2.png')
#show()

cat_vars = data.select_dtypes(include='object')
data[cat_vars.columns] = data.select_dtypes(['object']).apply(lambda x: x.astype('category'))
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
savefig('images/2.variable_types_dataset2.png')

mv = {}
for var in data:
    nr = data[var].isna().sum()
    if nr > 0:
        mv[var] = nr

figure()
bar_chart(list(mv.keys()), list(mv.values()), title='Nr of missing values per variable',
            xlabel='variables', ylabel='nr missing values', rotation=True)
savefig('images/3.mv_dataset2.png')
#show()
######
'''
Data Distribuition --------------------------------------------------------------------------------------------------
'''
'''
summary5 = data.describe()
print(summary5)
print("ola")
data.boxplot()
savefig('images/global_boxplot_dataset2.png')
show()
'''
# box plots for each numeric variable
summary5 = data.describe()
print(summary5)
columns_numeric=['CO_Mean','CO_Min','CO_Max','CO_Std','NO2_Mean', 'NO2_Min', 'NO2_Max','NO2_Std', 'O3_Mean','O3_Min'
                              ,'O3_Max', 'O3_Std', 'PM2.5_Mean', 'PM2.5_Min', 'PM2.5_Max', 'PM2.5_Std', 'PM10_Mean',
                              'PM10_Min', 'PM10_Max', 'PM10_Std', 'SO2_Mean', 'SO2_Min', 'SO2_Max', 'SO2_Std']
rows, cols = choose_grid(len(columns_numeric))
fig, axs = subplots(rows,cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
i, j = 0, 0
for n in range(len(columns_numeric)):
    axs[i, j].set_title('Boxplot for %s'%columns_numeric[n])
    axs[i, j].boxplot(data[columns_numeric[n]].dropna().values)
    i, j = (i +1, 0) if (n+1) % cols == 0 else (i, j+1)
savefig('images/4.global_boxpot_dataset2.png')

# Outliers for each numeric variable
NR_STDEV: int = 2
outliers_iqr = []
outliers_stdev = []
summary5 = data.describe(include='number')

for var in columns_numeric:
    iqr = 1.5 * (summary5[var]['75%'] - summary5[var]['25%'])
    outliers_iqr += [
        data[data[var] > summary5[var]['75%']  + iqr].count()[var] +
        data[data[var] < summary5[var]['25%']  - iqr].count()[var]]
    std = NR_STDEV * summary5[var]['std']
    outliers_stdev += [
        data[data[var] > summary5[var]['mean'] + std].count()[var] +
        data[data[var] < summary5[var]['mean'] - std].count()[var]]

outliers = {'iqr': outliers_iqr, 'stdev': outliers_stdev}
figure(figsize=(12, HEIGHT))
multiple_bar_chart(columns_numeric, outliers, title='Nr of outliers per variable', xlabel='variables', ylabel='nr outliers', percentage=False)
savefig('images/5.outliers_per_numeric_variable_dataset2.png')
#show()

# Histogram for each numeric value
fig, axs =subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze = False)
i, j = 0, 0
for n in range(len(columns_numeric)):
    axs[i, j].set_title('Histogram for %s'%columns_numeric[n])
    axs[i, j].set_xlabel(columns_numeric[n])
    axs[i, j].set_ylabel('nr records')
    axs[i, j].hist(data[columns_numeric[n]].dropna().values, 'auto')
    i, j = ( i + 1, 0) if (n+1) % cols == 0 else (i, j+1)

savefig('images/6.single_histograms_numeric_dataset2.png')
#show()

# Display the best fit for each variable
fig, axs =subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze = False)
i, j = 0, 0
for n in range(len(columns_numeric)):
    axs[i, j].set_title('Histogram with trend for %s'%columns_numeric[n])
    distplot(data[columns_numeric[n]].dropna().values, norm_hist = True, ax=axs[i,j], axlabel= columns_numeric[n])
    i, j = (i +1, 0) if (n+1) %cols == 0 else (i, j+1)
savefig('images/6.histograms_trend_numeric_dataset2.png')
#show()
'''
# Now compute distribuitions (norm, expon, skewnorm, etc)
def compute_known_distributions(x_values: list) -> dict:
    distributions = dict()
    # Gaussian
    mean, sigma = norm.fit(x_values)
    distributions['Normal(%.1f,%.2f)'%(mean,sigma)] = norm.pdf(x_values, mean, sigma)
    # Exponential
    #loc, scale = expon.fit(x_values)
    #distributions['Exp(%.2f)'%(1/scale)] = expon.pdf(x_values, loc, scale)
    # LogNorm
    #
    #sigma, loc, scale = lognorm.fit(x_values)
    #distributions['LogNor(%.1f,%.2f)'%(log(scale),sigma)] = lognorm.pdf(x_values, sigma, loc, scale)
    return distributions

def histogram_with_distributions(ax: Axes, series: Series, var: str):
    values = series.sort_values().values
    ax.hist(values, 20, density=True)
    distributions = compute_known_distributions(values)
    multiple_line_chart(values, distributions, ax=ax, title='Best fit for %s'%var, xlabel=var, ylabel='')

fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
i, j = 0, 0
for n in range(len(columns_numeric)):
    histogram_with_distributions(axs[i, j], data[columns_numeric[n]].dropna(), columns_numeric[n])
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
savefig('images/7.histogram_numeric_distribution_dataset2.png')
#show()
'''
'''
Histogram for Symbolic variables
'Symbolic': ['date', 'City_EN', 'Prov_EN', 'GbCity']}
'''
symbolic_vars = ['City_EN', 'Prov_EN'] # GB City numeric!(id); date?


for n in range(len(symbolic_vars)):
    rows, cols = choose_grid(1)
    fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)
    i, j = 0, 0
    counts = data[symbolic_vars[n]].value_counts()
    bar_chart(counts.index.to_list(), counts.values, ax=axs[i, j], title='Histogram for %s'%symbolic_vars[n], xlabel=symbolic_vars[n], ylabel='nr records', percentage=False,
              rotation = True)
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    string = 'images/8.histograms_symbolic_dataset2' + str(n) + '.png'
    savefig(string)
#show()

'''
Sparsity DataSet 2 -----------------------------------------------------------------------------------------------------
'''
# Numeric Variables
numeric_vars = get_variable_types(data)['Numeric']
if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')

rows, cols = len(numeric_vars)-1, len(numeric_vars)-1
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
for i in range(len(numeric_vars)):
    var1 = numeric_vars[i]
    for j in range(i+1, len(numeric_vars)):
        var2 = numeric_vars[j]
        axs[i, j-1].set_title("%s x %s"%(var1,var2))
        axs[i, j-1].set_xlabel(var1)
        axs[i, j-1].set_ylabel(var2)
        axs[i, j-1].scatter(data[var1], data[var2])
savefig(f'images/9.sparsity_study_numeric_dataset2.png')
#show()

#Symbolic variables
symbolic_vars = ['City_EN', 'Prov_EN']
if [] == symbolic_vars:
    raise ValueError('There are no symbolic variables.')

rows, cols = len(symbolic_vars)-1, len(symbolic_vars)-1
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
for i in range(len(symbolic_vars)):
    var1 = symbolic_vars[i]
    for j in range(i+1, len(symbolic_vars)):
        var2 = symbolic_vars[j]
        axs[i, j-1].set_title("%s x %s"%(var1,var2))
        axs[i, j-1].set_xlabel(var1)
        axs[i, j-1].set_ylabel(var2)
        axs[i, j-1].scatter(data[var1], data[var2])
savefig(f'images/10. sparsity_study_symbolic_dataset2.png')
#show()

corr_mtx = abs(data.corr())
print(corr_mtx)

fig = figure(figsize=[12, 12])

heatmap(abs(corr_mtx), xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
title('Correlation analysis')
savefig(f'images/correlation_analysis.png')
#show()
