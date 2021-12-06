from matplotlib.axes import Axes
from numpy import log
from pandas import read_csv, Series
from pandas import DataFrame
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig, show, subplots, hist
from scipy.stats import norm, expon, lognorm
from seaborn import distplot
from libs.ds_charts import bar_chart, get_variable_types, choose_grid, HEIGHT, multiple_bar_chart, multiple_line_chart
import re
from datetime import datetime

register_matplotlib_converters()

filename = 'dataset_1/NYC_collisions_tabular.csv'
data = read_csv(filename, na_values='na')

### DISTRIBUTION

# Fast description
summary5 = data.describe()
print(summary5)
"""
#Individual boxplot for age
data.boxplot(column=['PERSON_AGE'])
savefig('images/age_boxplot_dataset1.png')

#Number of outliers for age
NR_STDEV: int = 2
numeric_vars = ["PERSON_AGE"]
if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')

outliers_iqr = []
outliers_stdev = []
summary5 = data.describe(include='number')

for var in numeric_vars:
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
multiple_bar_chart(numeric_vars, outliers, title='Nr of outliers per variable', xlabel='variables', ylabel='nr outliers', percentage=False)
savefig('images/outliers_dataset1.png')


#Number of records per value
hist(data['PERSON_AGE'].dropna().values, range=(0,110))
savefig('images/single_histograms_numeric_dataset1.png')
"""
#Distribution of records per value
ages_list = data['PERSON_AGE'].dropna().values
ages_list_filter1 = ages_list[ages_list<110]
ages_list_filter2 = ages_list_filter1[ages_list_filter1>0]
distplot(ages_list_filter2, norm_hist=True,axlabel='PERSON_AGE')
savefig('images/histograms_trend_numeric_dataset1.png')
show()

def compute_known_distributions(x_values: list) -> dict:
    distributions = dict()
    # Gaussian
    mean, sigma = norm.fit(x_values)
    distributions['Normal(%.1f,%.2f)'%(mean,sigma)] = norm.pdf(x_values, mean, sigma)
    # Exponential
    loc, scale = expon.fit(x_values)
    distributions['Exp(%.2f)'%(1/scale)] = expon.pdf(x_values, loc, scale)
    # LogNorm
    sigma, loc, scale = lognorm.fit(x_values)
    distributions['LogNor(%.1f,%.2f)'%(log(scale),sigma)] = lognorm.pdf(x_values, sigma, loc, scale)
    return distributions

def histogram_with_distributions(ax: Axes, series: Series, var: str):
    values = series.sort_values().values
    ax.hist(values, 20, density=True)
    distributions = compute_known_distributions(values)
    multiple_line_chart(values, distributions, ax=ax, title='Best fit for %s'%var, xlabel=var, ylabel='')

""""

### DIMENSIONALITY

print(data.shape)
figure(figsize=(4,2))
values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
bar_chart(list(values.keys()), list(values.values()), title='Nr of records vs nr variables')
savefig('images/records_variables_dataset1.png')

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
figure(figsize=(6,3))
bar_chart(list(counts.keys()), list(counts.values()), title='Nr of variables per type')
savefig('images/variable_types_dataset1.png')

mv = {}
for var in data:
    nr = data[var].isna().sum()
    if nr > 0:
        mv[var] = nr

figure(figsize=(6,3))

bar_chart(list(mv.keys()), list(mv.values()), title='Nr of missing values per variable',
            xlabel='variables', ylabel='nr missing values', rotation=True)
savefig('images/mv_dataset1.png',bbox_inches="tight")
show()
"""
