import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from numpy import ndarray
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import libs.ds_charts as ds
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame, concat
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame, concat


dataset_2 = pd.read_csv('data/report/air_quality_tabular.csv', na_values='NaN')
new_dataset_2 = dataset_2.copy()

def missing_values(new_dataset_2):
    #FIND VARIABLES WITH MISSING VALUES
    mv = {}
    for var in new_dataset_2:
        nr = new_dataset_2[var].isna().sum()
        if nr > 0:
            mv[var] = nr
        
    #DISCARD COLUMNS WITH MORE THEN 90% MISSING VALUES
    threshold = new_dataset_2.shape[0] * 0.80

    missings = [c for c in mv.keys() if mv[c]>threshold]
    new_dataset_2.drop(columns=missings, inplace=True)
    print('Dropped variables', missings)

    #DISCARD RECORDS WITH MAJORITY OF MISSING VALUES
    threshold = new_dataset_2.shape[1] * 0.50

    new_dataset_2.dropna(thresh=threshold, inplace=True)
    print(new_dataset_2.shape)
    mv = {}
    for var in new_dataset_2:
        nr = new_dataset_2[var].isna().sum()
        if nr > 0:
            mv[var] = nr
    print(mv)
    # numeric values
    for column in mv:
        if column != "Field_1":
            vars = new_dataset_2[column]
            mean_vars = int(vars.mean())
            new_dataset_2[column].fillna(mean_vars,inplace=True)
    return new_dataset_2

def dummification_v1(new_dataset_2):
    new_dataset_2.drop('Field_1',axis=1,inplace=True)
    new_dataset_2.drop('FID', axis =1,inplace=True)
    new_dataset_2.drop('GbProv', axis =1,inplace=True)
    new_dataset_2.drop('GbCity', axis =1,inplace=True)
    ## Only cities were dummified because of our granularity study.
    new_dataset_2.drop('City_EN', axis =1,inplace=True)

    file = 'air_quality'
    filename = 'data/report/air_quality_tabular.csv'
    symbolic_vars = ['Prov_EN','date']

    def quarter_month(day):
        if day <= 8:
            return 1
        if day > 8 and day < 16:
            return 2
        if day >= 16 and day < 23:
            return 3
        if day >= 23:
            return 4
        else:
            return 0

    def dummify(df, vars_to_dummify):
        other_vars = [c for c in df.columns if not c in vars_to_dummify]
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=bool)
        X = df[vars_to_dummify]
        encoder.fit(X)
        new_vars = encoder.get_feature_names(vars_to_dummify)
        trans_X = encoder.transform(X)
        dummy = pd.DataFrame(trans_X, columns=new_vars, index=X.index)
        dummy = dummy.convert_dtypes(convert_boolean=True)

        final_df = pd.concat([df[other_vars], dummy], axis=1)
        return final_df

    ## Transform dates to quarter of month (1,2,3,4)
    new_dataset_2['date'] = new_dataset_2['date'].apply(lambda x: quarter_month(int(x.split('/')[0])))

    variables = ds.get_variable_types(new_dataset_2)
    new_dataset_2 = dummify(new_dataset_2, symbolic_vars)

    new_dataset_2.to_csv(f'data/report/dummified/air_quality_dummified_v1.csv', index=False)


def dummification_v2(new_dataset_2):
    new_dataset_2.drop('Field_1',axis=1,inplace=True)
    new_dataset_2.drop('FID', axis =1,inplace=True)
    new_dataset_2.drop('GbProv', axis =1,inplace=True)
    new_dataset_2.drop('GbCity', axis =1,inplace=True)
    ## Only cities were dummified because of our granularity study.
    new_dataset_2.drop('City_EN', axis =1,inplace=True)

    file = 'air_quality'
    filename = 'data/report/air_quality_tabular.csv'
    symbolic_vars = ['Prov_EN','date']


    def dummify(df, vars_to_dummify):
        other_vars = [c for c in df.columns if not c in vars_to_dummify]
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=bool)
        X = df[vars_to_dummify]
        encoder.fit(X)
        new_vars = encoder.get_feature_names(vars_to_dummify)
        trans_X = encoder.transform(X)
        dummy = pd.DataFrame(trans_X, columns=new_vars, index=X.index)
        dummy = dummy.convert_dtypes(convert_boolean=True)

        final_df = pd.concat([df[other_vars], dummy], axis=1)
        return final_df

    ## Transform dates to 1,2,3,4,5,6,7,8,9,10,11 (months)
    new_dataset_2['date'] = new_dataset_2['date'].apply(lambda x: int(x.split('/')[1]))

    variables = ds.get_variable_types(new_dataset_2)
    new_dataset_2 = dummify(new_dataset_2, symbolic_vars)

    new_dataset_2.to_csv(f'data/report/dummified/air_quality_dummified_v2.csv', index=False)

def dummification_v3(new_dataset_2):
    new_dataset_2.drop('Field_1',axis=1,inplace=True)
    new_dataset_2.drop('FID', axis =1,inplace=True)
    new_dataset_2.drop('GbProv', axis =1,inplace=True)
    #new_dataset_2.drop('GbCity', axis =1,inplace=True)
    new_dataset_2.drop('City_EN', axis =1,inplace=True)
    new_dataset_2.drop('Prov_EN', axis =1,inplace=True)


    new_dataset_2 = new_dataset_2[new_dataset_2['GbCity'] != 's']
    new_dataset_2['GbCity'] = new_dataset_2['GbCity'].apply(lambda x: int(x))

    file = 'air_quality'
    filename = 'data/report/air_quality_tabular.csv'
    symbolic_vars = ['date']


    def dummify(df, vars_to_dummify):
        other_vars = [c for c in df.columns if not c in vars_to_dummify]
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=bool)
        X = df[vars_to_dummify]
        encoder.fit(X)
        new_vars = encoder.get_feature_names(vars_to_dummify)
        trans_X = encoder.transform(X)
        dummy = pd.DataFrame(trans_X, columns=new_vars, index=X.index)
        dummy = dummy.convert_dtypes(convert_boolean=True)

        final_df = pd.concat([df[other_vars], dummy], axis=1)
        return final_df

    ## Transform dates to 1,2,3,4,5,6,7,8,9,10,11 (months)
    new_dataset_2['date'] = new_dataset_2['date'].apply(lambda x: int(x.split('/')[1]))

    variables = ds.get_variable_types(new_dataset_2)
    new_dataset_2 = dummify(new_dataset_2, symbolic_vars)

    new_dataset_2.to_csv(f'data/report/dummified/air_quality_dummified_v3.csv', index=False)

    
def outliers_imputation(new_dataset_2):
    new_dataset_2 = new_dataset_2[new_dataset_2.CO_Mean<10]
    new_dataset_2 = new_dataset_2[new_dataset_2.CO_Max<40]
    new_dataset_2 = new_dataset_2[new_dataset_2.CO_Std<15]
    new_dataset_2 = new_dataset_2[new_dataset_2.NO2_Max<300]
    new_dataset_2 = new_dataset_2[new_dataset_2.NO2_Std<60]
    new_dataset_2 = new_dataset_2[new_dataset_2.O3_Max<400]
    new_dataset_2 = new_dataset_2[new_dataset_2.O3_Std<125]
    return new_dataset_2
    
def scalling_zscore(new_dataset_2):
    variable_types = ds.get_variable_types(new_dataset_2)
    numeric_vars = ['Field_1','CO_Mean', 'CO_Min', 'CO_Max', 'CO_Std', 'NO2_Mean', 'NO2_Min', 'NO2_Max', 'NO2_Std', 'O3_Mean', 'O3_Min', 'O3_Max', 'O3_Std', 'PM2.5_Mean', 'PM2.5_Min', 'PM2.5_Max', 'PM2.5_Std', 'PM10_Mean', 'PM10_Min', 'PM10_Max', 'PM10_Std', 'SO2_Mean', 'SO2_Min', 'SO2_Max', 'SO2_Std']
    symbolic_vars = []
    boolean_vars = variable_types['Binary']

    df_nr = new_dataset_2[numeric_vars]
    df_sb = new_dataset_2[symbolic_vars]
    df_bool = new_dataset_2[boolean_vars]

    transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(df_nr)
    tmp = DataFrame(transf.transform(df_nr), index=new_dataset_2.index, columns= numeric_vars)
    norm_data_zscore = concat([tmp, df_sb,  df_bool], axis=1)
    norm_data_zscore.to_csv(f'data/report/scaled/air_quality_scaled_zscore', index=False)

def scalling_minmax(new_dataset_2):
    variable_types = ds.get_variable_types(new_dataset_2)
    numeric_vars = ['Field_1','CO_Mean', 'CO_Min', 'CO_Max', 'CO_Std', 'NO2_Mean', 'NO2_Min', 'NO2_Max', 'NO2_Std', 'O3_Mean', 'O3_Min', 'O3_Max', 'O3_Std', 'PM2.5_Mean', 'PM2.5_Min', 'PM2.5_Max', 'PM2.5_Std', 'PM10_Mean', 'PM10_Min', 'PM10_Max', 'PM10_Std', 'SO2_Mean', 'SO2_Min', 'SO2_Max', 'SO2_Std']
    symbolic_vars = []
    boolean_vars = variable_types['Binary']

    df_nr = new_dataset_2[numeric_vars]
    df_sb = new_dataset_2[symbolic_vars]
    df_bool = new_dataset_2[boolean_vars]
    
    transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(df_nr)
    tmp = DataFrame(transf.transform(df_nr), index=new_dataset_2.index, columns= numeric_vars)
    norm_data_minmax = concat([tmp, df_sb,  df_bool], axis=1)
    norm_data_minmax.to_csv(f'data/report/scaled/air_quality_scaled_minmax', index=False)
