import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from numpy import ndarray
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import libs.ds_charts as ds
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame, concat
from libs.ds_charts import get_variable_types
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

dataset_1 = pd.read_csv('data/report/NYC_collisions_tabular.csv', na_values='NaN')
new_dataset_1 = dataset_1.copy()

def missing_values(new_dataset_1):
    #FIND VARIABLES WITH MISSING VALUES
    mv = {}
    for var in new_dataset_1:
        nr = new_dataset_1[var].isna().sum()
        if nr > 0:
            mv[var] = nr

    #DISCARD COLUMNS WITH MORE THEN 90% MISSING VALUES
    threshold = new_dataset_1.shape[0] * 0.85

    missings = [c for c in mv.keys() if mv[c]>threshold]
    new_dataset_1.drop(columns=missings, inplace=True)
    print('Dropped variables', missings)

    #DISCARD RECORDS WITH MAJORITY OF MISSING VALUES
    threshold = new_dataset_1.shape[1] * 0.50

    new_dataset_1.dropna(thresh=threshold, inplace=True)
    print(new_dataset_1.shape)

    #PERSON_AGE
    person_age = dataset_1['PERSON_AGE']
    mean_ages = int(person_age.mean())
    new_dataset_1['PERSON_AGE'].fillna(mean_ages,inplace=True)

    #SAFETY_EQUIPMENT
    new_dataset_1['SAFETY_EQUIPMENT'].fillna('Unknown',inplace=True)

    #EJECTION
    new_dataset_1['EJECTION'].fillna('Not Ejected',inplace=True)

    #VEHICLE_ID
    new_dataset_1['VEHICLE_ID'].fillna(0,inplace=True)

    #POSITION IN VEHICLE
    new_dataset_1['POSITION_IN_VEHICLE'].fillna('Unknown',inplace=True)
    return new_dataset_1
    
def dummification_v1(new_dataset_1):
    symbolic_vars = ['CRASH_TIME','CRASH_DATE','BODILY_INJURY','SAFETY_EQUIPMENT','PERSON_SEX','EJECTION','PERSON_TYPE','COMPLAINT','EMOTIONAL_STATUS','POSITION_IN_VEHICLE','PED_ROLE']

    ## Transform hours to 0-22
    new_dataset_1['CRASH_TIME'] = new_dataset_1['CRASH_TIME'].apply(lambda x: int(x.split(':')[0]))
    
    ## Transform months to 0-11
    new_dataset_1['CRASH_DATE'] = new_dataset_1['CRASH_DATE'].apply(lambda x: int(x.split('/')[1]))

    def dummify(df, vars_to_dummify):
        other_vars = [c for c in df.columns if not c in vars_to_dummify]
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=bool)
        X = df[vars_to_dummify]
        encoder.fit(X)
        new_vars = encoder.get_feature_names_out(vars_to_dummify)
        trans_X = encoder.transform(X)
        dummy = pd.DataFrame(trans_X, columns=new_vars, index=X.index)
        dummy = dummy.convert_dtypes(convert_boolean=True)

        final_df = pd.concat([df[other_vars], dummy], axis=1)
        return final_df

    variables = get_variable_types(new_dataset_1)
    new_dataset_1 = dummify(new_dataset_1, symbolic_vars)

    #remove ids, 
    new_dataset_1.drop('PERSON_ID',axis=1,inplace=True)
    new_dataset_1.drop('UNIQUE_ID',axis=1,inplace=True)
    new_dataset_1.drop('COLLISION_ID',axis=1,inplace=True)
    new_dataset_1.drop('VEHICLE_ID',axis=1,inplace=True)
    new_dataset_1.drop('BODILY_INJURY_Unknown',axis=1,inplace=True)
    new_dataset_1.drop('SAFETY_EQUIPMENT_None',axis=1,inplace=True)
    new_dataset_1.drop('SAFETY_EQUIPMENT_Unknown',axis=1,inplace=True)
    new_dataset_1.drop('COMPLAINT_Unknown',axis=1,inplace=True)
    new_dataset_1.drop('EMOTIONAL_STATUS_Unknown',axis=1,inplace=True)
    new_dataset_1.drop('POSITION_IN_VEHICLE_Unknown',axis=1,inplace=True)



    new_dataset_1.to_csv(f'data/report/dummified/nyc_collisions_dummified_v1.csv', index=False)
    nr = new_dataset_1.isna().sum()

def dummification_v2(new_dataset_1):
    week_days=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

    ### weekdays
    new_dataset_1['CRASH_DATE'] = new_dataset_1['CRASH_DATE'].apply(lambda x: week_days[datetime.date(int(x.split('/')[0]),int(x.split('/')[1]),int(x.split('/')[0])).weekday()]) 
     
    symbolic_vars = ['CRASH_TIME','CRASH_DATE','BODILY_INJURY','SAFETY_EQUIPMENT','PERSON_SEX','EJECTION','PERSON_TYPE','COMPLAINT','EMOTIONAL_STATUS','POSITION_IN_VEHICLE','PED_ROLE']

    ## Transform hours to 0-22
    new_dataset_1['CRASH_TIME'] = new_dataset_1['CRASH_TIME'].apply(lambda x: int(x.split(':')[0]))

    def dummify(df, vars_to_dummify):
        other_vars = [c for c in df.columns if not c in vars_to_dummify]
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=bool)
        X = df[vars_to_dummify]
        encoder.fit(X)
        new_vars = encoder.get_feature_names_out(vars_to_dummify)
        trans_X = encoder.transform(X)
        dummy = pd.DataFrame(trans_X, columns=new_vars, index=X.index)
        dummy = dummy.convert_dtypes(convert_boolean=True)

        final_df = pd.concat([df[other_vars], dummy], axis=1)
        return final_df

    variables = get_variable_types(new_dataset_1)
    new_dataset_1 = dummify(new_dataset_1, symbolic_vars)

    #remove ids, 
    new_dataset_1.drop('PERSON_ID',axis=1,inplace=True)
    new_dataset_1.drop('UNIQUE_ID',axis=1,inplace=True)
    new_dataset_1.drop('COLLISION_ID',axis=1,inplace=True)
    new_dataset_1.drop('VEHICLE_ID',axis=1,inplace=True)
    new_dataset_1.drop('BODILY_INJURY_Unknown',axis=1,inplace=True)
    new_dataset_1.drop('SAFETY_EQUIPMENT_None',axis=1,inplace=True)
    new_dataset_1.drop('SAFETY_EQUIPMENT_Unknown',axis=1,inplace=True)
    new_dataset_1.drop('COMPLAINT_Unknown',axis=1,inplace=True)
    new_dataset_1.drop('EMOTIONAL_STATUS_Unknown',axis=1,inplace=True)
    new_dataset_1.drop('POSITION_IN_VEHICLE_Unknown',axis=1,inplace=True)

    new_dataset_1.to_csv(f'data/report/dummified/nyc_collisions_dummified_v2.csv', index=False)
    nr = new_dataset_1.isna().sum()
    
def outliers_imputation(new_dataset_1):
    new_dataset_1 = new_dataset_1[(new_dataset_1.PERSON_AGE < 110) & (new_dataset_1.PERSON_AGE > 0)]
    return new_dataset_1

def scalling_zscore(new_dataset_1):
    variable_types = get_variable_types(new_dataset_1)
    numeric_vars = ['PERSON_AGE']
    symbolic_vars2 = []
    boolean_vars = variable_types['Binary']

    df_nr = new_dataset_1[numeric_vars]
    df_sb = new_dataset_1[symbolic_vars2]
    df_bool = new_dataset_1[boolean_vars]
    
    transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(df_nr)
    tmp = pd.DataFrame(transf.transform(df_nr), index=new_dataset_1.index, columns= numeric_vars)
    norm_data_zscore = pd.concat([tmp, df_sb,  df_bool], axis=1)
    norm_data_zscore.to_csv(f'data/report/scaled/nyc_collisions_scaled_zscore.csv', index=False)

def scalling_zscore(new_dataset_1):
    variable_types = get_variable_types(new_dataset_1)
    numeric_vars = ['PERSON_AGE']
    symbolic_vars2 = []
    boolean_vars = variable_types['Binary']

    df_nr = new_dataset_1[numeric_vars]
    df_sb = new_dataset_1[symbolic_vars2]
    df_bool = new_dataset_1[boolean_vars]
    
    transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(df_nr)
    tmp = pd.DataFrame(transf.transform(df_nr), index=new_dataset_1.index, columns= numeric_vars)
    norm_data_minmax = pd.concat([tmp, df_sb,  df_bool], axis=1)
    norm_data_minmax.to_csv(f'data/report/scaled/nyc_collisions_scaled_minmax.csv', index=False)
