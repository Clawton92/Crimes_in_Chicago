import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics import regressionplots as smg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
import missingno as msno

df12_17 = pd.read_csv('data_sets/crimes-in-chicago/Chicago_Crimes_2012_to_2017.csv')
cols = df12_17.columns.tolist()
cols = [col.replace(' ', '_').lower() for col in cols]
df12_17.columns = cols

dfshort = df12_17[['block', 'primary_type', 'beat', 'district', 'ward', 'community_area', 'arrest', 'location_description', 'domestic', 'iucr', 'beat']]
'''
Block: 32774 unique items
    Primary Type: 33 unique items
    Beat: 302 unique items
    District: 24 unique items
    Ward: 50 unique items
    Community Area: 78 unique items
    Arrest: 2 unique (True/False)
    location_description: 142
    domestic:2 ()
    iucr: 365
    beat:302
    '''
#show frequnecy of rows with missing values
# miss_freq = msno.matrix(dfshort)

#removing all rows with any null value. District: 1, Ward: 14, Community Area: 40
dfdropna = dfshort.dropna()
dropped_rows_total = dfshort.shape[0] - dfdropna.shape[0]

#EDA
df_EDA = dfdropna[['arrest', 'ward', 'primary_type']]
#total arrests/non arrests
objects = ('Arrests', 'Non-arrests', 'Total_incidences')
y_pos = np.arange(len(objects))
plt.bar(y_pos, [dfdropna['arrest'].sum(), dfdropna['arrest'].count() - dfdropna['arrest'].sum(), dfdropna.shape[0]], align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Count')
plt.title('Arrests for reported incedences')
# plt.show()

#arrests for each ward
# df_ward_arrests = dfdropna[['ward', 'arrest']]
# df_ward_arr_count = df_ward_arrests.groupby('ward').agg({'arrest': 'sum'})
# objects = (np.arange(1,50))
# y_pos = np.arange(len(objects))
# plt.bar(y_pos, df_ward_arr_count['arrest'].values, align='center', alpha=0.5)
# plt.xticks(y_pos, objects)
# plt.ylabel('Arrest count')
# plt.title('Arrests count for each ward')
# plt.show()

def ward_group(val):
    if val in [50, 49, 40, 48, 47]:
        val = 100
    if val in [41, 39, 45, 38, 30]:
        val = 200
    if val in [46, 44, 43, 32, 2]:
        val = 300
    if val in [33, 35, 1, 31, 26]:
        val = 400
    if val in [36, 29, 37, 28, 24]:
        val = 500
    if val in [27, 42, 25, 11, 3]:
        val = 600
    if val in [22, 12, 14, 23, 15]:
        val = 700
    if val in [4, 5, 20, 16, 6]:
        val = 800
    if val in [13, 18, 17, 21, 19]:
        val = 900
    if val in [7, 8, 9, 10, 34]:
        val = 1000
    return val

def count_specific_item(df, col):
    return df.groupby(col).size()

'''Grouping wards together. 50 wards total, creating 10 features -> 5 wards grouped. Groups:
ward_dict = {100: [50, 49, 40, 48, 47], 200: [41, 39, 45, 38, 30], 300: [46, 44, 43, 32, 2], 400: [33, 35, 1, 31, 26], 500: [36, 29, 37, 28, 24],
600: [27, 42, 25, 11, 3], 700: [22, 12, 14, 23, 15] 800: [4, 5, 20, 16, 6], 900: [13, 18, 17, 21, 19], 1000: [7, 8, 9, 10, 34]}'''

dfward = dfdropna.copy()
#change ward to ints
dfward['ward'] = dfward['ward'].astype(int)

#Change wards to ward groups under ward_g using hundreds and then change to single integer
dfward['ward_g'] = dfward['ward'].apply(ward_group)
dfward['ward_g'] = (dfward['ward_g']/100).astype(int)

#changing domestic to int
dfward['domestic'] = dfward['domestic'].astype(int)

#changing arrest to int
dfward['arrest'] = dfward['arrest'].astype(int)

#get dummies for primary type
df_dummies_type = pd.get_dummies(dfward['primary_type'], prefix='type')
df_dummies_wards = pd.get_dummies(dfward['ward_g'], prefix='ward')

#joining ward and dummies dataframes
df_join1 = dfward.join(df_dummies_type)
df_joined = df_join1.join(df_dummies_wards)
#dropping unwanted rows
df_joined.drop(columns=['block', 'beat', 'primary_type', 'district', 'ward', 'community_area', 'location_description', 'iucr'], inplace=True)
# , 'primary_type', 'beat', 'district', 'community_area', 'location_desctription', 'iucr', 'ward']
#editing columns in joined dataframe
cols = df_joined.columns.tolist()
cols = [col.replace(' ', '_').lower() for col in cols]
df_joined.columns = cols
cols = [col.replace('_-_', '_') for col in cols]
df_joined.columns = cols


#making simplified dataset (2wards, 2types of crime, dropping
def north_south(val):
    if val in [50, 49, 40, 48, 47, 41, 39, 45, 38, 30, 46, 44, 43, 32, 2, 33, 35, 1, 31, 26, 36, 29, 37, 28, 24]:
        val = 'north'
    if val in [27, 42, 25, 11, 3, 22, 12, 14, 23, 15, 4, 5, 20, 16, 6, 13, 18, 17, 21, 19, 7, 8, 9, 10, 34]:
        val = 'south'
    return val

def violent_nonviolent(val):
    if val in ['BATTERY', 'PUBLIC PEACE VIOLATION', 'THEFT', 'WEAPONS VIOLATION',\
        'ROBBERY', 'MOTOR VEHICLE THEFT', 'ASSAULT', 'CRIMINAL DAMAGE',\
       'BURGLARY', 'STALKING', 'CRIM SEXUAL ASSAULT',\
       'SEX OFFENSE', 'OFFENSE INVOLVING CHILDREN',\
       'KIDNAPPING', 'HOMICIDE', 'ARSON',\
       'HUMAN TRAFFICKING']:
       val = 'violent'
    else:
        val = 'non_violent'
    return val

#making test data set with full dummies for wards and type, this includes domestic
df_full_dummies_set = df_joined.drop('ward_g', axis=1)

#making test data set with north/south, violent/nonviolent, arrest, domestic. Random choose 1000.
dfward['north_south'] = dfward['ward'].apply(north_south)
dfward['v_nonv'] = dfward['primary_type'].apply(violent_nonviolent)

df_dummies_ns = pd.get_dummies(dfward['north_south'], prefix='loc')
df_dummies_v_nonv = pd.get_dummies(dfward['v_nonv'], prefix='crime')

df_temp = dfward[['arrest', 'domestic']]
df_temp2 = df_temp.join(df_dummies_ns)
df_test = df_temp2.join(df_dummies_v_nonv)

def pd_concat_sampled_df(df, col, val1, val2, sample_size):
    query_string1 = "{} == {}".format(col, val1)
    query_string2 = "{} == {}".format(col, val2)
    first = df.query(query_string1)
    second = df.query(query_string2)
    samp1 = first.sample(sample_size)
    samp2 = second.sample(sample_size)
    samp1.index = range(sample_size)
    samp2.index = range(sample_size)
    frames = [samp1, samp2]
    final = pd.concat(frames)
    final_shuffle = final.sample(frac=1).reset_index(drop=True)
    return final

if __name__=='__main__':
    five_feat_balanced = pd_concat_sampled_df(df_test, 'arrest', 1, 0, df_test.query('arrest == 1').shape[0])
    five_feat_balanced.to_csv('data_sets/five_feat_balanced.csv')

    full_feat_balanced = pd_concat_sampled_df(df_full_dummies_set, 'arrest', 1, 0, df_full_dummies_set.query('arrest == 1').shape[0])
    full_feat_balanced.to_csv('data_sets/full_feat_balanced.csv')
