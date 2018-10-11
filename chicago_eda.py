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

#removing all rows with any null value. District: 1, Ward: 14, Community Area: 40
dfdropna = dfshort.dropna()
dropped_rows_total = dfshort.shape[0] - dfdropna.shape[0]

#EDA
df_EDA_ward = dfdropna[['arrest', 'ward']]
df_ward_dummies = pd.get_dummies(df_EDA_ward['arrest'], prefix='arrest')
df_plot_ward = df_EDA_ward.join(df_ward_dummies)
df_plot_ward.drop('arrest', axis=1, inplace=True)
df_plot_count_ward = df_plot_ward.groupby('ward').agg({'arrest_False': 'sum', 'arrest_True': 'sum'})
df_plot_count_ward.reset_index(inplace=True)
df_plot_ward_short = df_plot_count_ward.iloc[0:10]
df_plot_ward_short.plot(x="ward", y=["arrest_False", "arrest_True"], kind="bar", rot=0)
plt.show()

df_EDA_type = dfdropna[['primary_type']]
df_type_arr = df_EDA_type.join(df_ward_dummies)
df_plot_count_type = df_type_arr.groupby('primary_type').agg({'arrest_False': 'sum', 'arrest_True': 'sum'})
df_plot_count_type.reset_index(inplace=True)
# df_plot_type_short = df_plot_count_type.loc[df_plot_count_type['primary_type'].isin(['ARSON', 'ASSAULT', 'BURGLARY', 'HOMICIDE',\
#  'DECEPTIVE PRACTICE', 'CRIMINAL TRESPASS', 'CRIMINAL DAMAGE', 'MOTOR VEHICLE THEFT'])]
df_plot_type_short = df_plot_count_type.loc[df_plot_count_type['primary_type'].isin(['ARSON', 'ASSAULT', 'STALKING', 'ROBBERY',\
'PROSTITUTION', 'HOMICIDE', 'CRIMINAL DAMAGE'])]
df_plot_type_short.plot(x='primary_type', y=["arrest_False", "arrest_True"], kind="bar", rot=0)
plt.show()

#total arrests/non arrests
objects = ('Arrests', 'Non-arrests', 'Total_incidences')
y_pos = np.arange(len(objects))
plt.bar(y_pos, [dfdropna['arrest'].sum(), dfdropna['arrest'].count() - dfdropna['arrest'].sum(), dfdropna.shape[0]], align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Count')
plt.title('Arrests for reported incedences')
plt.show()
