import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
import pickle


#Data Gathering
ZomatoData=pd.read_csv('ZomatoData.csv', encoding='latin')
ZomatoData=ZomatoData.drop('Unnamed: 0', axis=1)

#Initial Data Study
UselessColumns = ['RestaurantID', 'RestaurantName','Longitude','Latitude','Address',
                  'Locality', 'LocalityVerbose','Currency']
ZomatoData = ZomatoData.drop(UselessColumns,axis=1)

#Feature Extraction
ZomatoData['Number_of_Cuisines'] = ZomatoData['Cuisines'].apply(lambda x: len((str(x).split(","))))
ZomatoData = ZomatoData.drop('Cuisines',axis=1)

#Data Processing
ZomatoData=ZomatoData.drop_duplicates()

#Outliers Treatment
mean2 = np.mean(ZomatoData['Votes'])
std2 = np.std(ZomatoData['Votes'])
threshold2 = 3
outlier2 = []
for i in ZomatoData['Votes']:
    z2 = (float(i)-(mean2))/std2
    if z2 > threshold2:
        outlier2.append(i)
ol2 = sorted(ZomatoData['Votes'][ZomatoData['Votes']< (sorted(outlier2)[0])], reverse=True)[0]
ZomatoData['Votes'][ZomatoData['Votes']>ol2 ] = ol2

#Skewness Correction
ZomatoData['Votes'] = np.cbrt(ZomatoData['Votes'])

#Feature Selection
SelectedColumns=['Votes','Has_Table_booking','Has_Online_delivery','Price_range','Number_of_Cuisines']
# Selecting final columns
DataForML=ZomatoData[SelectedColumns]



#Data Encoding
# Converting the binary nominal variable sex to numeric
DataForML['Has_Table_booking'].replace({'Yes':1, 'No':0}, inplace=True)
DataForML['Has_Online_delivery'].replace({'Yes':1, 'No':0}, inplace=True)

# Treating all the nominal variables at once using dummy variables
DataForML_Numeric=pd.get_dummies(DataForML)

# Adding Target Variable to the data
DataForML_Numeric['Rating']=ZomatoData['Rating']



#Holdout Validation
# Separate Target Variable and Predictor Variables
TargetVariable='Rating'
Predictors=['Votes','Has_Table_booking',
           'Has_Online_delivery', 'Price_range', 'Number_of_Cuisines']

X=DataForML_Numeric[Predictors].values
y=DataForML_Numeric[TargetVariable].values

# Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6)




#Feature Scaling

#Using Min Max Normalization
PredictorScaler=MinMaxScaler()
# Storing the fit object for later reference
PredictorScalerFit=PredictorScaler.fit(X)
# Generating the standardized values of X
X=PredictorScalerFit.transform(X)
y=DataForML_Numeric[TargetVariable].values
# Split the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6)


#Fitting the model
from sklearn.tree import DecisionTreeRegressor
DT = DecisionTreeRegressor()
DT_hpt = DecisionTreeRegressor(max_depth=7, min_samples_leaf=6, min_samples_split=7)
DT_hpt_mod = DT_hpt.fit(X_train, y_train)

#Saving the Pickle file
with open("rrg_mod.pkl",'wb') as f:
    pickle.dump(DT_hpt_mod, f)

#Loading the Pickle file
# with open(r'rrg_mod.pickle', 'rb') as f:
#     model = pickle.load(f)

#print(model.predict([[5,1,0,3,5]]))