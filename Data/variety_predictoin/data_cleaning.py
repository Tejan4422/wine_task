import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix


traindf = pd.read_csv('train.csv')
testdf = pd.read_csv('test.csv')

print("Total number of examples: ", traindf.shape[0])
print("Number of examples with the same title and description: ", traindf[traindf.duplicated(['review_description','review_title'])].shape[0])
#dropping duplicates
dropped_duplicates=traindf.drop_duplicates(['review_description','review_title'])

dropped_duplicates_test=testdf.drop_duplicates(['review_description','review_title'])
dropped_duplicates=dropped_duplicates.reset_index(drop=True)
dropped_duplicates_test=dropped_duplicates_test.reset_index(drop=True)


dropped_duplicates.info()
dropped_duplicates.isna().sum()
dropped_duplicates.nunique()
dropped_duplicates.variety.unique()

dropped_duplicates = dropped_duplicates.assign(description_length = dropped_duplicates['review_description'].apply(len))
dropped_duplicates = dropped_duplicates.assign(title_length = dropped_duplicates['review_title'].apply(len))

dropped_duplicates_test = dropped_duplicates_test.assign(description_length = dropped_duplicates_test['review_description'].apply(len))
dropped_duplicates_test = dropped_duplicates_test.assign(title_length = dropped_duplicates_test['review_title'].apply(len))


#converting target variable to numericals 
var_dict = {'Chardonnay' : 1, 'Red Blend' : 2, 'Nebbiolo' : 3, 'Bordeaux-style White Blend' : 4,
            'Malbec' : 5, 'Cabernet Sauvignon' : 6, 'Zinfandel' : 7,
            'Pinot Noir' : 8, 'Sauvignon Blanc' : 9, 'Gamay' : 10, 'Grüner Veltliner' : 11,
            'Bordeaux-style Red Blend' : 12, 'Sangiovese' : 13, 'Syrah' : 14, 
            'White Blend' : 15, 'Cabernet Franc' : 16, 'Portuguese Red' : 17,
            'Portuguese White' : 18, 'Rhône-style Red Blend' : 19, 'Rosé' : 20,
            'Champagne Blend' : 21, 'Merlot' : 22, 'Riesling' : 23, 'Sparkling Blend' : 24,
            'Pinot Grigio' : 25, 'Tempranillo' : 26, 'Pinot Gris' : 27, 'Gewürztraminer' : 28,
            }
dropped_duplicates.variety = [var_dict[item] for item in dropped_duplicates.variety]

training_data_final = dropped_duplicates[['country', 'price', 'points', 'province','region_1', 'region_2', 'description_length', 'title_length']]
training_data_final['price'] = training_data_final.price.fillna(training_data_final.price.mean()) 
training_data_final['province'] = training_data_final.province.fillna(-1)
training_data_final['region_1'] = training_data_final.region_2.fillna(-1)
training_data_final['region_2'] = training_data_final.region_2.fillna(-1)
training_data_final.isna().sum()

training_data = dropped_duplicates.iloc[0:50000, [1, 5, 6, 7, 8, 9, 12, 13]]
training_data['price'] = training_data.price.fillna(training_data.price.mean()) 
training_data['province'] = training_data.province.fillna(-1)
training_data['region_1'] = training_data.region_2.fillna(-1)
training_data['region_2'] = training_data.region_2.fillna(-1)
training_data.isna().sum()

test_data_final = dropped_duplicates_test[['country', 'price', 'points', 'province','region_1', 'region_2', 'description_length', 'title_length']]
test_data_final['price'] = test_data_final.price.fillna(test_data_final.price.mean()) 
test_data_final['province'] = test_data_final.province.fillna(-1)
test_data_final['region_1'] = test_data_final.region_2.fillna(-1)
test_data_final['region_2'] = test_data_final.region_2.fillna(-1)
test_data_final.isna().sum()



df_dum = pd.get_dummies(training_data_final)
df_dum.isna().sum()                                                                                                                                                                                                     

df_dum_test = pd.get_dummies(test_data_final)
df_dum.isna().sum()                                                                                                                                                                                                     


test_data_final = dropped_duplicates['variety'].values                                                                                                                                                                                                                                                                         
test_data = dropped_duplicates.iloc[0:50000, 11].values

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(df_dum)
X_testdf = sc_X.fit_transform(df_dum_test)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, test_data_final, test_size = 0.1)

from sklearn.ensemble import RandomForestClassifier
#hyperparameter tuning on small subset
rf = RandomForestClassifier()
parameters_rf = {
        'n_estimators' : [50, 250],
        'max_depth' : [8,16,32, None]
        }
cv_rf = GridSearchCV(rf, parameters_rf, cv = 5)
cv_rf.fit(X_train, y_train)
cv_rf.best_estimator_
y_pred_rf = cv_rf.best_estimator_.predict(X_test)
cm_rf = confusion_matrix(y_test, y_pred_rf)
accuracy_score(y_test, y_pred_rf)

#normal training 
rf1 = RandomForestClassifier()
rf1.fit(X_train, y_train)
y_pred_rf1 = rf1.predict(X_test)
cm_rf1 = confusion_matrix(y_test, y_pred_rf1)
accuracy_score(y_test, y_pred_rf1)

from sklearn.neural_network import MLPClassifier
nn = MLPClassifier()
nn.fit(X_train, y_train)
y_pred_nn = nn.predict(X_test)
cm_nn = confusion_matrix(y_test, y_pred_nn)
accuracy_score(y_test, y_pred_nn)

#final training on Train set
rf_final = RandomForestClassifier()
parameters_rf = {
        'n_estimators' : [50, 250],
        'max_depth' : [8,16,32, None]
        }
cv_rf = GridSearchCV(rf_final, parameters_rf, cv = 5)
cv_rf.fit(X, test_data_final)
cv_rf.best_estimator_

nn = MLPClassifier()
nn.fit(X, test_data_final)

