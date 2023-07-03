import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split


pd.set_option('display.float_format', lambda x: '{:.4f}'.format(x)) #Limiting 4 decimal
plt.rcParams["figure.figsize"] = [9,5]
plt.style.use('ggplot')

data_df = pd.read_csv("forestfires.csv")

#print(data_df.info)

# We need to convert month and day to either int or float from object data type or just drop them really
# Other observation is no missing values are present so we are good to go ahead.

df = data_df.drop(['month', 'day'], axis=1)
#print(df)

#print("Skew: \n{}".format(df.skew()))
#print("\nKurtosis: \n{}".format(df.kurtosis()))

# After analyzing skew and kurtosis we find that the outliers in some columns is very much so we remove them using zscore and log operations

outlier_columns = ['area','FFMC','ISI','rain']
# Performing the logarithmic operation which gives output ln(x+1) where x is all elements of input array

#print(np.log1p(data_df[outlier_columns]).skew())
#print("\nKurtosis:\n")
#print(np.log1p(data_df[outlier_columns]).kurtosis())

# Even after transformation we still have high skewness and kurtosis in FFMC & rain

# Removing outliers by zscore method.

mask = data_df.loc[:,['FFMC']].apply(zscore).abs() < 3
data_df = data_df[mask.values]
#print(data_df.shape)

## Since most of the values in rain are 0.0, we can convert it as a categorical column
data_df['rain'] = data_df['rain'].apply(lambda x: int(x > 0.0))
outlier_columns.remove('rain')
data_df[outlier_columns] = np.log1p(data_df[outlier_columns])

#print(data_df[outlier_columns].skew())
print(data_df[outlier_columns].kurtosis())
#print(data_df.describe())

### MODEL PREP -->

data_sel = data_df.copy()

le = LabelEncoder() 
  
data_sel['day'] = le.fit_transform(data_sel['day']) 
data_sel['month'] = le.fit_transform(data_sel['month']) 
X, y = data_sel.iloc[:, : -1], data_sel.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 15)

#eval_set = [(X_test, y_test)]
eval_set = [(X_train, y_train), (X_test, y_test)]

# verbose set to False so that we can hide results of model fit progress
xg_reg.fit(X_train, y_train, eval_metric=["rmse"], eval_set = eval_set)
preds = xg_reg.predict(X_test)


def calc_ISE(X_train, y_train, model):
    '''returns the in-sample R^2 and RMSE; assumes model already fit.'''
    predictions = model.predict(X_train)
    mse = mean_squared_error(y_train, predictions)
    rmse = np.sqrt(mse)
    return model.score(X_train, y_train), rmse
    
def calc_OSE(X_test, y_test, model):
    '''returns the out-of-sample R^2 and RMSE; assumes model already fit.'''
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    return model.score(X_test, y_test), rmse

# Calculate In-Sample and Out-of-Sample R^2 and Error

is_r2, ise = calc_ISE(X_train, y_train,xg_reg )
os_r2, ose = calc_OSE(X_test, y_test, xg_reg)

# show dataset sizes
data_list = (('R^2_in', is_r2), ('R^2_out', os_r2), 
             ('ISE', ise), ('OSE', ose))
for item in data_list:
    print('{:10}: {}'.format(item[0], item[1]))

# Clearly test error(OSE) is near to the training error(ISE). i.e our model is ok.

print('train/test: ',ose/ise)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

#xgb.plot_importance(xg_reg)
#plt.rcParams['figure.figsize'] = [7, 7]
#plt.show()

# retrieve performance metrics
results = xg_reg.evals_result()
epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)

# plot Learning Curve
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
ax.legend()
plt.ylabel('RMSE')
plt.title('XGBoost RMSE')
plt.show()

# Saving model for future prediction on different datasets

xg_reg.save_model('0001_model_forest_fire.json')