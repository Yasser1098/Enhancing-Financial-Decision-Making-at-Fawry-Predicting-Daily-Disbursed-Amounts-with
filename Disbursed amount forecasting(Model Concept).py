#!/usr/bin/env python
# coding: utf-8

# ## Importing required packages and libraries

# In[163]:


import pandas as pd
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')
pd.options.mode.chained_assignment = None


# ## Data Exploration

# In[164]:


my_df= pd.read_excel(r'D:\Sarima Forcasting\Data for model.xlsx',sheet_name='XGBoost-Data')
##DATA CLIPING#####
# Calculate the 95th percentile to identify outliers
outlier_threshold = my_df['Disbursedamount'].quantile(0.985)

# Clip the outliers to a specific value (e.g., 8200000)
my_df['Disbursedamount'] = my_df['Disbursedamount'].clip(upper=outlier_threshold)


# In[165]:


plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
sns.boxplot(data=my_df)

# Add labels and title
plt.xlabel('X-axis Label')  # Replace 'X-axis Label' with an appropriate label
plt.ylabel('Y-axis Label')  # Replace 'Y-axis Label' with an appropriate label
plt.title('Box Plot of Data')  # Replace 'Box Plot of Data' with an appropriate title

# Show the plot
plt.show()


# In[166]:


my_df_on= my_df.set_index('Date')
my_df_on.index = pd.to_datetime(my_df_on.index)

my_df_all= my_df_on[:1521]


# In[167]:


my_df_all.plot(figsize=(15,5),color=color_pal[0],title='Disbursed Amount Daily')
plt.show()


# ## Spliting Data into 3 sets (train-Test-Validation)

# In[168]:


my_df_train= my_df_on[:1492]
my_df_validation= my_df_on[1492:1521]
my_df_test= my_df_on[1521:]


# In[169]:


fig, ax= plt.subplots(figsize=(15,5))
my_df_train.plot(ax=ax,label='Train')
my_df_validation.plot(ax=ax,label='Validation')
ax.legend(['Train Set', 'Validation Set'])
plt.show()


# In[170]:


my_df_all.loc[(my_df_all.index > '01-01-2021') & (my_df_all.index < '05-01-2021')] \
    .plot(figsize=(15, 2), title='From Jan-May 2021')

my_df_all.loc[(my_df_all.index > '01-01-2022') & (my_df_all.index < '05-01-2022')] \
    .plot(figsize=(15, 2), title='From Jan-May 2022')

my_df_all.loc[(my_df_all.index > '01-01-2023') & (my_df_all.index < '05-01-2023')] \
    .plot(figsize=(15, 2), title='From Jan-May 2023')
plt.show()


# In[171]:


def create_features(df):
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    df['dayofmonth'] = df.index.day
    df['dayofweek'] = df.index.dayofweek
    df['dayofyear'] = df.index.dayofyear
    df['weekofyear'] = df.index.isocalendar().week
    
    return df

my_df_all= create_features(my_df_all)
my_df_all


# In[172]:


fig, axes = plt.subplots(2, 1, figsize=(10, 10))

# Plot the first boxplot in the left subplot
sns.boxplot(data=my_df_all, x='dayofmonth', y='Disbursedamount', ax=axes[0])
axes[0].set_title('Disb by day of month')

# Plot the second boxplot in the right subplot
sns.boxplot(data=my_df_all, x='dayofweek', y='Disbursedamount', ax=axes[1])
axes[1].set_title('Disb by day of week')

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()


# In[173]:


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot the first boxplot in the left subplot
sns.boxplot(data=my_df_all, x='quarter', y='Disbursedamount', ax=axes[0])
axes[0].set_title('Disb by quarter')

# Plot the second boxplot in the right subplot
sns.boxplot(data=my_df_all, x='month', y='Disbursedamount', ax=axes[1])
axes[1].set_title('Disb by month')

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()


# ## Grid Search to tune the Hyperparameters

# In[174]:


# train_set = create_features(my_df_train)
# val_set = create_features(my_df_validation)

# combined_data = pd.concat([train_set, val_set], axis=0)

# Features = ['month', 'quarter','year','dayofmonth',
#             'dayofweek', 'dayofyear', 'weekofyear','vacation','Weekends','Ramadan']
# Target = 'Disbursedamount'

# X_combined = combined_data[Features]
# Y_combined = combined_data[Target]

# # Split the combined data into training and validation sets
# X_train, X_val, Y_train, Y_val = train_test_split(X_combined, Y_combined, test_size=0.2, random_state=42)

# # XGBoost model
# reg = xgb.XGBRegressor(
#     base_score=0.5,
#     booster='gbtree',
#     n_estimators=1000,
#     early_stopping_rounds=50,
#     objective='reg:linear',
#     max_depth=3,
#     learning_rate=0.01
# )

# param_grid = {
#     'learning_rate': [0.01, 0.05, 0.1],
#     'n_estimators': [100, 200, 500],
#     'max_depth': [3, 5, 7],
#     'min_child_weight': [1, 3, 5],
#     'subsample': [0.5, 0.8, 1.0],
#     'colsample_bytree': [0.5, 0.8, 1.0],
#     'gamma': [0, 0.1, 0.2],
#     'reg_alpha': [0, 0.1, 0.5],
#     'reg_lambda': [0, 0.1, 0.5]
# }

# grid_search = GridSearchCV(estimator=reg, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
# grid_search.fit(X_train, Y_train, eval_set=[(X_val, Y_val)], verbose=100)

# best_params_xgb = grid_search.best_params_

# print("Best Parameters for XGBoost:", best_params_xgb)

# # Random Forest model
# reg_rf = RandomForestRegressor(random_state=0)

# rf_param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [5, 10, 15]
# }

# rf_grid_search = GridSearchCV(estimator=reg_rf, param_grid=rf_param_grid, cv=3, n_jobs=-1, verbose=2)
# rf_grid_search.fit(X_combined, Y_combined)
# best_params_rf = rf_grid_search.best_params_

# # Gradient Boosting model
# reg_gb = GradientBoostingRegressor(random_state=0)

# gb_param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [3, 5, 7],
#     'learning_rate': [0.01, 0.05, 0.1]
# }

# gb_grid_search = GridSearchCV(estimator=reg_gb, param_grid=gb_param_grid, cv=3, n_jobs=-1, verbose=2)
# gb_grid_search.fit(X_combined, Y_combined)
# best_params_gb = gb_grid_search.best_params_

# print("Random Forest Best Params:", best_params_rf)
# print("Gradient Boosting Best Params:", best_params_gb)


# ## Models Training

# In[175]:


train_set = create_features(my_df_train)
val_set = create_features(my_df_validation)


Features = ['month', 'quarter','year','dayofmonth',
            'dayofweek', 'dayofyear', 'weekofyear','vacation','Weekends','Ramadan']
Target = 'Disbursedamount'

X_train= train_set[Features]
Y_train= train_set[Target]

X_test= val_set[Features]
Y_test= val_set[Target]

# XGBoost model with best parameters
reg_xgb = xgb.XGBRegressor(
    base_score=0.5,
    booster='gbtree',
    n_estimators=500,
    early_stopping_rounds=50,
    objective='reg:linear',
    max_depth=7,
    learning_rate=0.01,
    colsample_bytree=1.0,
    gamma=0,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=0.5,
    subsample=0.5
)

# Random Forest model with best parameters
reg_rf = RandomForestRegressor(
    random_state=0,
    n_estimators=300,
    max_depth=10
)

# Gradient Boosting model with best parameters
reg_gb = GradientBoostingRegressor(
    random_state=0,
    n_estimators=100,
    max_depth=5,
    learning_rate=0.05
)

# Fit the models
reg_xgb.fit(X_train, Y_train, eval_set=[(X_train, Y_train), (X_test, Y_test)], verbose=100)
reg_rf.fit(X_train, Y_train)
reg_gb.fit(X_train, Y_train)


# ## Validating Data Prediction

# In[176]:


val_set['prediction_xgb'] = reg_xgb.predict(X_test)
val_set['prediction_rf'] = reg_rf.predict(X_test)
val_set['prediction_gb'] = reg_gb.predict(X_test)
# Merge the columns from val_set into my_df_all
my_df_all = my_df_all.merge(val_set[['prediction_xgb','prediction_rf','prediction_gb']], how='left', left_index=True, right_index=True)

# Plot the data
ax = my_df_all[['Disbursedamount']][1492:].plot(figsize=(15, 5))
my_df_all['prediction_xgb'][1492:].plot(ax=ax, style='-')
my_df_all['prediction_rf'][1492:].plot(ax=ax, style='--')
my_df_all['prediction_gb'][1492:].plot(ax=ax, style=':')
plt.legend(['Truth Data', 'XGBoost Prediction', 'Random Forest Prediction', 'Gradient Boosting Prediction'])
ax.set_title('Raw Data and Predictions')
plt.show()


# In[177]:


pd.set_option('display.float_format', lambda x: '{:,.0f}'.format(x))

# Your code to convert columns to float
my_df_all['Disbursedamount'] = my_df_all['Disbursedamount'].astype(float)
my_df_all['prediction_xgb'] = my_df_all['prediction_xgb'].astype(float)
my_df_all['prediction_rf'] = my_df_all['prediction_rf'].astype(float)
my_df_all['prediction_gb'] = my_df_all['prediction_gb'].astype(float)

# Display the DataFrame
Models_Data_df= my_df_all[['Disbursedamount', 'prediction_xgb', 'prediction_rf', 'prediction_gb']].tail(29)
Models_Data_df


# In[179]:


import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Assuming Models_Data_df is your DataFrame containing the data
# Columns: 'Disbursedamount', 'prediction_xgb', 'prediction_rf', 'prediction_gb'

# Calculate MAE, MSE, and RMSE for each model
mae_xgb = mean_absolute_error(Models_Data_df['Disbursedamount'], Models_Data_df['prediction_xgb'])
rmse_xgb = mse_xgb ** 0.5

mae_rf = mean_absolute_error(Models_Data_df['Disbursedamount'], Models_Data_df['prediction_rf'])
rmse_rf = mse_rf ** 0.5

mae_gb = mean_absolute_error(Models_Data_df['Disbursedamount'], Models_Data_df['prediction_gb'])
rmse_gb = mse_gb ** 0.5

# # Print the error metrics
# print("XGBoost Model:")
# print("Mean Absolute Error (MAE):", mae_xgb)
# print("Root Mean Squared Error (RMSE):", rmse_xgb)
# print()

# print("Random Forest Model:")
# print("Mean Absolute Error (MAE):", mae_rf)
# print("Root Mean Squared Error (RMSE):", rmse_rf)
# print()

# print("Gradient Boosting Model:")
# print("Mean Absolute Error (MAE):", mae_gb)
# print("Root Mean Squared Error (RMSE):", rmse_gb)


# # 📊 Model Evaluation Results:
# 
# ## <span style="color:blue">Random Forest Model:</span>
# 
# - The Random Forest model performed the best among the three, with the lowest error metrics.
#   - **Mean Absolute Error (MAE):** 814,352
#   - **Root Mean Squared Error (RMSE):** 2,231,771
#   
#  ## <span style="color:blue">Gradient Boosting Model:</span>
# 
# - The Gradient Boosting model showed competitive performance, it had slightly higher error metrics compared to Random Forest.
#   - **Mean Absolute Error (MAE):** 862,539
#   - **Root Mean Squared Error (RMSE):** 2,082,167
# 
# ## <span style="color:blue">XGBoost Model:</span>
# 
# - While the XGBoost has the least performance.
#   - **Mean Absolute Error (MAE):** 999,450
#   - **Root Mean Squared Error (RMSE):** 2,004,963
# 
# # 🏆 Conclusion:
# 
# Considering the evaluation results, the Random Forest model emerges as the top performer for this particular dataset and task. However, it's worth noting that the Gradient Boosting model showed competitive performance as well. As always, it's crucial to consider other factors such as model complexity, interpretability, and computational efficiency when selecting the best model for your specific application.
# 
