import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# 1.Read in the data
weather = pd.read_csv("london_weather.csv",parse_dates=[0] #,index_col=1
                     )
weather.head()
weather.info()

# 2.Clean
# date
weather['month']=weather['date'].dt.month
weather['year']=weather['date'].dt.year

# 3.EDA
sns.pairplot(weather)
plt.show()

#fig, ax=plt.subplots(ncols=10, figsize=[30,5])
#sns.lineplot(x='month',y='mean_temp',data=weather, ax=ax[0])
#sns.lineplot(x='year',y='mean_temp',data=weather, ax=ax[1])
#sns.lineplot(x='cloud_cover',y='mean_temp',data=weather, ax=ax[2])
#sns.lineplot(x='sunshine',y='mean_temp',data=weather, ax=ax[3])
#sns.lineplot(x='global_radiation',y='mean_temp',data=weather, ax=ax[4])
#sns.lineplot(x='max_temp',y='mean_temp',data=weather, ax=ax[5])
#sns.lineplot(x='min_temp',y='mean_temp',data=weather, ax=ax[6])
#sns.lineplot(x='precipitation',y='mean_temp',data=weather, ax=ax[7])
#sns.lineplot(x='pressure',y='mean_temp',data=weather, ax=ax[8])
#sns.lineplot(x='snow_depth',y='mean_temp',data=weather, ax=ax[9])
#plt.show()

co_mtx = weather.corr(numeric_only=False)
# Plot correlation heatmap
sns.heatmap(co_mtx, cmap="YlGnBu", annot=True)
plt.show()

# 4.Feature Selection
feature_selection=["min_temp","global_radiation", "max_temp", "sunshine", "cloud_cover", "month", "year","pressure", "snow_depth"] #discard precipitation 
# to bin: cloud_cover, month, snow_depth, pressure, maybe year
weather=weather.dropna(subset=['mean_temp'])
y=weather['mean_temp'].copy()
X=weather[feature_selection]

# 5.Preprocess
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# importing pipes for making the Pipe flow
from sklearn.pipeline import Pipeline
# The sequence of pipe flow is :
# PCA dimension is reduced by 2 >> Data gets scaled >> Classification of decission tree
pipe = Pipeline([('simple', SimpleImputer()),('std', StandardScaler())], verbose = True)
 
# fitting the data in the pipeline
X_train=pipe.fit_transform(X_train)
X_test=pipe.transform(X_test)

# 6.ML Train & Eval
for idx, depth in enumerate([1, 2, 10]):
    run_name = f"run_{idx}"
    with mlflow.start_run(run_name=run_name) as run:
        # Model Training Code here
        lr = LinearRegression().fit(X_train, y_train)
        dt=DecisionTreeRegressor(max_depth=depth,random_state=42).fit(X_train, y_train)
        rf=RandomForestRegressor(max_depth=depth,random_state=42).fit(X_train, y_train)
        # Model evaluation Code here
        mlflow.sklearn.log_model(dt,"DecisionTree")
        mlflow.sklearn.log_model(lr,"LinReg")
        mlflow.sklearn.log_model(rf,"RandomForest")

        lr_y_pred= lr.predict(X_test)
        dt_y_pred= dt.predict(X_test)
        rf_y_pred= rf.predict(X_test)
        
        lin_reg_rmse=mean_squared_error(y_test, lr_y_pred,squared=False)
        tree_reg_rmse=mean_squared_error(y_test, dt_y_pred,squared=False)
        forest_reg_rmse=mean_squared_error(y_test, rf_y_pred,squared=False)
        
        # Log a metric
        mlflow.log_param("max_depth", depth)
        mlflow.log_metric("rmse_lin", lin_reg_rmse)
        mlflow.log_metric("rmse_tree", tree_reg_rmse)
        mlflow.log_metric("rmse_forest", forest_reg_rmse)

# 7.Search and save results
experiment_results = mlflow.search_runs()
experiment_results
