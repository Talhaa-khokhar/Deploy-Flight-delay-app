#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.compose import ColumnTransformer


from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing


# # Understand the Data:

# In[2]:


data = pd.read_csv('DelayedFlights.csv')


# In[3]:


data


# In[4]:


rows, cols = data.shape
print("The dataset contains",rows,"rows and",cols,"columns" )


# In[5]:


data.info()


# In[6]:


data.head()


# In[7]:


data.tail(10)


# In[8]:


data.size


# In[9]:


data.columns


# # Data Cleaning

# In[10]:


data.duplicated().sum()


# In[11]:


data.nunique()


# In[12]:


# Check for missing values in each column
missing_values = data.isnull().sum()
print(missing_values)


# In[13]:


len(data.columns)


# In[14]:


data.describe()


# # Preprocessing

# In[15]:


# Handle missing values
imputer = SimpleImputer(strategy='median')
columns_to_impute = ['ArrTime', 'CRSArrTime', 'ActualElapsedTime', 'CRSElapsedTime', 'AirTime', 'ArrDelay', 'TaxiIn', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay']
data[columns_to_impute] = imputer.fit_transform(data[columns_to_impute])


# In[16]:


# Remove duplicates
data = data.drop_duplicates()


# In[17]:


def remove_outliers(data, columns):
    for col in columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        data = data[~((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR)))]
    return data

numerical_columns = ['DepTime', 'CRSDepTime', 'ArrTime', 'CRSArrTime', 'ActualElapsedTime', 'CRSElapsedTime', 'AirTime', 'ArrDelay', 'TaxiIn', 'TaxiOut', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay']
data = remove_outliers(data, numerical_columns)


# In[18]:


data


# In[19]:


rows, cols = data.shape
print("The dataset contains",rows,"rows and",cols,"columns" )


# In[20]:


data.info()


# In[21]:


data.size


# In[22]:


data.duplicated().sum()


# In[23]:


# Check for missing values in each column
missing_values = data.isnull().sum()
print(missing_values)


# # Data Summeries

# In[24]:


# Basic statistics for numerical columns
Descriptive_statistics = data.describe()
print(Descriptive_statistics)


# # Frequency Tables

# In[25]:


carrier_counts = data['UniqueCarrier'].value_counts()
print(carrier_counts)


# In[26]:


CancellationCode_counts = data['CancellationCode'].value_counts()
print(CancellationCode_counts)


# In[27]:


TailNum_counts = data['TailNum'].value_counts()
print(TailNum_counts)


# In[28]:


Origin_counts = data['Origin'].value_counts()
print(Origin_counts)


# In[29]:


Dest_counts = data['Dest'].value_counts()
print(Dest_counts)


# # Bar Plot

# In[30]:


sns.set(style="ticks")  # Set a plot style
plt.figure(figsize=(14, 7))  # Set the figure size
sns.barplot(x=carrier_counts.index, y=carrier_counts.values, palette="viridis")  # Create the bar plot
plt.title("Distribution of UniqueCarrier")
plt.xlabel("UniqueCarrier")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()


# In[31]:


sns.set(style="ticks")  # Set a plot style
plt.figure(figsize=(14, 7))  # Set the figure size
sns.barplot(x=CancellationCode_counts.index, y=CancellationCode_counts.values, palette="viridis")  # Create the bar plot
plt.title("Distribution of CancellationCode")
plt.xlabel("CancellationCode")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()


# In[32]:


sns.set(style="ticks")  # Set a plot style
plt.figure(figsize=(14, 7))  # Set the figure size
sns.barplot(x=TailNum_counts.index, y=TailNum_counts.values, palette="viridis")  # Create the bar plot
plt.title("Distribution of TailNum")
plt.xlabel("TailNum")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()


# In[33]:


sns.set(style="ticks")  # Set a plot style
plt.figure(figsize=(14, 7))  # Set the figure size
sns.barplot(x=Origin_counts.index, y=Origin_counts.values, palette="viridis")  # Create the bar plot
plt.title("Distribution of Origin")
plt.xlabel("Origin")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()


# In[34]:


sns.set(style="ticks")  # Set a plot style
plt.figure(figsize=(14, 7))  # Set the figure size
sns.barplot(x=Dest_counts.index, y=Dest_counts.values, palette="viridis")  # Create the bar plot
plt.title("Distribution of Dest")
plt.xlabel("Dest")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()


# In[35]:


numeric_cols = data.select_dtypes(include=[int, float]).columns


# # Hist Plot

# In[36]:


#histograms
for col in numeric_cols:
    plt.figure(figsize=(10, 12))
    sns.histplot(data[col], kde=True)
    plt.title(f'Histogram of {col}')
    plt.show()


# # Time Analysis

# In[37]:


flights_by_year = data.groupby('Year').size()


# In[38]:


plt.figure(figsize=(8, 6))
plt.stackplot(flights_by_year.index, flights_by_year.values)
plt.title('Flights by Year')
plt.xlabel('Year')
plt.ylabel('Number of Flights')
plt.show()


# In[39]:


flights_by_month = data.groupby('Month').size()


# In[40]:


sns.set(style="ticks")
plt.figure(figsize=(8, 6))
sns.barplot(x=flights_by_month.index, y=flights_by_month.values, palette="viridis")
plt.title('Flights by Month')
plt.xlabel('Month')
plt.ylabel('Number of Flights')
plt.show()


# In[41]:


flights_by_day_of_month = data.groupby('DayofMonth').size()


# In[42]:


sns.set(style="ticks")
plt.figure(figsize=(8, 6))
sns.barplot(x=flights_by_day_of_month.index, y=flights_by_day_of_month.values, palette="viridis")
plt.title('Flights by Day of the Month')
plt.xlabel('Day of the Month')
plt.ylabel('Number of Flights')
plt.show()


# In[43]:


flights_by_day_of_week = data.groupby('DayOfWeek').size()


# In[44]:


sns.set(style="ticks")
plt.figure(figsize=(8, 6))
sns.barplot(x=flights_by_day_of_week.index, y=flights_by_day_of_week.values, palette="viridis")
plt.title('Flights by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Flights')
plt.show()


# # Flight Delays

# In[45]:


plt.figure(figsize=(12, 5))
sns.histplot(data['ArrDelay'], kde=True, color='g', label='Arrival Delay')
sns.histplot(data['DepDelay'], kde=True, color='r', label='Departure Delay')
plt.xlabel('Delay (minutes)')
plt.ylabel('Frequency')
plt.title('Distribution of Flight Delays')
plt.legend()

# Analyze the impact of delay reasons
delay_reasons = data[['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']]
delay_reasons_total = delay_reasons.sum()


# # Carrier Analysis

# In[46]:


# Calculate on-time performance metrics for carriers
on_time_threshold = 15  # Define a threshold for on-time performance (e.g., 15 minutes)
on_time_arrival = (data['ArrDelay'] <= on_time_threshold).groupby(data['UniqueCarrier']).mean()
on_time_departure = (data['DepDelay'] <= on_time_threshold).groupby(data['UniqueCarrier']).mean()

# Create bar charts to compare carriers
plt.figure(figsize=(10, 6))
sns.barplot(x=on_time_arrival.index, y=on_time_arrival.values, palette="Blues_d", label="Arrival On-Time Rate")
sns.barplot(x=on_time_departure.index, y=on_time_departure.values, palette="Reds_d", label="Departure On-Time Rate")
plt.xlabel('Carrier')
plt.ylabel('On-Time Rate')
plt.title('Carrier On-Time Performance')
plt.legend()
plt.xticks(rotation=45)


# # Raute Analysis

# In[47]:


common_routes = data.groupby(['Origin', 'Dest']).size().sort_values(ascending=False).head(5)
print(f"Common routes are: {common_routes}")


# In[48]:


# Assuming common_routes is a Series with MultiIndex (origin, destination)
# Get the top 5 most common routes
top_routes = common_routes.head(5).reset_index()
top_routes.columns = ['origin', 'destination', 'count']

# Create a new column for the route
top_routes['route'] = top_routes.apply(lambda x: f"{x['origin']} to {x['destination']}", axis=1)

# Plot the count plot
plt.figure(figsize=(10, 6))
sns.barplot(x='route', y='count', data=top_routes, palette="viridis")
plt.xlabel('Route')
plt.ylabel('Frequency')
plt.title('Most 5 Common Routes')
plt.xticks(rotation=45)
plt.show()


# In[49]:


# Identify routes with the highest and lowest average delays
average_delays_by_route = data.groupby(['Origin', 'Dest'])['ArrDelay'].mean().sort_values(ascending=False).head(5)
plt.figure(figsize=(10, 6))
sns.barplot(x=average_delays_by_route.index.map(lambda x: f"{x[0]} to {x[1]}"), y=average_delays_by_route.values,palette="viridis")
plt.xlabel('Route')
plt.ylabel('Average Delay (minutes)')
plt.title('Average Delays by Route')
plt.xticks(rotation=45)
plt.show()


# # Cancellation Analysis

# In[50]:


# Explore the reasons for flight cancellations
cancellation_reasons = data['CancellationCode'].value_counts()
print(cancellation_reasons)


# In[51]:


# Calculate the percentage of canceled flights
canceled_flights_percentage = (data['Cancelled'] == 1).mean() * 100
print(f"Percentage of canceled flights: {canceled_flights_percentage:.2f}%")


# # Percentage of delays & flights

# In[52]:


# Calculate the percentage of delayed flights
delayed_flights_percentage = (data['ArrDelay'] > 0).mean() * 100
print(f"Percentage of delayed flights: {delayed_flights_percentage:.2f}%")

# Calculate the percentage of each delay reason
carrier_delay_percentage = (data['CarrierDelay'] > 0).mean() * 100
weather_delay_percentage = (data['WeatherDelay'] > 0).mean() * 100
nas_delay_percentage = (data['NASDelay'] > 0).mean() * 100
security_delay_percentage = (data['SecurityDelay'] > 0).mean() * 100
late_aircraft_delay_percentage = (data['LateAircraftDelay'] > 0).mean() * 100

print(f"Percentage of carrier delays: {carrier_delay_percentage:.2f}%")
print(f"Percentage of weather delays: {weather_delay_percentage:.2f}%")
print(f"Percentage of NAS delays: {nas_delay_percentage:.2f}%")
print(f"Percentage of security delays: {security_delay_percentage:.2f}%")
print(f"Percentage of late aircraft delays: {late_aircraft_delay_percentage:.2f}%")


# # Correlation

# In[53]:


# Select numeric columns
numeric_data = data.select_dtypes(include=np.number)

# Calculate the correlation matrix for numeric columns
correlation_matrix = numeric_data.corr()

# Create a heatmap to visualize correlations
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, cmap='viridis')
plt.title('Correlation Matrix')
plt.show()


# # Visualizations

# In[54]:


# Create a histogram for Arrival Delay
plt.figure(figsize=(8, 6))
sns.histplot(data['ArrDelay'], bins=50, kde=True)
plt.title('Distribution of Arrival Delay')
plt.xlabel('Arrival Delay (minutes)')
plt.ylabel('Frequency')


# In[55]:


# Create a histogram for Departure Delay
plt.figure(figsize=(8, 6))
sns.histplot(data['DepDelay'], bins=50, kde=True)
plt.title('Distribution of Departure Delay')
plt.xlabel('Departure Delay (minutes)')
plt.ylabel('Frequency')


# In[56]:


# Create a scatterplot to explore the relationship between Departure Delay and Arrival Delay
plt.figure(figsize=(8, 6))
sns.scatterplot(x='DepDelay', y='ArrDelay', data=data)
plt.title('Departure Delay vs. Arrival Delay')
plt.xlabel('Departure Delay (minutes)')
plt.ylabel('Arrival Delay (minutes)')


# In[57]:


# Create a box plot for Arrival Delay by UniqueCarrier
plt.figure(figsize=(12, 6))
sns.boxplot(x='UniqueCarrier', y='ArrDelay', data=data)
plt.title('Arrival Delay by Carrier')
plt.xlabel('Carrier')
plt.ylabel('Arrival Delay (minutes)')


# In[58]:


# Create a box plot for Departure Delay by UniqueCarrier
plt.figure(figsize=(12, 6))
sns.boxplot(x='UniqueCarrier', y='DepDelay', data=data)
plt.title('Departure Delay by Carrier')
plt.xlabel('Carrier')
plt.ylabel('Departure Delay (minutes)')


# # Key features that impact flight delays

# In[59]:


# Group by weather delay and calculate average delay
weather_delay = data.groupby('WeatherDelay')['ArrDelay'].mean()
print(weather_delay)

# Plot the results
import matplotlib.pyplot as plt
weather_delay.plot(kind='bar')
plt.title('Average Delay by Weather Delay')
plt.xlabel('Weather Delay')
plt.ylabel('Average Delay (minutes)')
plt.show()


# In[60]:


# Group by security delay and calculate average delay
security_delay = data.groupby('SecurityDelay')['ArrDelay'].mean()
print(security_delay)

# Plot the results
security_delay.plot(kind='bar')
plt.title('Average Delay by Security-related Delays')
plt.xlabel('Security-related Delays')
plt.ylabel('Average Delay (minutes)')
plt.show()


# In[61]:


# Group by late aircraft delay and calculate average delay
late_aircraft_delay = data.groupby('LateAircraftDelay')['ArrDelay'].mean()
print(late_aircraft_delay)

# Plot the results
late_aircraft_delay.plot(kind='bar')
plt.title('Average Delay by Late Aircraft Issues')
plt.xlabel('Late Aircraft Issues')
plt.ylabel('Average Delay (minutes)')
plt.show()


# In[62]:


# Group by carrier and calculate cancellation rate
carrier_cancellations = data.groupby('UniqueCarrier')['Cancelled'].mean()
print(carrier_cancellations)

# Plot the results
carrier_cancellations.plot(kind='bar')
plt.title('Cancellation Rate by Carrier')
plt.xlabel('Carrier')
plt.ylabel('Cancellation Rate')
plt.show()


# # Label Encoding

# In[63]:


from sklearn.preprocessing import LabelEncoder

categorical_columns = ['UniqueCarrier', 'TailNum', 'Origin', 'Dest', 'CancellationCode']

for col in categorical_columns:
    data[col] = data[col].astype(str)

label_encoders = {col: LabelEncoder() for col in categorical_columns}

for col in categorical_columns:
    data[col] = label_encoders[col].fit_transform(data[col])


# # Normallization

# In[64]:


from sklearn.preprocessing import MinMaxScaler

numerical_columns = ['DepTime', 'CRSDepTime', 'ArrTime', 'CRSArrTime', 'ActualElapsedTime', 
                     'CRSElapsedTime', 'AirTime', 'ArrDelay', 'DepDelay', 'Distance', 
                     'TaxiIn', 'TaxiOut', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 
                     'SecurityDelay', 'LateAircraftDelay']

scaler = MinMaxScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])


# # Model Development & Evaluation

# # LinearRegression

# In[65]:


import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

columns_to_impute = [
    'ArrTime', 'CRSArrTime', 'ActualElapsedTime', 'CRSElapsedTime', 'AirTime', 
    'ArrDelay', 'TaxiIn', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay'
]
imputer = SimpleImputer(strategy='median')
data[columns_to_impute] = imputer.fit_transform(data[columns_to_impute])
data.replace([np.nan, np.inf, -np.inf], 0, inplace=True)

X = data.drop('ArrDelay', axis=1)
y = data['ArrDelay']

numerical_columns = [col for col in X.columns if X[col].dtype != 'object']
X = X[numerical_columns]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred = linear_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean Squared Error:', mse)
print('R2 Score:', r2)


# # DecisionTreeRegressor

# In[66]:


import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

columns_to_impute = [
    'ArrTime', 'CRSArrTime', 'ActualElapsedTime', 'CRSElapsedTime', 'AirTime', 
    'ArrDelay', 'TaxiIn', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay'
]
imputer = SimpleImputer(strategy='median')
data[columns_to_impute] = imputer.fit_transform(data[columns_to_impute])

data.replace([np.nan, np.inf, -np.inf], 0, inplace=True)

X = data.drop('ArrDelay', axis=1)
y = data['ArrDelay']

numerical_columns = [col for col in X.columns if X[col].dtype != 'object']
X = X[numerical_columns]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)
y_pred = tree_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean Squared Error:', mse)
print('R2 Score:', r2)


# # RandomForestRegressor

# In[67]:


import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

columns_to_impute = [
    'ArrTime', 'CRSArrTime', 'ActualElapsedTime', 'CRSElapsedTime', 'AirTime', 
    'ArrDelay', 'TaxiIn', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay'
]
imputer = SimpleImputer(strategy='median')
data[columns_to_impute] = imputer.fit_transform(data[columns_to_impute])

data.replace([np.nan, np.inf, -np.inf], 0, inplace=True)

X = data.drop('ArrDelay', axis=1)
y = data['ArrDelay']

numerical_columns = [col for col in X.columns if X[col].dtype != 'object']
X = X[numerical_columns]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
forest_model.fit(X_train, y_train)

y_pred = forest_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean Squared Error:', mse)
print('R2 Score:', r2)


# # GradientBoostingRegressor

# In[68]:


import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

columns_to_impute = [
    'ArrTime', 'CRSArrTime', 'ActualElapsedTime', 'CRSElapsedTime', 'AirTime', 
    'ArrDelay', 'TaxiIn', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay'
]
imputer = SimpleImputer(strategy='median')
data[columns_to_impute] = imputer.fit_transform(data[columns_to_impute])

data.replace([np.nan, np.inf, -np.inf], 0, inplace=True)

X = data.drop('ArrDelay', axis=1)
y = data['ArrDelay']
numerical_columns = [col for col in X.columns if X[col].dtype != 'object']
X = X[numerical_columns]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)

y_pred = gb_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean Squared Error:', mse)
print('R2 Score:', r2)


# # Compare the performance

# In[69]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

linear_model = LinearRegression()
tree_model = DecisionTreeRegressor(random_state=42)
forest_model = RandomForestRegressor(random_state=42)
gbm_model = GradientBoostingRegressor(random_state=42)

# List of models for iteration
models = [linear_model, tree_model, forest_model, gbm_model]
model_names = ['Linear Regression', 'Decision Trees', 'Random Forests', 'Gradient Boosting Machines']

for model, name in zip(models, model_names):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model: {name}")
    print(f"Mean Squared Error: {mse}")
    print(f"R² Score: {r2}")
    print()

# Compare performances based on MSE and R² Score
# Select the model with the lowest MSE and highest R² Score for deployment


# # Visually: Compare the performance

# In[70]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Initialize models
linear_model = LinearRegression()
tree_model = DecisionTreeRegressor(random_state=42)
forest_model = RandomForestRegressor(random_state=42)
gbm_model = GradientBoostingRegressor(random_state=42)

# List of models for iteration
models = [linear_model, tree_model, forest_model, gbm_model]
model_names = ['Linear Regression', 'Decision Trees', 'Random Forests', 'Gradient Boosting Machines']

# Train and evaluate each model
mse_scores = []
r2_scores = []

for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mse_scores.append(mse)
    r2_scores.append(r2)

# Plotting MSE
plt.figure(figsize=(10, 6))
plt.bar(model_names, mse_scores, color='skyblue')
plt.xlabel('Models')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Comparison of Mean Squared Error (MSE) among Models')
plt.ylim(0, max(mse_scores) * 1.1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plotting R2 Score
plt.figure(figsize=(10, 6))
plt.bar(model_names, r2_scores, color='lightgreen')
plt.xlabel('Models')
plt.ylabel('R² Score')
plt.title('Comparison of R² Score among Models')
plt.ylim(min(r2_scores) * 0.9, 1.0)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[73]:


import pickle

# Assuming 'forest_model' is your trained model
filename = 'model1.pkl'
pickle.dump(forest_model, open(filename, 'wb'))


# In[74]:


import pickle

# Assuming 'forest_model' is your trained model
filename = 'model_implement.pkl'
pickle.dump(forest_model, open(filename, 'wb'))


# In[75]:


joblib.dump(model, 'model_implement')


# In[76]:


from sklearn import __version__ as sklearn_version
print(f"scikit-learn version used for model: {sklearn_version}")


# In[ ]:




