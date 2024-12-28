import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from matplotlib import pyplot
from matplotlib import style
from google.colab import drive

#Data
#Import from https://www.kaggle.com/datasets/imtkaggleteam/household-power-consumption/data

drive.mount('/content/gdrive')

URL = '/content/gdrive/My Drive/Power Vortex/pv-lstm/household_power_consumption.txt'
df = pd.read_csv(URL, sep=';', header=0, low_memory=False, na_values='?', parse_dates={'datetime':[0, 1]}, dayfirst=True)
for col in df.columns:
    if col != 'datetime':
        df[col] = pd.to_numeric(df[col], errors='coerce')

#Data Exploration
#Filled Missing Values

print("First 5 rows.","\n")
df.head()

df.info()

df.fillna(method='ffill', inplace=True)

values = df.values
df['Sub_metering_4'] = (values[:,1] * 1000 / 60) - (values[:,5] + values[:,6] + values[:,7])
df['Sub_metering_4'] = df['Sub_metering_4'].astype('float64')
df.to_csv('household_power_consumption.csv')

print("First 5 rows.","\n")
df.head()

print("Last 5 rows.","\n")
df.tail()

df.info()

print("Descriptive statistics", "\n")
print(df.describe(), "\n")

#Feature Extraction and Engineering

data_set = pd.read_csv('/content/household_power_consumption.csv')

del data_set["Global_reactive_power"]
del data_set["Voltage"]
del data_set["Global_intensity"]
del data_set["Sub_metering_1"]
del data_set["Sub_metering_2"]
del data_set["Sub_metering_3"]
del data_set["Sub_metering_4"]
data_set.rename(columns={"datetime":"DateTime","Global_active_power":"Consumption"},inplace=True)

data_set = data_set.drop(columns=['Unnamed: 0'])
print(data_set.head(5))

dataset = data_set
dataset["Month"] = pd.to_datetime(data_set["DateTime"]).dt.month
dataset["Year"] = pd.to_datetime(data_set["DateTime"]).dt.year
dataset["Date"] = pd.to_datetime(data_set["DateTime"]).dt.date
dataset["Time"] = pd.to_datetime(data_set["DateTime"]).dt.time
dataset["Week"] = pd.to_datetime(data_set["DateTime"]).dt.isocalendar().week
dataset["Day"] = pd.to_datetime(data_set["DateTime"]).dt.day_name()
dataset = data_set.set_index("DateTime")
dataset.index = pd.to_datetime(dataset.index)

dataset.head()

dataset.tail()

print("Total Number of Years: ", dataset.Year.nunique() )
print(dataset.Year.unique())

#Assuming week starts on Mondey and ends on Sunday.
#Monday 18-12-2006 to Sunday 26-12-2021
#Omit first 1837 and last 7021 rows
dataset = dataset[1836:-7023]
dataset.tail()

dataset.head()

#Data Visualization

style.use("ggplot")
plt.figure(figsize=(10, 6))
sns.lineplot(x='DateTime', y='Consumption', data=dataset)

plt.title('Consumption Over Time')
plt.xlabel("Date")
plt.ylabel("Energy in KWh")
plt.show()

style.use("ggplot")
fig, axes = plt.subplots(6, 1, figsize=(30, 30))
for i, year in enumerate(range(2006, 2011)):
    yearly_data = dataset[dataset['Year'] == year]
    axes[i].plot(yearly_data.index, yearly_data['Consumption'], color="blue", linewidth=1.7)
    axes[i].set_title(f'Energy Consumption {year}')
    if i == 5:
        axes[i].set_xlabel("Date")
    axes[i].set_ylabel("Energy in MW")
plt.tight_layout()
fig.suptitle('Energy Consumption Each Year', fontsize=16, y=1.02)
plt.show()

fig = plt.figure(figsize = (15,10))
sns.histplot(dataset["Consumption"], kde=True)
plt.title("Energy Consumption Distribution")
plt.xlabel("Energy Consumption")
plt.ylabel("Density")
plt.show()

fig = plt.figure(figsize=(15, 10))
ax = sns.boxplot(x="Month", y="Consumption", data=dataset)
plt.title("Energy Consumption VS Month")
plt.xlabel("Month")
plt.ylabel("Energy Consumption")
plt.grid(True, alpha=1)
plt.show()

fig = plt.figure(figsize=(15, 10))
ax = sns.boxplot(x=dataset.index.hour, y=dataset['Consumption'])
plt.title("Energy Consumption VS Hour")
plt.xlabel("Hour")
plt.ylabel("Energy Consumption")
plt.grid(True, alpha=1)
plt.show()

fig = plt.figure(figsize=(15, 10))
ax = sns.boxplot(x=dataset.index.year, y=dataset['Consumption'])
plt.title("Energy Consumption VS Year")
plt.xlabel("Year")
plt.ylabel("Energy Consumption")
plt.grid(True, alpha=1)
plt.show()



#LSTM MODEL

##Dataset Spliting

newDataSet = dataset[['Consumption']].resample("D").mean()
newDataSet['Month'] = newDataSet.index.month
newDataSet['Year'] = newDataSet.index.year
newDataSet['Week'] = newDataSet.index.isocalendar().week
newDataSet['Week'] = newDataSet['Week'].astype('float64')
newDataSet['Month'] = newDataSet['Month'].astype('float64')
newDataSet['Year'] = newDataSet['Year'].astype('float64')

print("df(First): ", df.shape)
print("data_set(2nd): ", data_set.shape)
print("Old Dataset: ", dataset.shape)
print("New Dataset: ", newDataSet.shape)

df.head()

data_set.head()

dataset.head()

newDataSet.head()

newDataSet.to_csv("newDataSet.csv")

from google.colab import files
files.download("newDataSet.csv")

y = newDataSet["Consumption"]
print(y[0])
y.shape

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))
y = scaler.fit_transform(np.array(y).reshape(-1,1))
print("Normalizing data before model fitting")
print(y[:10])

training_size = int(len(y)*0.80)
test_size = len(y)- training_size
val_size = int(training_size*0.20)
print("Train Size: ", training_size)
print("Test Size: ", test_size)
print("Validate Size: ", val_size)

train_data = y[0:training_size-val_size,:]
val_data = y[training_size-val_size:training_size,:]
test_data = y[training_size:len(y),:]

def create_dataset(dataset, time_step = 1):
  dataX, dataY = [] , []
  for i in range(len(dataset)-time_step-1):
    a = dataset[i:(i+time_step),0]
    dataX.append(a)
    dataY.append(dataset[i + time_step,0])
  return np.array(dataX), np.array(dataY)

time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)
X_val, y_val = create_dataset(val_data, time_step)

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1],1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1],1)

print("X_train shape: ", X_train.shape)
print("X_test shape: ",X_test.shape)
print("X_val shape: ",X_val.shape)

##Model Structure

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

model = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (time_step, 1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))

model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mean_squared_error'])

model.summary()

##Model Training

#from keras.callbacks import EarlyStopping, ModelCheckpoint
#checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')
#early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=20,
    validation_data=(X_val, y_val),
    #callbacks=[checkpoint, early_stop],
    verbose=1
)

import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

import tensorflow as tf
tf.__version__

train_predict=model.predict(X_train)
test_predict=model.predict(X_test)
val_predict=model.predict(X_val)

train_predict = scaler.inverse_transform(train_predict)
y_train_inv = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test_inv = scaler.inverse_transform([y_test])
val_predict = scaler.inverse_transform(val_predict)
y_val_inv = scaler.inverse_transform([y_val])

from sklearn.metrics import mean_squared_error, mean_absolute_error

train_rmse = np.sqrt(mean_squared_error(y_train_inv[0], train_predict[:,0]))
test_rmse = np.sqrt(mean_squared_error(y_test_inv[0], test_predict[:,0]))
val_rmse = np.sqrt(mean_squared_error(y_val_inv[0], val_predict[:,0]))

train_mae = mean_absolute_error(y_train_inv[0], train_predict[:,0])
test_mae = mean_absolute_error(y_test_inv[0], test_predict[:,0])
val_mae = mean_absolute_error(y_val_inv[0], val_predict[:,0])

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

train_mape = mean_absolute_percentage_error(y_train_inv[0], train_predict[:,0])
test_mape = mean_absolute_percentage_error(y_test_inv[0], test_predict[:,0])
val_mape = mean_absolute_percentage_error(y_val_inv[0], val_predict[:,0])

print(f"Train RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")
print(f"Validation RMSE: {val_rmse}\n")

print(f"Train MAE: {train_mae}")
print(f"Test MAE: {test_mae}")
print(f"Validation MAE: {val_mae}\n")

print(f"Train MAPE: {train_mape}")
print(f"Test MAPE: {test_mape}")
print(f"Validation MAPE: {val_mape}")

print(train_predict.shape)
print(test_predict.shape)
print(val_predict.shape)
print(train_predict[0])
print(test_predict[0])
print(val_predict[0])
print(y_train.shape)
print(y_test.shape)
print(y_val.shape)

train_predictions = model.predict(X_train)
train_predictions =scaler.inverse_transform(train_predictions)

y_train = y_train.reshape(y_train.shape[0], 1)
actual = scaler.inverse_transform(y_train)
train_results = pd.DataFrame()

train_results["Train Predictions"] = train_predictions.tolist()
train_results["Actuals"] = actual.tolist()

train_results

# Plotting train predictions
plt.figure(figsize=(15,6))
plt.plot(y_train_inv[0], label='Actual')
plt.plot(train_predict[:,0], label='Predicted')
plt.title('Train Data - Actual vs Predicted')
plt.legend()
plt.show()

test_predictions = model.predict(X_test)
test_predictions =scaler.inverse_transform(test_predictions)

actual_test = y_test.reshape(-1, 1)
actual_test = scaler.inverse_transform(actual_test)

test_results = pd.DataFrame()
test_results["test Predictions"] = test_predictions.tolist()
test_results["Actuals_test"] = actual_test.tolist()

test_results

# Plotting test predictions
plt.figure(figsize=(15,6))
plt.plot(y_test_inv[0], label='Actual')
plt.plot(test_predict[:,0], label='Predicted')
plt.title('Test Data - Actual vs Predicted')
plt.legend()
plt.show()

val_predictions = model.predict(X_val)
val_predictions =scaler.inverse_transform(val_predictions)

actual_val = y_val.reshape(-1, 1)
actual_val = scaler.inverse_transform(actual_val)

val_results = pd.DataFrame()
val_results["Val Predictions"] = val_predictions.tolist()
val_results["Actuals_val"] = actual_val.tolist()

val_results

# Plotting validation predictions
plt.figure(figsize=(15,6))
plt.plot(y_val_inv[0], label='Actual')
plt.plot(val_predict[:,0], label='Predicted')
plt.title('Validation Data - Actual vs Predicted')
plt.legend()
plt.show()

y = np.array(y).reshape(-1,1)
look_back = 100
trainPredictPlot = np.empty_like(y)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
testPredictPlot = np.empty_like(y)
testPredictPlot[:, :] = np.nan
test_start_point = len(train_predict) + (look_back * 2)
testPredictPlot[test_start_point:test_start_point + len(test_predict), :] = test_predict

plt.figure(figsize=(15, 6))
plt.plot(scaler.inverse_transform(y), label='Actual Consumption')
plt.plot(trainPredictPlot, label='Train Predictions')
plt.plot(testPredictPlot, label='Test Predictions')
plt.legend()
plt.xlabel('Time Steps')
plt.ylabel('Consumption KWh')
plt.show()

#Future Forecasting

n_future = 30
n_steps = 100

last_batch = y[-n_steps:].reshape(-1, 1)

temp_input = last_batch.flatten().tolist()

lst_output = []

# Predict the future values
for i in range(n_future):
    x_input_reshaped = np.array(temp_input).reshape(1, n_steps, 1)
    yhat = model.predict(x_input_reshaped, verbose=0)
    lst_output.append(yhat[0][0])
    temp_input.append(yhat[0][0])
    temp_input = temp_input[1:]

lst_output_unscaled = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1)).flatten()

lst_output

lst_output_unscaled

n_past = 100
final_output = lst_output_unscaled
actual = scaler.inverse_transform(y[-n_past:]).flatten()

time_steps = list(range(-(n_past), 0))
future_steps = list(range(0, n_future))

plt.figure(figsize=(15, 6))
plt.plot(time_steps, actual, label='Actual Data', color='blue')
plt.plot(future_steps, final_output, label='Predicted Data', color='red')

plt.title('Future Prediction vs Actual Data')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.show()