# Part 1 - Data Preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values ## return dataframe matrix training_set with all rows & column 1

# Feature Scaling to scale stock prices between (0,1)
from sklearn.preprocessing import MinMaxScaler 
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure(list) with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0]) # rows from [i-60 to i) & column 0
    y_train.append(training_set_scaled[i, 0])      # row value i
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping (since each layers output is the input of the next layer, their shape must be equal)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - Building the RNN
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN (sequential because the output of one layer is input of other layer)
# sequential is just a linear stack of layers
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
# number of LSTM neurons in layer=50 
# dropout=0.2 (means 20% of 50 neurons will be ignored randomly during each iteration of training)
# return_seq=True(as output of this layer is input for the next layer)
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
# return_seq=False(Default)(as we will not add more LSTM layer)
regressor.add(Dense(units = 1))

# Compiling the RNN
# optimizer used to change the attributes of neural network to reduce loss (like weight,learning rate)
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
# One Epoch=ENTIRE dataset is passed forward and backward through the neural network ONCE.
# 1 epoch=underfit, many epoch=overfit
# 32 number of training examples present in a single batch.
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

# Part 3 - Making the predictions and visualising the results
# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values # all rows and column 1

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0) ## along rows add both data
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80): # test data with 20 rows each row has 60 columns for consecutive 60 days stock price
    X_test.append(inputs[i-60:i, 0]) #rows from [i-60,i) and column 0
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
