import pandas as pd
import numpy as np
from math import sqrt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def prepDataFrame(filename):
    df_1m = pd.read_csv(filename)
    # Replace all strings with " NAN" to actual np.nan
    df_1m = df_1m.replace(' NAN', np.nan)

    # Cast all values as 64 bit integers
    df_1m['pc'] = df_1m['pc'].astype('int64') # This seems to work, but be careful
    df_1m['effective_address'] = df_1m['effective_address'].astype('int64') 
    df_1m['num_values'] = df_1m['num_values'].astype('int64')
    df_1m['val0'] = df_1m['val0'].astype('float64')

    # Get a dataframe with only the pc, effective_address, ad val0
    df = pd.DataFrame(index=df_1m.index)
    df['x0'] = df_1m['pc']
    df['x1'] = df_1m['effective_address']
    df['y']  = df_1m['val0']

    # Find the range of each field, use them to scale each field into (0,1)
    x0_range = df['x0'].max() - df['x0'].min()
    x1_range = df['x1'].max() - df['x1'].min()
    y_range = df['y'].max() - df['y'].min()

    # Scale each column into (0,1)
    df['x0'] = df['x0']/x0_range
    df['x1'] = df['x1']/x1_range
    df['y']  = df['y']/y_range
    return df.dropna()

# Functions for geting the data ready for the neural network.

def input_response_split(df):
    """ Splits the dataframe into inputs and response sets.
        Particularly, outputs X (ML inputs), and y (responses)
        X's cols are, repectively, the PC and Effective Address
        and its rows are training/testing examples. X \in R^{N * 2}
        y is a vector whose rows contain the groundtruth loaded
        valeu. y \in R^N
    """
    # Put the data into a matrix with each row corresponding
    # to an instruction and each column is a dataframe column.
    data = df.values

    # Break the data up into input and output.
    return data[:, :-1], data[:, -1]
    
def test_train_split(df, num_training_examples):
    """ Splits the dataframe into a test set and training set. Uses
        input_response_split to produce X, y then splits these into
        training and testing sets.
        usage: 
        X_tr, X_test, y_tr, y_test = test_train_split(df, num_training_examples)
    """
    assert(num_training_examples < len(df))
    X, y = input_response_split(df)
    X_tr = X[:num_training_examples,]
    X_te  = X[num_training_examples:,]
    y_tr = y[:num_training_examples]
    y_te  = y[num_training_examples:]
    return X_tr, X_te, y_tr, y_te 

# Build and return the network
def build_model(X_shape, dense_out1=50, dense_out2=50, l_rate=0.0001):
    model = Sequential()
    model.add(Dense(dense_out1,input_dim=2, kernel_initializer='normal', activation='relu')) # A dense layer with dense_out1 outputs
    model.add(Dense(dense_out2, kernel_initializer='normal', activation='relu')) # A dense layer with dense_out2 outputs
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))        # A dense layer with 1 output
    adam = Adam(learning_rate=l_rate)
    model.compile(optimizer=adam, loss='mse')
    return model

def get_rmse(pred, truth, n):
    rmse = 0
    for i in range(n):
        rmse += (truth[i] - pred[i][0])**2
    return sqrt(rmse)/n