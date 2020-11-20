import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from math import sqrt

def prepDataFrame(filename): 
    df_1m = pd.read_csv(filename)
    # Replace all strings with " NAN" to actual np.nan
    df_1m = df_1m.replace(' NAN', np.nan)

    # Cast all values as 64 bit integers
    df_1m['pc'] = df_1m['pc'].astype('int64') # This seems to work, but be careful
    df_1m['effective_address'] = df_1m['effective_address'].astype('int64') 
    df_1m['num_values'] = df_1m['num_values'].astype('int64')
    df_1m['val0'] = df_1m['val0'].astype('float64')

    """ Initially we shal only try to predict val0 given a pc and
        effective_address.
    """
    # Get a dataframe with only the pc, effective_address, ad val0
    df = pd.DataFrame(index=df_1m.index)
    # Use only the effective address, in LVP paper this was a better
    # hash than pc into their LVP table, so lets go with that here.
    df['x'] = df_1m['effective_address']
    df['y'] = df_1m['val0']

    # Find the range of each field, use them to scale each field into (0,1)
    x_range = df['x'].max() - df['x'].min()
    y_range = df['y'].max() - df['y'].min()

    # Scale each column into (0,1)
    df['x'] = df['x']/x_range
    df['y'] = df['y']/y_range
    
    return df.dropna()

def split_df(df, step):
    """ Split a dataframe into training example, ground truth pairs.
        Input: df is the dataframe to split which has 2 columns, the
                 first is for the effective address, second for value.
               step is the number of effective addrs to use in each
                 training example.
        The training examples contain step # of eff_addrs as the input
        tuple and the value associated with the last eff_addr in the
        input tuple as the ground truth.
        Example pair:
        [x_1, x_2, ..., x_step] |----> y_step
        where x is the column corresponding to the eff_addrs and y
        corresponds to the values.
    """
    arr  = df.to_numpy()
    rows = arr.shape[0]
    EFFADDR_IDX = 0
    VALUE_IDX   = 1
    X, y_hat = [], []
    for i in range(rows):
        end = i + step
        if end > rows-1:
            break
        x, y = arr[i:end, EFFADDR_IDX], arr[end-1, VALUE_IDX]
        X.append(x)
        y_hat.append(y)
    return np.array(X), np.array(y_hat)

def test_train_split(df, num_training_examples, step):
    """ Splits the dataframe into a test set and training set. Uses
        input_response_split to produce X, y then splits these into
        training and testing sets.
        usage: 
        X_tr, X_test, y_tr, y_test = test_train_split(df, num_training_examples)
    """
    assert(num_training_examples < len(df))
    X, y = split_df(df, step)
    X_tr = X[:num_training_examples,]
    X_te  = X[num_training_examples:,]
    y_tr = y[:num_training_examples]
    y_te  = y[num_training_examples:]
    return X_tr, X_te, y_tr, y_te 

def build_model(step=16, filters=20, kernel_size=4, pool_size=4, dense_out1=50, dense_out2=50):
    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(step, 1)))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Flatten())
    model.add(Dense(dense_out1, activation='relu'))
    model.add(Dense(dense_out2, kernel_initializer='normal', activation='relu')) # A dense layer with dense_out2 outputs
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))        # A dense layer with 1 output
    adam = Adam(learning_rate=0.001)
    model.compile(optimizer=adam, loss='mse')
    return model

def get_rmse(pred, truth, n):
    rmse = 0
    for i in range(n):
        rmse += (truth[i] - pred[i][0])**2
    return sqrt(rmse)/n