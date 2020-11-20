import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from math import sqrt

def prepDataFrame(filename):
    df_1m = pd.read_csv(filename, error_bad_lines=False)
    # Replace all strings with " NAN" to actual np.nan
    df_1m = df_1m.replace(' NAN', np.nan)
    df_1m = df_1m.replace('NAN', np.nan)

    # pc_fst32 pc_snd32 eff_addr_fst32 eff_addr_snd32 num_values 
    # val0_fst32 val0_snd32

    # Cast all values as 64 bit integers
    df_1m['pc_fst32'] = df_1m['pc_fst32'].astype('int64')
    df_1m['pc_snd32'] = df_1m['pc_snd32'].astype('int64')
    df_1m['eff_addr_fst32'] = df_1m['eff_addr_fst32'].astype('int64')
    df_1m['eff_addr_snd32'] = df_1m['eff_addr_snd32'].astype('int64')
    df_1m['num_values'] = df_1m['num_values'].astype('int64')
    df_1m['val0_fst32'] = df_1m['val0_fst32'].astype('float64')
    df_1m['val0_snd32'] = df_1m['val0_snd32'].astype('float64')

    # Prediction scheme:
        # Break up the pc, eff_addr, and val0 into high and low 32 bits
        # (pc_h, pc_l, eff_addr_h, eff_addr_l) |----> (val0_h, val0_l)
        # This is accomplished by 2 'sub-networks'
        # One for predicting val0_h
        # Another for val0_l

    # Get a dataframe for the subnetwork that predicts the higher bits
    dflo = pd.DataFrame(index=df_1m.index)
    dflo['x0'] = df_1m['pc_fst32']
    dflo['x1'] = df_1m['pc_snd32']
    dflo['x2'] = df_1m['eff_addr_fst32']
    dflo['x3'] = df_1m['eff_addr_snd32']
    dflo['y0'] = df_1m['val0_fst32']

    # Get a dataframe for the subnetwork that predicts the lower bits
    dfhi = pd.DataFrame(index=df_1m.index)
    dfhi['x0'] = df_1m['pc_fst32']
    dfhi['x1'] = df_1m['pc_snd32']
    dfhi['x2'] = df_1m['eff_addr_fst32']
    dfhi['x3'] = df_1m['eff_addr_snd32']
    dfhi['y1'] = df_1m['val0_snd32']

    # Find the range of each field, use them to scale each field into (0,1)
    x0_range = dflo['x0'].max() - dflo['x0'].min()
    x1_range = dflo['x1'].max() - dflo['x1'].min()
    x2_range = dflo['x2'].max() - dflo['x2'].min()
    x3_range = dflo['x3'].max() - dflo['x3'].min()

    y0_range = dflo['y0'].max() - dflo['y0'].min()
    y1_range = dfhi['y1'].max() - dfhi['y1'].min()


    # Scale each column into (0,1)
    if x0_range != 0:
        dflo['x0'] = dflo['x0']/x0_range
        dfhi['x0'] = dfhi['x0']/x0_range
    if x1_range != 0:
        dflo['x1'] = dflo['x1']/x1_range
        dfhi['x1'] = dfhi['x1']/x1_range
    if x2_range != 0:
        dflo['x2'] = dflo['x2']/x2_range
        dfhi['x2'] = dfhi['x2']/x2_range
    if x3_range != 0:
        dflo['x3'] = dflo['x3']/x3_range
        dfhi['x3'] = dfhi['x3']/x3_range

    if y0_range != 0:
        dflo['y0'] = dflo['y0']/y0_range
    if y1_range != 0:
        dfhi['y1'] = dfhi['y1']/y1_range
        
    return dflo.dropna(), dfhi.dropna(), y0_range, y1_range

# Functions for geting the data ready for the neural network.

def input_response_split(df):
    """ Splits the dataframe into inputs and response sets.
        Particularly, outputs X (ML inputs), and y (responses)
        X's cols are the values used to predict
        and its rows are training/testing examples. X \in R^{N x 4}
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
    model.add(Dense(dense_out1,input_dim=X_shape[1], kernel_initializer='normal', activation='relu')) # A dense layer with dense_out1 outputs
    model.add(Dense(dense_out2, kernel_initializer='normal', activation='relu')) # A dense layer with dense_out2 outputs
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))        # A dense layer with 1 output
    adam = Adam(learning_rate=l_rate)
    model.compile(optimizer=adam, loss='mse')
    return model