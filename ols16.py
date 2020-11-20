import pandas as pd
import numpy as np
from math import sqrt

def prepDataFrame(filename):
    """ Reads in data from a csv file (given by filename) and
        prepares a dataframe for OLS. The dataframe columns are:
        |PC | EffAddr | 16 cols for previous 16 values | loaded value 
    """
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
    
    # Add columns that contain the 16 previously seen values
    for i in range(16):
        columnName = 'x' + str(2+i)
        df[columnName] = df['y'].shift(i+1)
    
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

def solver(X_tr, y_tr):
    """ Takes a matrix A and vector y and finds x that minimizes |Ax - y|. 
        x is returned.
    """
    # Set up the equation above as lhs*sol = rhs
    XT  = np.transpose(X_tr)
    lhs = np.matmul(XT, X_tr)
    rhs = np.matmul(XT, y_tr)
    return np.linalg.solve(lhs, rhs)

def getError(sol, X_te, y_te):
    # Make predictions using sol
    # First, quickly name the columns
    cols = []
    for i in range(18):
        cols += ['x' + str(i)]
    results = pd.DataFrame(X_te, columns=cols)

    # Quckly build the predictions by taking the dot product
    # of each row with the sol vector
    results['pred'] = 0
    for i in range(18):
            results['pred'] += sol[i]*results["x" + str(i)]
    results['pred'] += results['pred']
    results['truth'] = pd.Series(y_te)
    return sqrt(((results['truth'] - results['pred'])**2).sum())/len(y_te)