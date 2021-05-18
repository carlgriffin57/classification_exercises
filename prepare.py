import pandas as pd
import numpy as np
import os
from env import host, username, password

def prep_iris(df):
    '''
        This function accepts the untransformed iris data and returns the data with tranformations applied.
    '''
    df = df.drop(columns=(['species_id', 'measurement_id']))
    df = df.rename(columns = {'species_name':'species'})
    dummy_df = pd.get_dummies(df[['species']], dummy_na = False, drop_first=[True])
    df = pd.concat([df, dummy_df], axis=1)
    return df