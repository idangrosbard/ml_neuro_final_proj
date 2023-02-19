import pandas as pd
import numpy as np


if __name__ == '__main__':
    df = pd.read_csv('unrestricted_hcp_freesurfer.csv', index='Subject')
    print(df)