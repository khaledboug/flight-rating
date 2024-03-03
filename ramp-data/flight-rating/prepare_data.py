import os
import pandas as pd
from sklearn.model_selection import train_test_split

rs = 57

# 1 - read in the data.
df = pd.read_csv(os.path.join('data', 'data.csv'))

# 2 - split into train/test subsets
df_train, df_test = train_test_split(df, test_size=0.2, random_state=rs)

# 3 - Split public train/test subsets. In this case the private training
# data will be used as the public data
df_public_train, df_public_test = train_test_split(
    df_train, test_size=0.2, random_state=rs)

# 4 - save the private data
df_train.to_csv(os.path.join('data', 'train.csv'),
                index=False)
df_test.to_csv(os.path.join('data', 'test.csv'),
               index=False)

# 5 - save the public data
df_public_train.to_csv(os.path.join('data', 'public', 'train.csv'),
                       index=False)
df_public_test.to_csv(os.path.join('data', 'public', 'test.csv'),
                      index=False)
