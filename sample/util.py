import numpy as np
import pandas as pd

def split_and_scale(s):
    return np.array(s.split()).astype(np.int) / 255.0

def getData():
    df = pd.read_csv("../datasets/fer2013.csv.zip")
    df["pixels_separated"] = df.pixels.apply(lambda x:split_and_scale(x))
    # 偏りデータの対処
    df_1 = df[df.emotion == 1]
    df_else = df[~(df.emotion == 1)]
    df_1_concat = pd.concat([df_1]*10, ignore_index=True)
    df_balance = pd.concat([df_1_concat, df_else], ignore_index=True)
    # データ作成
    X = []
    y = []
    for lis, lb in zip(df_balance.pixels_separated, df_balance.emotion):
        X.append(lis.tolist())
        y.append(lb)
    X = np.array(X)
    y = np.array(y)
    train_idx = df_balance[df_balance.Usage == "Training"].index
    test_idx = df_balance[df_balance.Usage == "PublicTest"].index
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    del df
    del df_1
    del df_else
    del df_1_concat
    del df_balance
    del X
    del y
    del train_idx
    del test_idx
    return X_train, X_test, y_train, y_test
