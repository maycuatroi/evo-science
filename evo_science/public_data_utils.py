import pandas as pd


def get_titanic_data():
    url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
    return pd.read_csv(url)
