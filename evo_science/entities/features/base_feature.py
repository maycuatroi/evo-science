import numpy as np


class BaseFeature:
    virtual = False
    categories = None
    column_type = None
    column_name = None
    use_columns = None

    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.is_skip = False
        self.column_name = self.column_name or self.__class__.__name__

    @property
    def data(self):
        """Return the data for the feature. If test data is available, return the concatenated train and test data."""
        data = self.train_data
        if self.test_data is not None:
            data = np.concatenate((data, self.test_data), axis=0)
        return data

    def clean(self, df):
        return df

    def encode(self, x: np.array) -> np.array:
        if self.column_type == "category":
            return self.encode_category(x)
        return x.astype(self.column_type)

    def build(self, df):
        if self.virtual:
            df[self.column_name] = self.compute(df)

        df = self.clean(df)
        df[self.column_name] = self.fillna(df[self.column_name])
        df[self.column_name] = self.encode(df[self.column_name].values)

        return df

    def to_numpy(self, is_train=True):
        if is_train:
            return self.train_data
        else:
            return self.test_data

    def is_all_nan(self):
        """
        Check if the feature contains only NaN values.
        """
        return np.all(np.isnan(self.train_data))

    def encode_category(self, x: np.array):
        if isinstance(self.categories, list):
            return [self.categories[_x] for _x in x]
        if isinstance(self.categories, dict):
            return [self.categories[_x] for _x in x]
        if self.categories == "__auto__":
            all_categories = set(x)
            # build auto encoding
            all_categories = list(all_categories)
            all_categories.sort()
            np_x = np.zeros(len(x))
            for i, category in enumerate(all_categories):
                np_x[x == category] = i
            return np_x
        if self.categories == "__one_hot__":
            all_categories = x.unique()
            # build one hot encoding
            np_x = np.zeros((len(x), len(all_categories)))
            for i, category in enumerate(all_categories):
                np_x[:, i] = x == category
            return np_x
        raise ValueError(f"Unknown category type: {self.categories}")

    def fillna(self, x):
        is_na_values = x.isna()
        if is_na_values.any():
            raise ValueError(f"Feature {self.column_name} contains NaN values.")
        return x

    def compute(self, df):
        raise NotImplementedError("Virtual feature must implement compute method")
