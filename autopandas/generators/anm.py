# Additive Noise Model

# Imports
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

class ANM():
    def __init__(self, model=None, **kwargs):
        """ Data generator using multiple imputations with random forest
        """
        # List of Random Forests
        self.models = []
        # Random forest from sklearn
        # TODO custom models
        self.regressor = RandomForestRegressor
        self.classifier = RandomForestClassifier
        # Store data to be able to sample from original data
        self.data = None

    def fit(self, data, **kwargs):
        """ Fit one random forest for each column, given the others.

            Use kwargs to define model's (Random Forest) parameters.
        """
        self.data = data
        for i in range(len(data.columns)):
            # May bug with duplicate names in columns
            y = data[data.columns[i]]
            X = data.drop(data.columns[i], axis=1)
            # Regressor or classifier
            if data.columns[i] in data.indexes['numerical']:
                model = self.regressor(**kwargs)
            else:
                model = self.classifier(**kwargs)
            model.fit(X, y)
            self.models.append(model)

    def partial_fit_generate(self, n=1, p=0.8, replace=True, **kwargs):
        """ Fit and generate for high dimensional case.
            To avoid memory error, features are trained and generated one by one.

            :param p: The probability of changing a value
                        if p=0, the generated dataset will be equals to the original
                        if p=1, the generated dataset will contains only new values
            :param kwargs: Random Forest parameters
            :return: Generated data
            :rtype: pd.DataFrame
        """
        data = self.data
        data = data.sample(n=n, replace=replace)
        gen_data = data.copy()
        # Features are trained and generated one by one
        for i in range(len(data.columns)):
            # May bug with duplicate names in columns
            y = data.columns[i] # name
            Y = data[y]         # data
            X = data.drop(data.columns[i], axis=1)
            # Regressor or classifier
            if data.columns[i] in data.indexes['numerical']:
                model = self.regressor(**kwargs)
            else:
                model = self.classifier(**kwargs)
            # FIT
            model.fit(X, Y)
            # GENERATE
            for x in list(data.index.values): # Loop over rows
                if np.random.random() < p:
                    row = data.loc[[x]].drop(y, axis=1)
                    # DEBUG
                    prediction = model.predict(row)
                    if isinstance(prediction, np.ndarray):
                        gen_data.at[x, y] = prediction[0]
                    else:
                        gen_data.at[x, y] = prediction
        return gen_data

    def sample(self, n=1, p=0.8, replace=True):
        """ Generate n rows by copying data and then do values imputations.
        
            :param n: Number of examples to sample
            :param p: The probability of changing a value
                        if p=0, the generated dataset will be equals to the original
                        if p=1, the generated dataset will contains only new values

            :return: Generated data
            :rtype: pd.DataFrame
        """
        if self.data is not None:
            data = self.data
        else:
            raise('The generator needs to be trained before you can sample from it. Please use fit method.')
        data = data.sample(n=n, replace=replace)
        gen_data = data.copy()
        # Loop over examples
        for x in list(data.index.values):
            # Loop over features
            for i, y in enumerate(list(data.columns.values)):
                if np.random.random() < p:
                    row = data.loc[[x]].drop(y, axis=1)
                    # DEBUG
                    prediction = self.models[i].predict(row)
                    if isinstance(prediction, np.ndarray):
                        gen_data.at[x, y] = prediction[0]
                    else:
                        gen_data.at[x, y] = prediction
        return gen_data.reset_index(drop=True)
