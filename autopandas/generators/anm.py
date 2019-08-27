# Additive Noise Model

# Imports
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone

class ANM():
    def __init__(self, model=None):
        """ Data generator using multiple imputations with random forest (or another model).

            :param model: Model used for imputations.
        """
        # List of Random Forests
        self.models = []
        if model is None: # Default Random Forest
            self.regressor = RandomForestRegressor()
            self.classifier = RandomForestClassifier()
        else: # Custom model
            self.regressor = model
            self.classifier = model
        # Store data to be able to sample from original data
        self.data = None

    def fit(self, data):
        """ Fit one random forest for each column, given the others.
        """
        self.data = data
        for i in range(len(data.columns)):
            # May bug with duplicate names in columns
            y = data[data.columns[i]]
            X = data.drop(data.columns[i], axis=1)
            # Regressor or classifier
            if data.columns[i] in data.indexes['numerical']:
                model = clone(self.regressor)
            else:
                model = clone(self.classifier)
            model.fit(X, y)
            self.models.append(model)

    def partial_fit_generate(self, n=1, p=0.8, replace=True):
        """ Fit and generate for high dimensional case.
            To avoid memory error, features are trained and generated one by one.

            :param p: The probability of changing a value
                        if p=0, the generated dataset will be equals to the original
                        if p=1, the generated dataset will contains only new values
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
                model = self.regressor
            else:
                model = self.classifier
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
        # TODO: predicted matrix, residual matrix, noise boolean argument
        if self.data is not None:
            data = self.data
        else:
            raise('The ANM generator needs to be trained before you can sample from it. Please use fit method.')
        gen_data = data.sample(n=n, replace=replace)
        # Loop over examples
        for x in list(gen_data.index.values):
            # Loop over features
            for i, y in enumerate(list(data.columns.values)):
                if np.random.random() < p:
                    row = data.loc[[x]].drop(y, axis=1)
                    prediction = self.models[i].predict(row)
                    if isinstance(prediction, np.ndarray):
                        prediction = prediction[0] # select first value if needed
                    gen_data.at[x, y] = prediction
        return gen_data.reset_index(drop=True)

# MEMO (for the TODO)
'''
def generate(self):
    data = self.get_data()
    predicted_matrix = np.zeros(data.shape)
    residual_matrix = np.zeros(data.shape)
    for x in list(data.index.values):
        for i, y in enumerate(list(data.columns.values)):
            row = data.loc[[x]].drop(y, axis=1)
            predicted_matrix[x, i] = self.models[i].predict(row)
            residual_matrix[x, i] = (predicted_matrix[x,i] - data.loc[x, y])**2
    var_vector = np.mean(residual_matrix, axis=0)
    for i in range(predicted_matrix.shape[0]):
        row = predicted_matrix[i, :]
        for j, y in enumerate(list(data.columns.values)):
            self.gen_data.at[i, y] = row[j] + np.random.normal(loc=0, scale=np.sqrt(var_vector[j]))

return self.gen_data
'''
