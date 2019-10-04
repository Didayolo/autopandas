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
        # For noise behaviour
        self.predicted_matrix = None
        self.var_vector = None

    def fit(self, data, noise=False):
        """ Fit one random forest (or another model) for each column, given the others.

            :param noise: If True, add noise during sampling relative to the residual matrix
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
            # Fit one predictive model for each variable
            model.fit(X, y)
            self.models.append(model)
        # NOISE BEHAVIOUR
        # takes more time because needs to compute the residual matrix for the whole dataset
        if noise:
            self.predicted_matrix = np.zeros(data.shape)
            residual_matrix = np.zeros(data.shape)
            for x in list(data.index.values):
                for i, y in enumerate(list(data.columns.values)):
                    row = data.loc[[x]].drop(y, axis=1)
                    self.predicted_matrix[x, i] = self.models[i].predict(row)
                    residual_matrix[x, i] = (self.predicted_matrix[x,i] - data.loc[x, y])**2
            self.var_vector = np.mean(residual_matrix, axis=0)
        else: # RESET
            self.predicted_matrix = None
            self.var_vector = None

    def partial_fit_generate(self, n=1, p=0.8, replace=True, noise=False):
        """ Fit and generate for high dimensional case.
            To avoid memory error, features are trained and generated one by one.

            :param n: Number of examples to sample
            :param p: The probability of changing a value
                        if p=0, the generated dataset will be equals to the original
                        if p=1, the generated dataset will contains only new values
            :param replace: If True, sample the original data with replacement before the imputations
            :param noise: If True, add noise relative to the residual matrix. NOT IMPLEMENTED (not possible?)

            :return: Generated data
            :rtype: pd.DataFrame
        """
        if noise:
            raise Exception('noise argument is not compatible with partial_fit_generate. Please use fit and then sample methods.')
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

    def sample(self, n=1, p=0.8, replace=True, noise=False):
        """ Generate n rows by copying data and then do values imputations.

            :param n: Number of examples to sample
            :param p: The probability of changing a value
                        if p=0, the generated dataset will be equals to the original
                        if p=1, the generated dataset will contains only new values
            :param replace: If True, sample the original data with replacement before the imputations
            :param noise: If True, add noise relative to the residual matrix

            :return: Generated data
            :rtype: pd.DataFrame
        """
        if self.data is not None:
            data = self.data
        else:
            raise Exception('The ANM generator needs to be trained before you can sample from it. Please use fit method.')
        gen_data = data.sample(n=n, replace=replace)
        # NOISE BEHAVIOUR
        if noise:
            if self.var_vector is None:
                raise Exception('You must call fit method with noise=True before calling sample method with noise=True.')
            for x in list(gen_data.index.values):
                row = self.predicted_matrix[x, :]
                for i, y in enumerate(list(data.columns.values)):
                    if np.random.random() < p: # with probability p
                        # may need the ndarray debug...
                        gen_data.at[x, y] = row[i] + np.random.normal(loc=0, scale=np.sqrt(self.var_vector[i]))
        # CLASSICAL BEHAVIOUR
        else:
            # Loop over examples
            for x in list(gen_data.index.values):
                # Loop over features
                for i, y in enumerate(list(data.columns.values)):
                    if np.random.random() < p: # with probability p
                        row = data.loc[[x]].drop(y, axis=1)
                        prediction = self.models[i].predict(row)
                        if isinstance(prediction, np.ndarray):
                            prediction = prediction[0] # select first value if needed
                        gen_data.at[x, y] = prediction
        return gen_data.reset_index(drop=True)
