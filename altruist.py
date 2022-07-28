import numpy as np
import math


class Altruist:
    """
    Altruist V2:  A tool for providing more truthful interpretations, as well as a tool for selection and benchmarking.
    Altruist works with every machine learning model which provides predictions in the form of continuous values (eg. probabilities).
    It uses feature importance techniques like LIME, SHAP, etc.
    Extends our previous tool: https://github.com/iamollas/Altruist
    ...

    Methods
    -------
    find_untruthful_features(instance)
        It generates the interpretations for this instance's prediction. Then, identifies the untruthful features in each interpretation. In the process
        saves any counterfactual information found in the process. It returns the list of untruthful features of each interpretation.
    explain_why(instance, fi, truthful_only=False)
        It runs the exact same pipeline as the previous pipeline, but only for a selected feature importance technique. It generates the arguments which explain how the untruthful features occured, and why the interpretation is not trusted.
    """

    def __init__(self, predict, dataset, fi_technique, feature_names=None, level='weak', delta=0.0001, data_type='Tabular', logs=False, args=False):
        """
        Parameters
        ----------
            predict: ml predict function
                The machine learning model which must provide either continuous output (regression problems), or output in the form of probabilities (classification problems).
                predict must be further designed by the user. Check the predict examples. 
            dataset: seed dataset
                A set of instances to extract few statistics (mean, std, max, min) across the features
            fi_technique: function or list of functions
                The interpretation(s) technique(s) provided by the system designer / user
            feature_names: list
                The names of the features
            data_type: string
                can be Tabular, TFIDF (Text), Embeddings (Text), per Sensor (TimeSeries), Image
        """
        if type(dataset) == type(np.array([])):
            self.dataset = dataset
        else:
            self.dataset = np.array(dataset)
        self.shape = self._identify_shape()
        self.data_type = data_type
        if self.data_type != 'Embeddings':
            self.flat_dataset = self._flatten_data(
                self.dataset)  # 1D, 2D, 3D only
            self.number_of_features = len(self.flat_dataset[0])
            self.features_statistics = self._extract_feature_statistics()
        if data_type == 'Image':
            self.number_of_features = 36  # number of segments
        if data_type == 'per Sensor' or data_type == 'Embeddings':
            self.number_of_features = len(feature_names)

        if feature_names is None:
            self.feature_names = [str('F_'+str(j))
                                  for j in range(self.number_of_features)]
        else:
            self.feature_names = feature_names
        if self.data_type != 'Embeddings':
            # identify integer only features to reduce tests
            self.cbi_features = self._identify_cbi_features()

        self.predict = predict
        self.fi_technique = fi_technique  # function when 1, list of function when 2+
        self.multiple_fis = False
        if type(self.fi_technique) is list:
            self.multiple_fis = True
        if self.multiple_fis:
            self.fis = len(fi_technique)
        else:
            self.fis = 1
        self.level = level
        self.delta = delta
        self.noise_level = {'very weak': 1/4, 'weak': 1 /
                            2, 'normal': 1, 'strong': 2, 'extreme': 4}
        self.logs = logs
        self.args = args

    def find_untruthful(self, instance, precomputed=None, segmentation=None):
        """
        Parameters
        ----------
            instance: array
                The instance which prediction's interpretation(s) Altruist will investigate.

        Returns
        -------
        list
            The untruthful features appearing in the interpretation(s) technique(s)
        list
            A list of counterfactual values, that may change drastically the prediction
        """
        if type(instance) != type(np.array([])):
            instance = np.array(instance)
        self.precomputed = precomputed
        self.original_instance = instance.copy()
        instance = self._flatten_data(np.array([instance]))
        self.temp_tests = {}
        if self.data_type == 'Image':
            self.key = tuple(instance[0].copy().reshape(
                (self.shape[0]*self.shape[1]*self.shape[2])).tolist())
            return self._query(instance, segmentation)
        elif self.data_type == 'Embeddings':
            self.key = tuple(instance[0].copy().reshape(
                (self.shape[0]*self.shape[1])).tolist())
            return self._query(instance)
        else:
            self.key = tuple(instance[0])
        return self._query(instance)

    def _query(self, instance, segmentation=None):
        fi_truthfulness = []
        fis = []
        argumentation = []
        for fi in range(1, self.fis+1):
            if self.data_type == 'Image':
                evaluated, feature_importance = self._test(
                    instance, fi, segmentation)
                feature_names = [str('S_'+str(i))
                                 for i in range(len(np.unique(segmentation)))]
            elif self.data_type == 'Embeddings':
                evaluated, feature_importance = self._test(instance, fi)
                feature_names = self.feature_names[:len(evaluated)]
            else:
                evaluated, feature_importance = self._test(instance, fi)
                feature_names = self.feature_names
            fis.append(feature_importance)
            untruthful_features = []
            partly_truthful_featues = []
            arguments = {}
            for feature in feature_names:
                find = feature_names.index(feature)
                if len(evaluated[find]) == 0 or len(evaluated[find]) == 1:
                    untruthful_features.append(feature)
                    arguments[find] = []
                elif self.args:
                    ts = self.temp_tests[tuple(self.key)]
                    if feature_importance[find] > 0:
                        w1 = 'INC'
                        w2 = 'Increased'
                        w3 = 'DEC'
                        w4 = 'Decreased'
                    elif feature_importance[find] < 0:
                        w1 = 'INC'
                        w2 = 'Decreased'
                        w3 = 'DEC'
                        w4 = 'Increased'
                    else:
                        w1 = 'STA'
                        w2 = 'Remained Stable'
                        w3 = 'STA'
                        w4 = 'Remained Stable'
                    altr = 'to'
                    if self.data_type in ['per Sensor', 'Image']:
                        altr = 'by'
                    if instance[0][find] == ts[find]['alterations'][0]:
                        arg_a = 'The alteration (' + w1 + ') did not happen, as the feature value had the max value'
                    else:
                        arg_a = '$f_{'+feature+', '+w1+'}$: The evaluation of the alteration of $' + feature + '$\'s value '+altr+' $'+str(round(ts[find]['alterations'][0],4)) + '$ ($'+w1+'$) was performed and the model\'s behaviour was as expected $' + w2+'$ ('+str(round(
                        ts['prediction'], 4))+' to '+str(round(ts[find]['alt_predictions'][0], 4)) + '), according to its importance $z_{'+feature+'}=' + str(round(feature_importance[find], 4)) + '$.'
                    if instance[0][find] == ts[find]['alterations'][1]:
                        arg_b = 'The alteration (' + w3 + ') did not happen, as the feature value had the max value'
                    else:
                        arg_b = '$f_{'+feature+', '+w3+'}$: The evaluation of the alteration of $' + feature + '$\'s value '+altr+' $'+str(round(ts[find]['alterations'][1],4)) + '$ ($'+w3+'$) was performed and the model\'s behaviour was as expected $' + w4+'$ ('+str(round(
                        ts['prediction'], 4))+' to '+str(round(ts[find]['alt_predictions'][1], 4)) + '), according to its importance $z_{'+feature+'}=' + str(round(feature_importance[find], 4)) + '$.'
                    arguments[find] = [arg_a, arg_b]
            fi_truthfulness.append(untruthful_features)
            argumentation.append(arguments)
        counter_factuals = []
        average_change = []
        for feature in range(len(feature_names)):
            [counter_factuals.append(
                c) for c in self.temp_tests[self.key][feature]['counter_factuals'] if len(c) != 0]
            average_change.append(
                self.temp_tests[self.key][feature]['average_change'])
        if self.args:
            return fi_truthfulness, counter_factuals, average_change, fis, argumentation
        return fi_truthfulness, counter_factuals, average_change, fis

    def _test(self, instance, fi, segmentation=None):
        if self.key in self.temp_tests:
            prediction = self.temp_tests[self.key]['prediction']
        else:
            prediction = self._predict_function(
                np.array([instance[0], instance[0]]))[0]
            self.temp_tests[self.key] = {}
            self.temp_tests[self.key]['prediction'] = prediction

        if self.multiple_fis:
            feature_importance = self.precomputed[fi-1] if (
                self.precomputed is not None) else self.fi_technique[fi-1](self.original_instance, self.predict)
        else:
            feature_importance = self.precomputed if (
                self.precomputed is not None) else self.fi_technique(self.original_instance, self.predict)
        list_of_evaluated = []
        if self.data_type == 'Image':
            for segment in range(len(np.unique(segmentation))):
                importance = feature_importance[segment]
                list_of_evaluated.append(self._evaluation(
                    segment, importance, self.original_instance, segmentation))
        elif self.data_type == 'Embeddings':
            for feature in range(self.number_of_features):
                if np.sum(self.original_instance[feature]) != 0:
                    importance = feature_importance[feature]
                    list_of_evaluated.append(
                        self._evaluation(feature, importance, instance))
        else:
            for feature in range(self.number_of_features):
                importance = feature_importance[feature]
                list_of_evaluated.append(
                    self._evaluation(feature, importance, instance))
        return list_of_evaluated, feature_importance

    def _evaluation(self, feature, importance, instance, segmentation=None):

        list_of_evaluated = []
        counter_factuals = []
        if self.logs:
            print(feature)
        if feature in self.temp_tests[self.key]:
            alterations = self.temp_tests[self.key][feature]['alterations']
            alt_predictions = self.temp_tests[self.key][feature]['alt_predictions']
            alt_predictions = [alt_predictions[0], alt_predictions[1]]
            prediction = self.temp_tests[self.key]['prediction']
        else:
            self.temp_tests[self.key][feature] = {}
            if self.data_type == 'per Sensor':
                temp_instance_i = self.original_instance.copy()
                temp_instance_d = self.original_instance.copy()
                alterations = self._determine_feature_change(
                    self.original_instance[:, feature].mean(axis=0), feature)
                min_ = self.features_statistics[feature][0]
                max_ = self.features_statistics[feature][1]
                for j in range(self.shape[0]):
                    temp_instance_i[j][feature] = min(
                        max_, temp_instance_i[j][feature] + alterations[0])
                    temp_instance_d[j][feature] = max(
                        min_, temp_instance_d[j][feature] - alterations[1])
                temp_instance_d = temp_instance_d.reshape(
                    (self.shape[0]*self.shape[1]))
                temp_instance_i = temp_instance_i.reshape(
                    (self.shape[0]*self.shape[1]))
            elif self.data_type == 'Image':
                temp_instance_i = self.original_instance.copy()
                temp_instance_d = self.original_instance.copy()
                sum = []
                for dim in range(self.shape[0]):
                    indexes = np.where(segmentation[dim] == feature)[0]
                    for index in indexes:
                        for k in range(3):
                            sum.append(self.original_instance[dim][index][k])
                alterations = self._determine_feature_change(
                    np.average(sum), feature)
                for dim in range(self.shape[0]):
                    indexes = np.where(segmentation[dim] == feature)[0]
                    for index in indexes:
                        for k in range(3):
                            temp_instance_i[dim][index][k] = min(
                                1, temp_instance_i[dim][index][k] + alterations[0])
                            temp_instance_d[dim][index][k] = max(
                                0, temp_instance_d[dim][index][k] - alterations[1])
            elif self.data_type == 'Embeddings':
                temp_instance_i = self.original_instance.copy()
                temp_instance_d = self.original_instance.copy()
                for k in range(len(temp_instance_d[feature])):
                    temp_instance_d[feature][k] = 0
                alterations = [0, 0]
            else:
                alterations = self._determine_feature_change(
                    instance[0][feature], feature, True)
                temp_instance_i = instance[0].copy()
                temp_instance_i[feature] = alterations[0]
                temp_instance_d = instance[0].copy()
                temp_instance_d[feature] = alterations[1]
            self.temp_tests[self.key][feature]['alterations'] = alterations

            alt_predictions = self._predict_function(
                np.array([temp_instance_i, temp_instance_d]))
            self.temp_tests[self.key][feature]['alt_predictions'] = alt_predictions
            alt_predictions = [alt_predictions[0], alt_predictions[1]]
            prediction = self.temp_tests[self.key]['prediction']

            average_change = (
                abs(prediction-alt_predictions[0]) + abs(prediction-alt_predictions[1]))/2
            self.temp_tests[self.key][feature]['average_change'] = average_change

            if (prediction < 0.5 and alt_predictions[0] >= 0.5) or (prediction >= 0.5 and alt_predictions[0] < 0.5):
                counter_factuals.append([feature, alterations[0]])
            if (prediction < 0.5 and alt_predictions[1] >= 0.5) or (prediction >= 0.5 and alt_predictions[1] < 0.5):
                counter_factuals.append([feature, alterations[1]])

            self.temp_tests[self.key][feature]['counter_factuals'] = counter_factuals

        flag_max = False
        flag_min = False
        if self.data_type == 'Embeddings':
            flag_max = True
            flag_min = False
        elif self.data_type != 'Image':
            if instance[0][feature].shape == alterations[0].shape:
                if instance[0][feature] == alterations[0]:
                    flag_max = True
                if instance[0][feature] == alterations[1]:
                    flag_min = True

        if str(type(importance)) == "<class 'numpy.ndarray'>":
            importance = importance[0]
        if self.logs:
            print(feature, prediction, alt_predictions[0], alt_predictions[1])
        if importance > 0:
            if flag_max or (prediction - alt_predictions[0] < self.delta):
                # Truthful, Positive, Increased
                list_of_evaluated.append("TPI")
            if flag_min or (prediction - alt_predictions[1] > -self.delta):
                # Truthful, Positive, Decreased
                list_of_evaluated.append("TPD")
        elif importance < 0:
            if flag_max or (prediction - alt_predictions[0] > -self.delta):
                # Truthful, Negative, Decreased
                list_of_evaluated.append("TND")
            if flag_min or (prediction - alt_predictions[1] < self.delta):
                # Truthful, Negative, Increased
                list_of_evaluated.append("TNI")
        else:
            if flag_max or prediction == alt_predictions[0] or abs(prediction - alt_predictions[0]) < self.delta:
                list_of_evaluated.append("TNS")  # Truthful, Neutral, Stable
            if flag_min or prediction == alt_predictions[1] or abs(prediction - alt_predictions[1]) < self.delta:
                list_of_evaluated.append("TNS")  # Truthful, Neutral, Stable
        return list_of_evaluated

    def _determine_feature_change(self, value, feature, random_state=0):
        """
        Parameters
        ----------
            value: integer
                The value we want to add noise to
            feature: integer
                The feature this value represents
            random_state: float
                Random state for anchoring randomness

        Returns
        -------
            new_value
                The value with added noise
            new_value_op
                The value with subtracted noise
        """
        if self.data_type == 'TextEmb':
            new_value = value
            new_value_op = 0
            return new_value, new_value_op
        elif len(self.shape) == 3 or self.data_type == 'Image':
            min_ = self.features_statistics[0]
            max_ = self.features_statistics[1]
            mean_ = self.features_statistics[2]
            std_ = self.features_statistics[3]
            seed = abs(int((value**2)*100)+feature +
                       int((mean_**2)*100)+int((std_**2)*100)+random_state)
            while abs(seed) >= (2**32-1):
                seed = int(seed/10)
            np.random.seed(seed)
            noise = abs(mean_ - np.random.normal(mean_, std_, 1)
                        [0])*self.noise_level[self.level]  # Gaussian Noise/
            return noise, noise

        min_ = self.features_statistics[feature][0]
        max_ = self.features_statistics[feature][1]
        mean_ = self.features_statistics[feature][2]
        std_ = self.features_statistics[feature][3]
        seed = abs(int((value**2)*100)+int((mean_**2)*100) +
                   int((std_**2)*100)+random_state)
        while abs(seed) >= (2**32-1):
            seed = int(seed/10)
        np.random.seed(seed)
        noise = abs(mean_ - np.random.normal(mean_, std_, 1)
                    [0])*self.noise_level[self.level]  # Gaussian Noise/
        if self.data_type == 'per Sensor':
            return noise, noise

        new_value = value + noise
        new_value_op = value - noise

        if new_value > max_:
            new_value = max_
        if new_value_op < min_:
            new_value_op = min_
        if self.cbi_features is not None and feature in self.cbi_features:
            if self.cbi_features[feature] == 1:
                new_value = math.ceil(new_value)
                new_value_op = math.floor(new_value_op)
        return new_value, new_value_op

    def _identify_shape(self):
        return self.dataset[0].shape

    def _flatten_data(self, data):
        if len(self.shape) == 1:
            return data
        elif len(self.shape) == 2:
            return data.reshape((len(data), self.shape[0]*self.shape[1]))
        elif len(self.shape) == 3:
            return data.reshape((len(data), self.shape[0]*self.shape[1]*self.shape[2]))
        else:
            print('Sorry! Altruist supports 1D, 2D and 3D inputs only')

    def _extract_feature_statistics(self):
        """
        This function computes few feature statistics (min, max, mean, std per feature) given a dataset
        """
        if self.data_type == 'Image':
            features_statistics = []
            features_statistics.append(self.dataset.min())
            features_statistics.append(self.dataset.max())
            features_statistics.append(self.dataset.mean())
            features_statistics.append(self.dataset.std())
        elif self.data_type == 'per Sensor':
            features_statistics = {}
            for sensor in range(self.shape[1]):
                features_statistics[sensor] = []
                features_statistics[sensor].append(
                    self.dataset[:, sensor:sensor + 1].min())
                features_statistics[sensor].append(
                    self.dataset[:, sensor:sensor + 1].max())
                features_statistics[sensor].append(
                    self.dataset[:, sensor:sensor + 1].mean())
                features_statistics[sensor].append(
                    self.dataset[:, sensor:sensor + 1].std())
        else:
            dataset = self.flat_dataset
            number_of_features = self.number_of_features
            features_statistics = {}

            for feature in range(number_of_features):
                features_statistics[feature] = []
                features_statistics[feature].append(
                    dataset[:, feature:feature + 1].min())
                features_statistics[feature].append(
                    dataset[:, feature:feature + 1].max())
                features_statistics[feature].append(
                    dataset[:, feature:feature + 1].mean())
                features_statistics[feature].append(
                    dataset[:, feature:feature + 1].std())
        return features_statistics

    def _identify_cbi_features(self):
        """
        This function identifies categorical/boolean/integer features.

        """
        cbi = []
        for i in range(len(self.flat_dataset[0])):
            temp = self.flat_dataset[:, i]
            if np.mod(temp, 1).nonzero()[0].shape[0] == 0:
                cbi.append(self.feature_names[i])
        return cbi

    def _predict_function(self, instances):
        """
        This function creates a pseudo predict function. Based on the data type it performs a few reshapes and it then feeds them into the predictive model

        Parameters
        ----------
            instances: sample of data
                instances to predict
        """
        if type(instances) != type(np.array([])):
            instances = np.array(instances)
        if len(self.shape) == 1:
            return self.predict(instances)
        elif len(self.shape) == 2:
            instances = instances.reshape(
                (len(instances), self.shape[0], self.shape[1]))
            return self.predict(instances)
        elif len(self.shape) == 3:
            if self.data_type == 'Image':
                if len(instances.shape) < 4:
                    instances = instances.reshape(
                        (instances.shape[0], self.shape[0], self.shape[1], self.shape[2]))
                return self.predict(instances)
            else:
                instances = instances.reshape(
                    (len(instances), self.shape[0], self.shape[1], self.shape[2]))
            return self.predict(instances)
        else:
            print('Sorry! Altruist supports 1D, 2D and 3D inputs only')

    def _set_level(self, level):
        """
        This function sets the level of noise. Noise can have these values 'very weak', 'weak', 'normal', 'strong' and 'extreme'. Setting level to 'normal'."

        Parameters
        ----------
            level: string
                Level of noise can have these values 'very weak', 'weak', 'normal', 'strong' and 'extreme'. Setting level to 'normal'
        """

        if level in ['very weak', 'weak', 'normal', 'strong', 'extreme']:
            self.level = level
        else:
            print("Level of noise can have these values 'very weak', 'weak', 'normal', 'strong' and 'extreme'. Setting level to 'normal'.")
