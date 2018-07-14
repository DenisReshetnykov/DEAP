from features import ParticipantFeatureVectors, createHugeFeatureVector, saveHugeFeatureVector, Features
from metrics import calculateAccuracy

import itertools
from inspect import signature
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import pydotplus

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from sklearn.metrics import confusion_matrix

CONST_C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
CONST_gamma = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
CONST_alpha = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

def classBalanceInfo(y):
    # some python magic with dict comperhension to compute class balance. {class: (points, portion)}
    # ex.: {1: (404, 0.5), 2: (106, 0.13), 3: (80, 0.1), 4: (85, 0.1), 5: (137, 0.17)}
    return {unique_y: (
    len([y_i for y_i in y if y_i == unique_y]), round(len([y_i for y_i in y if y_i == unique_y]) / len(y), 2)) for
            unique_y in list(set(y))}

def getModelScores(model, X_train, Y_train, X_test, Y_test, coef=True):
    print('Model itself score: ' + str(model.score(X_train, Y_train)))
    print('Model X_test score: ' + str(model.score(X_test, Y_test)))
    if coef:
        print("model.coef_: {}".format(model.coef_))
        print("model.intercept_: {}".format(model.intercept_))

def useModel(modelFunction, X, y, *args):
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=1987)
    predicted = modelFunction(X_train, Y_train, X_test, Y_test, *args)
    accuracy = np.mean(predicted == Y_test)
    return accuracy

def useModelWithParams(modelFunction, X, y, params):
    '''
    combine all possible parameters, put each combination to the modelFunction and save result
    :param modelFunction: function for data model
    :param X: feature dataframe
    :param y: target variable vector where y.shape = X.shape[1]
    :param params: tuple of list with parameters values, eg. ([0.01, 0.03, 0.1], [1, 3, 10], [2, 5, 10])
    :return: dict where key is tuple of parameters for model combination and value is model accuracy
    example: useModelWithParams(supportVectorClassifier, X, y, (CONST_C, CONST_gamma))
    '''
    returnByParam = {}
    for combination in itertools.product(*params):
        returnByParam[combination] = useModel(modelFunction, X, y, *combination)
    return returnByParam


def linearRegression(X_train, Y_train, X_test, Y_test):
    model = LinearRegression(normalize=False).fit(X_train, Y_train)
    predicted = model.predict(X_test)
    # plt.plot(model.coef_)
    # plt.show()
    # getModelScores(model, X_train, Y_train, X_test, Y_test)
    return predicted, model.score(X_test, Y_test)

def useLinearRegressionModel(X, y):
    X_train, X_test, Y_train, Y_test = splitDataToTrainTest(X, y)
    predicted, test_score = linearRegression(X_train, Y_train, X_test, Y_test)
    return test_score

def RidgeRegression(X_train, Y_train, X_test, Y_test, alpha):
    model = Ridge(alpha=alpha).fit(X_train, Y_train)
    predicted = model.predict(X_test)
    # plt.plot(model.coef_)
    # plt.show()
    # getModelScores(model, X_train, Y_train, X_test, Y_test)
    return predicted, model.score(X_test, Y_test)

def useRidgeRegressionModel(X, y, alpha=1):
    X_train, X_test, Y_train, Y_test = splitDataToTrainTest(X, y)
    predicted, test_score = RidgeRegression(X_train, Y_train, X_test, Y_test, alpha)
    return test_score

def LassoRegression(X_train, Y_train, X_test, Y_test, alpha, max_iter):
    model = Lasso(alpha=alpha, max_iter=max_iter).fit(X_train, Y_train)
    predicted = model.predict(X_test)
    # getModelScores(model, X_train, Y_train, X_test, Y_test)
    print("Количество использованных признаков: {}".format(np.sum(model.coef_ != 0)))
    return predicted, model.score(X_test, Y_test)

def useLassoRegressionModel(X, y, alpha=0.001, max_iter=1000000):
    X_train, X_test, Y_train, Y_test = splitDataToTrainTest(X, y)
    predicted, test_score = LassoRegression(X_train, Y_train, X_test, Y_test, alpha, max_iter)
    return test_score


#                                                             Classification models
def logisticRegression(X_train, Y_train, X_test, Y_test, C):
    model = LogisticRegression(C=C).fit(X_train, Y_train)
    predicted = model.predict(X_test)
    # getModelScores(model, X_train, Y_train, X_test, Y_test)
    return predicted

def linearSVC(X_train, Y_train, X_test, Y_test, C):
    model = LinearSVC(C=C).fit(X_train, Y_train)
    predicted = model.predict(X_test)
    # getModelScores(model, X_train, Y_train, X_test, Y_test)
    return predicted

def kNeighborsClassification(X_train, Y_train, X_test, Y_test, n_neighbors):
    model = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train, Y_train)
    predicted = model.predict(X_test)
    return predicted

def naiveBayesClassification(X_train, Y_train, X_test, Y_test):
    model = GaussianNB().fit(X_train, Y_train)
    predicted = model.predict(X_test)
    return predicted

def decisionTreeClassification(X_train, Y_train, X_test, Y_test, max_depth):
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=0).fit(X_train, Y_train)
    predicted = model.predict(X_test)
    # getModelScores(model, X_train, Y_train, X_test, Y_test, coef=False)
    # dot_data = export_graphviz(model, out_file=None, class_names=["1", "2", "3", "4", "5"],
    #                 feature_names=X_train.columns, impurity=False, filled=True)
    # graph = pydotplus.graph_from_dot_data(dot_data)
    # graph.write_pdf("tree.pdf")
    # for name, importance in zip (X_train.columns, model.feature_importances_):
    #     if importance != 0:
    #         print(name, importance)
    return predicted

def ensembleRandomForestClassifier(X_train, Y_train, X_test, Y_test, n_estimators, max_features):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=0, max_features=max_features).fit(X_train,
                                                                                                             Y_train)
    predicted = model.predict(X_test)
    # getModelScores(model, X_train, Y_train, X_test, Y_test, coef=False)
    return predicted

def gradientBoostingClassifier(X_train, Y_train, X_test, Y_test, max_depth=3, learning_rate=0.1, n_estimators=100):
    model = GradientBoostingClassifier(random_state=0, max_depth=max_depth, learning_rate=learning_rate,
                                       n_estimators=n_estimators).fit(X_train, Y_train)
    predicted = model.predict(X_test)
    # getModelScores(model, X_train, Y_train, X_test, Y_test, coef=False)
    return predicted

def supportVectorClassifier(X_train, Y_train, X_test, Y_test, C, gamma):
    model = SVC(kernel='rbf', C=C, gamma=gamma).fit(X_train, Y_train)
    predicted = model.predict(X_test)
    # getModelScores(model, X_train, Y_train, X_test, Y_test, coef=False)
    return predicted

def mLPClassifier(X_train, Y_train, X_test, Y_test, hidden_layer_sizes=[10], alpha=0.1):
    model = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=hidden_layer_sizes, alpha=alpha).fit(
        X_train, Y_train)
    predicted = model.predict(X_test)
    # getModelScores(model, X_train, Y_train, X_test, Y_test, coef=False)
    return predicted

if __name__ == "__main__":
    MODELS = [(logisticRegression, 1), (linearSVC, 1), (kNeighborsClassification, 5), (naiveBayesClassification,),
            (decisionTreeClassification, 4), (ensembleRandomForestClassifier, 8, 58),
            (gradientBoostingClassifier, 1, 0.01, 100), (supportVectorClassifier, 0.3, 3), (mLPClassifier, [10, 10], 3)]

    X = Features(pd.read_csv('feature_vectors/huge/FV_familiarity.csv', index_col=0)).normalizeData()
    # X = Features(pd.read_csv('feature_vectors/huge/FV_familiarity.csv', index_col=0)).dataframe
    y = np.array(pd.read_csv('feature_vectors/huge/YV_familiarity.csv', index_col=0).astype(int)).ravel()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter([xs for xs in range(8, 9) for ys in range(40, 70)], [ys for xs in range(8, 9) for ys in range(40, 70)],
    #            [useModel(ensembleRandomForestClassifier, X, y, n, m) for n in range(8, 9) for m in range(40, 70)])
    # plt.show()

    # print('LinearRegression model score: {}'.format(useLinearRegressionModel(X, y)))
    # print('RidgeRegression model score: {}'.format(useRidgeRegressionModel(X, y)))
    # print('LassoRegression model score: {}'.format(useLassoRegressionModel(X, y)))

    for model in MODELS:
        print('{} model with parameters {} score: {}'.format(model[0].__name__,
                                                             ['{}={}'.format(name, value) for (name, value) in
                                                              zip(list(signature(model[0]).parameters.keys())[4:],
                                                                  model[1:])], useModel(model[0], X, y, *model[1:])))



















