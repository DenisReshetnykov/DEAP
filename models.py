from features import ParticipantFeatureVectors, createHugeFeatureVector, saveHugeFeatureVector, Features
from metrics import calculateAccuracy

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pydotplus

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.svm import SVR, LinearSVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from sklearn.metrics import confusion_matrix


def classBalanceInfo(y):
    # some python magic with dict comperhension to compute class balance. {class: (points, portion)}
    # ex.: {1: (404, 0.5), 2: (106, 0.13), 3: (80, 0.1), 4: (85, 0.1), 5: (137, 0.17)}
    return {unique_y: (
    len([y_i for y_i in y if y_i == unique_y]), round(len([y_i for y_i in y if y_i == unique_y]) / len(y), 2)) for
            unique_y in list(set(y))}

def splitDataToTrainTest(X, y, test_size=0.3, random_state=1987):
    # print('X type is {} and shape {}. y type is {} and shape {}'.format(type(X), X.shape, type(y), y.shape))
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # print('ClassBalance for train {} and for test {}'.format(classBalanceInfo(Y_train), classBalanceInfo(Y_test)))
    # print('X_train {}, X_test {}, Y_train {}, Y_test {}'.format(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape))
    return X_train, X_test, Y_train, Y_test

def getModelScores(model, X_train, Y_train, X_test, Y_test, coef=True):
    print('Model itself score: ' + str(model.score(X_train, Y_train)))
    print('Model X_test score: ' + str(model.score(X_test, Y_test)))
    if coef:
        print("model.coef_: {}".format(model.coef_))
        print("model.intercept_: {}".format(model.intercept_))


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
    # plt.plot(model.coef_)
    # plt.show()
    # getModelScores(model, X_train, Y_train, X_test, Y_test)
    print("Количество использованных признаков: {}".format(np.sum(model.coef_ != 0)))
    return predicted, model.score(X_test, Y_test)

def useLassoRegressionModel(X, y, alpha=0.001, max_iter=1000000):
    X_train, X_test, Y_train, Y_test = splitDataToTrainTest(X, y)
    predicted, test_score = LassoRegression(X_train, Y_train, X_test, Y_test, alpha, max_iter)
    return test_score


#                                                             Classification algorithms
def logisticRegression(X_train, Y_train, X_test, Y_test, C):
    model = LogisticRegression(C=C).fit(X_train, Y_train)
    predicted = model.predict(X_test)
    getModelScores(model, X_train, Y_train, X_test, Y_test)
    return predicted

def useLogisticRegression(X, y, C=1):
    X_train, X_test, Y_train, Y_test = splitDataToTrainTest(X, y)
    predicted = logisticRegression(X_train, Y_train, X_test, Y_test, C)
    accuracy = np.mean(predicted == Y_test)
    return accuracy

def linearSVC(X_train, Y_train, X_test, Y_test, C):
    model = LinearSVC(C=C).fit(X_train, Y_train)
    predicted = model.predict(X_test)
    # getModelScores(model, X_train, Y_train, X_test, Y_test)
    return predicted

def useLinearSVC(X, y, C=1):
    X_train, X_test, Y_train, Y_test = splitDataToTrainTest(X, y)
    predicted = linearSVC(X_train, Y_train, X_test, Y_test, C)
    accuracy = np.mean(predicted == Y_test)
    return accuracy

def kNeighborsClassification(X_train, Y_train, X_test, Y_test, n_neighbors):
    model = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train, Y_train)
    predicted = model.predict(X_test)
    return predicted

def usekNeighborsClassificationModel(X, y, n_neighbors=1):
    X_train, X_test, Y_train, Y_test = splitDataToTrainTest(X, y)
    predicted = kNeighborsClassification(X_train, Y_train, X_test, Y_test, n_neighbors)
    accuracy = np.mean(predicted == Y_test)
    return accuracy

def naiveBayesClassification(X_train, Y_train, X_test, Y_test):
    model = GaussianNB().fit(X_train, Y_train)
    predicted = model.predict(X_test)
    return predicted

def useNBClassification(X, y):
    X_train, X_test, Y_train, Y_test = splitDataToTrainTest(X, y)
    predicted = naiveBayesClassification(X_train, Y_train, X_test, Y_test)
    accuracy = np.mean(predicted == Y_test)
    return accuracy

def decisionTreeClassification(X_train, Y_train, X_test, Y_test, max_depth):
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=0).fit(X_train, Y_train)
    predicted = model.predict(X_test)
    getModelScores(model, X_train, Y_train, X_test, Y_test, coef=False)
    # export_graphviz(model, out_file="tree.dot", class_names=["1", "2", "3", "4", "5"],
    #                 feature_names=X_train.columns, impurity=False, filled=True)
    dot_data = export_graphviz(model, out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("tree.pdf")
    return predicted

def useDecisionTreeClassifier(X, y, max_depth=5):
    X_train, X_test, Y_train, Y_test = splitDataToTrainTest(X, y)
    predicted = decisionTreeClassification(X_train, Y_train, X_test, Y_test, max_depth)
    accuracy = np.mean(predicted == Y_test)
    return accuracy



if __name__ == "__main__":
    X = Features(pd.read_csv('feature_vectors/huge/FV_familiarity.csv', index_col=0)).normalizeData()
    # X = Features(pd.read_csv('feature_vectors/huge/FV_familiarity.csv', index_col=0)).dataframe
    y = np.array(pd.read_csv('feature_vectors/huge/YV_familiarity.csv', index_col=0).astype(int)).ravel()
    # print('classBalanceInfo is:'+str(classBalanceInfo(y)))

    # plt.plot([n for n in range(1, 41)], [useModel(X.normDF, y, n_neighbors) for n_neighbors in range(1, 41)])
    # plt.show()

    # plt.plot([n for n in range(1, 10)], [useDecisionTreeClassifier(X, y, max_depth) for max_depth in range(1, 10)])
    # plt.show()


    # print('LinearRegression model score: {}'.format(useLinearRegressionModel(X, y)))
    # print('RidgeRegression model score: {}'.format(useRidgeRegressionModel(X, y)))
    # print('LassoRegression model score: {}'.format(useLassoRegressionModel(X, y)))
    # print('LogisticRegression model score: {}'.format(useLogisticRegression(X, y, 1)))
    # print('LinearSVC model score: {}'.format(useLinearSVC(X, y, 1)))
    # print('KNeighborsClassification model accuracy: {}'.format(usekNeighborsClassificationModel(X, y, 5)))
    # print('NaiveBayesClassification model score: {}'.format(useNBClassification(X, y)))
    print('DecisionTreeClassification model score: {}'.format(useDecisionTreeClassifier(X, y, 2)))



