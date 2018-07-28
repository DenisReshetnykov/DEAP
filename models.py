from features import ParticipantFeatureVectors, createHugeFeatureVector, saveHugeFeatureVector, Features
from metrics import calculateAccuracy
from plotting import plotHistForPCA, plotPCA

import itertools
from inspect import signature
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import pydotplus

from sklearn.decomposition import PCA, NMF
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectPercentile, SelectFromModel
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
CONST_mLP1 = [[n] for n in range(10, 100, 10)]
CONST_mLP2 = [[n, m] for n in range(10, 100, 10) for m in range(10, 100, 10)]
CONST_mLP3 = [[n, m, l] for n in range(10, 100, 10) for m in range(10, 100, 10) for l in range(10, 100, 10)]

def classBalanceInfo(y):
    # some python magic with dict comperhension to compute class balance. {class: (points, portion)}
    # ex.: {1: (404, 0.5), 2: (106, 0.13), 3: (80, 0.1), 4: (85, 0.1), 5: (137, 0.17)}
    y = np.array(y).ravel()
    return {unique_y: (
    len([y_i for y_i in y if y_i == unique_y]), round(len([y_i for y_i in y if y_i == unique_y]) / len(y), 2)) for
            unique_y in list(set(y))}

def classBalancing(X_unb, y_unb):
    classdict = {key: value[0] for key, value in classBalanceInfo(y_unb).items()}
    Xy = X_unb.assign(y=pd.DataFrame(data=y_unb))
    Xy_bal = []
    for cls in classdict.keys():
        Xy_bal.append(Xy.loc[Xy['y'] == cls].sample(min(classdict.values()), random_state=1987))
    result = pd.concat(Xy_bal)
    X_bal = result.iloc[:, :-1]
    y_bal = result.iloc[:, -1]
    return X_bal, y_bal

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

def adaBoostClassifier(X_train, Y_train, X_test, Y_test, n_estimators, learning_rate):
    model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=1).fit(X_train, Y_train)
    predicted = model.predict(X_test)
    return predicted

def principalComponent(X):
    pca = PCA(n_components=2)
    pca.fit(X)
    X_pca = pca.transform(X)
    print("Форма исходного массива: {}".format(str(X.shape)))
    print("Форма массива после сокращения размерности: {}".format(str(X_pca.shape)))
    print("форма главных компонент: {}".format(pca.components_.shape))
    print("компоненты PCA:\n{}".format(pca.components_))
    plt.matshow(pca.components_, cmap='viridis')
    plt.yticks([0, 1], ["Первая компонента", "Вторая компонента"])
    plt.colorbar()
    plt.xticks(range(len(X.columns)),
               X.columns, rotation=85, ha='left')
    plt.xlabel("Характеристика")
    plt.ylabel("Главные компоненты")
    plt.show()
    return X_pca

def negativeMatrixFactorization(X):
    nmf = NMF(n_components=2, random_state=0)
    nmf.fit(X)
    X_nmf = nmf.transform(X)
    print("Форма исходного массива: {}".format(str(X.shape)))
    print("Форма массива после сокращения размерности: {}".format(str(X_nmf.shape)))
    print("форма гNMF компонент: {}".format(nmf.components_.shape))
    print("компоненты NMF:\n{}".format(nmf.components_))
    plt.matshow(nmf.components_, cmap='viridis')
    plt.yticks([0, 1], ["Первая компонента", "Вторая компонента"])
    plt.colorbar()
    # plt.xticks(range(len(X.columns)),
    #            X.columns, rotation=85, ha='left')
    plt.xlabel("Характеристика")
    plt.ylabel("Главные компоненты")
    plt.show()
    return X_nmf

def selectFeatures(X_train, Y_train, X_test, percentile):
    select = SelectPercentile(percentile=percentile)
    select.fit(X_train, Y_train)
    X_train_selected = select.transform(X_train)
    X_test_selected = select.transform(X_test)
    print("форма массива X_train: {}".format(X_train.shape))
    print("форма массива X_train_selected: {}".format(X_train_selected.shape))
    return X_train_selected, X_test_selected

def scalingData(X_train, X_test, type):
    if type == 'MinMax':
        scaler = MinMaxScaler()
    elif type == 'Robust':
        scaler = RobustScaler()
    elif type == 'Standart':
        scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def useModel(modelFunction, X, y, scaling=False, selection=False, *args):
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=1987)
    if scaling:
        X_train, X_test = scalingData(X_train, X_test, scaling)
    if selection:
        X_train, X_test = selectFeatures(X_train, Y_train, X_test, selection)
    predicted = modelFunction(X_train, Y_train, X_test, Y_test, *args)
    accuracy = np.mean(predicted == Y_test)
    return accuracy

def useModelWithParams(modelFunction, X, y, scaling, params):
    '''
    combine all possible parameters, put each combination to the modelFunction and save result
    :param modelFunction: function for data model
    :param X: feature dataframe
    :param y: target variable vector where y.shape = X.shape[1]
    :param scaling: scalertype for X scaling, can be on of [False, 'MinMax', 'Robust', 'Standart']
    :param params: tuple of list with parameters values, eg. ([0.01, 0.03, 0.1], [1, 3, 10], [2, 5, 10])
    :return: dict where key is tuple of parameters for model combination and value is model accuracy
    example: useModelWithParams(supportVectorClassifier, X, y, (CONST_C, CONST_gamma))
    '''
    returnByParam = {}
    for combination in itertools.product(*params):
        returnByParam[combination] = useModel(modelFunction, X, y, scaling, False, *combination)
    best_key = max(returnByParam.keys(), key=(lambda key: returnByParam[key]))
    print('key {} value {}'.format(best_key, returnByParam[best_key]))
    return returnByParam

if __name__ == "__main__":
    MODELS = [(logisticRegression, 1), (linearSVC, 1), (kNeighborsClassification, 5), (naiveBayesClassification,),
            (decisionTreeClassification, 4), (ensembleRandomForestClassifier, 8, 58),
            (gradientBoostingClassifier, 1, 0.01, 100), (supportVectorClassifier, 0.3, 3), (mLPClassifier, [10, 10], 3),
            (adaBoostClassifier, 10, 0.01)]

    MOD_BAL = [(logisticRegression, 30), (linearSVC, 300), (kNeighborsClassification, 210), (naiveBayesClassification,),
              (decisionTreeClassification, 19), (ensembleRandomForestClassifier, 1, 23),
              (gradientBoostingClassifier, 1, 3, 11), (supportVectorClassifier, 3, 30), (mLPClassifier, [50], 0.3),
              (mLPClassifier, [50, 40], 0.01), (mLPClassifier, [50, 40, 10], 0.01), (adaBoostClassifier, 10, 0.01)]

    X = Features(pd.read_csv('feature_vectors/huge/FV_familiarity.csv', index_col=0)).dataframe.iloc[:, :-1]
    # y = np.array(pd.read_csv('feature_vectors/huge/YV_familiarity.csv', index_col=0).astype(int)).ravel()

    # X = Features(pd.read_csv('feature_vectors/4/FV_all_f.csv', index_col=0)).dataframe
    y = np.array(pd.read_csv('feature_vectors/huge/YV_familiarity_binary.csv', index_col=0)).ravel()
    X_bal, y_bal = classBalancing(X, y)

    # print('LinearRegression model score: {}'.format(useLinearRegressionModel(X, y)))
    # print('RidgeRegression model score: {}'.format(useRidgeRegressionModel(X, y)))
    # print('LassoRegression model score: {}'.format(useLassoRegressionModel(X, y)))
    # print(useModelWithParams(MODELS[9][0], X, y, 'MinMax', (range(1,100,10), CONST_alpha)))
    # print(useModelWithParams(MOD_BAL[6][0], X_bal, y_bal, 'MinMax', (range(1, 6), CONST_alpha, range(1, 101, 10))))
    # print(useModelWithParams(MOD_BAL[7][0], X_bal, y_bal, 'MinMax', (CONST_C, CONST_gamma)))
    max = 0
    for trt in CONST_mLP2:
        for alpha in CONST_alpha:
            if useModel(MOD_BAL[8][0], X_bal, y_bal, 'MinMax', False, trt, alpha) > max:
                max = useModel(MOD_BAL[8][0], X_bal, y_bal, 'MinMax', False, trt, alpha)
                print('{} with alpha = {} :'.format(trt, alpha))
                print(useModel(MOD_BAL[8][0], X_bal, y_bal, 'MinMax', False, trt, alpha))

    #     print(useModelWithParams(MOD_BAL[8][0], X_bal, y_bal, 'MinMax', ([10, 20], CONST_alpha)))
    # print(CONST_mLP3)

    # model_result
    # for n in range(10):


    # print('adaBoostClassifier model with scaling accuracy: {}'.format(useModel(MODELS[9][0], X, y, 'MinMax', False, *MODELS[9][1:])))

    # print('\nMinMax scaling:')
    # for model in MODELS:
    #     print('{} model with parameters {} score: {}'.format(model[0].__name__,
    #                                                          ['{}={}'.format(name, value) for (name, value) in
    #                                                           zip(list(signature(model[0]).parameters.keys())[4:],
    #                                                               model[1:])],
    #                                                          useModel(model[0], X, y, 'MinMax', False, *model[1:])))

    # print('\nMinMax scaling:')
    # for model in MOD_BAL:
    #     print('{} model with parameters {} score: {}'.format(model[0].__name__,
    #                                                          ['{}={}'.format(name, value) for (name, value) in
    #                                                           zip(list(signature(model[0]).parameters.keys())[4:],
    #                                                               model[1:])],
    #                                                          useModel(model[0], X_bal, y_bal, 'MinMax', False, *model[1:])))




