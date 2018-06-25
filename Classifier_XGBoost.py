from xgboost import XGBClassifier

from numpy import array
from numpy import argmax
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def fit_and_predict(feature_matrix_train, train_labels, feature_matrix_validation):
    model = XGBClassifier()
    model.fit(feature_matrix_train, train_labels)
    validation_prediction = model.predict(feature_matrix_validation)

    print(model.feature_importances_)
    # plot
    #pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
    #pyplot.show()

    return validation_prediction