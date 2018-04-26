from sklearn.ensemble import AdaBoostClassifier


def fit_and_predict(feature_matrix_train, train_labels, feature_matrix_validation):
    ada = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=1)
    ada.fit(feature_matrix_train, train_labels)
    validation_prediction = ada.predict(feature_matrix_validation)
    return validation_prediction
