from sklearn.linear_model import LogisticRegression

def fit_and_predict(feature_matrix_train, train_labels, feature_matrix_validation):
    lr = LogisticRegression(random_state = 1)
    lr.fit(feature_matrix_train, train_labels)
    validation_prediction = lr.predict(feature_matrix_validation)
    return validation_prediction