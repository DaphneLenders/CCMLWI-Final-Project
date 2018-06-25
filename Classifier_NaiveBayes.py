from sklearn.naive_bayes import GaussianNB

def fit_and_predict(feature_matrix_train, train_labels, feature_matrix_validation):
    # Now, do the actual classification:
    nb_classifier = GaussianNB()
    nb_classifier.fit(feature_matrix_train, train_labels)
    validation_prediction = nb_classifier.predict(feature_matrix_validation)
    return validation_prediction

