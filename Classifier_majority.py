from sklearn.ensemble import VotingClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd

def fit_and_predict(feature_matrix_train, train_labels, feature_matrix_validation):
    clf1 = LogisticRegression(random_state = 1)
    clf2 = GaussianNB()
    clf3 = svm.SVC(kernel='linear')
    clf4 = RandomForestClassifier(random_state=1)
    clf5 = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=0)

    majority_classifier = VotingClassifier(estimators=[('lr', clf1), ('nb', clf2), ('svm', clf3), ('rfc', clf4), ('ada', clf5)], voting='hard')

    majority_classifier.fit(feature_matrix_train, train_labels)
    validation_prediction = majority_classifier.predict(feature_matrix_validation)
    return validation_prediction


def compare_performances():
    data = pd.read_excel('translated_texts.xlsx')
    class_labels = data.label
    texts = data.text


