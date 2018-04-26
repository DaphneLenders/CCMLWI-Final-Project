from ExtractFeatures import extract_features
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE
from Classify_utils import eight_labels, four_labels, evaluate, average_scores, makeOverview
from InverseDocumentFrequentizer import idf_vectorizer, generate_stemmed_bigrams, generate_bigrams, generate_trigrams, generate_character_trigrams, generate_functionword_trigrams
import Classifier_adaBoost
import Classifier_randomForest
import Classifier_Bayes
import Classifier_SVM
import Classifier_majority
import Classifier_logisticRegression
import Classifier_ecoc_svm


def classify(folds, number_of_classes, classifier):
    scores = []
    for (train_data, train_labels, validation_data, validation_labels) in folds:

        idf_vectorizer(train_data)
        generate_stemmed_bigrams(train_data)
        print(generate_functionword_trigrams(train_data))
        #print(calc_document_frequencies_stemmed_words(train_data))

        # Features are extracted
        feature_matrix_train = list(map(extract_features, train_data))
        feature_matrix_validation = list(map(extract_features, validation_data))

        # Select features with enough variance
        #sel = VarianceThreshold(threshold=0.3)
        #sel = sel.fit(feature_matrix_train)
        #feature_matrix_train = sel.transform(feature_matrix_train)
        #feature_matrix_validation = sel.transform(feature_matrix_validation)


        # Oversampling is used to balance out dataset
        #feature_matrix_train, train_labels = SMOTE().fit_sample(feature_matrix_train, train_labels)

        validation_prediction = classifier.fit_and_predict(feature_matrix_train, train_labels, feature_matrix_validation)


        if number_of_classes == 4:
            print(confusion_matrix(validation_labels, validation_prediction, labels=['SAMEN-BOVEN', 'SAMEN-ONDER', 'TEGEN-ONDER', 'TEGEN-BOVEN']))

        elif number_of_classes == 8:
            print(confusion_matrix(validation_labels, validation_prediction, labels=['LEIDEND', 'HELPEND', 'MEEWERKEND', 'VOLGEND', 'TERUGGETROKKEN', 'OPSTANDIG', 'AANVALLEND', 'COMPETITIEF']))


        score = evaluate(validation_labels, validation_prediction)
        scores.append(score)

    average_scores(scores)
    return average_scores

def compare_performances():
    folds = eight_labels()
    for clf, label in zip([Classifier_randomForest, Classifier_adaBoost, Classifier_logisticRegression, Classifier_Bayes, Classifier_SVM, Classifier_majority],
                          ['Random Forest', 'Ada boost', 'Logistic Regression', 'Naive Bayes', 'Support Vector Machine',
                           'Majority vote']):
        print(label)
        classify(folds, 8, clf)
        print("******************")


def main():
    folds = eight_labels()
    classify(folds, 8, Classifier_ecoc_svm)




if __name__ == '__main__':
    main()