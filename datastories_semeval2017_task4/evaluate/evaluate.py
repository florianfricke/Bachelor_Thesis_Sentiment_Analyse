"""
Created by Florian Fricke.
"""

from sklearn import metrics


def performance_analysis(testing, model, file_name="", file_information="", verbose=True, accuracy=True, confusion_matrix=True, classification_report=True):
    with open('results_artificial_neural_network/evaluation_{}.txt'.format(file_name), 'w') as f:
        print(file_information, file=f)
        y_pred = model.predict(testing[0])
        y_pred = y_pred.argmax(axis=1)
        metric_list = list()
        if(accuracy):
            accuracy = model.evaluate(testing[0], testing[1])
            metric_list.append(accuracy)
            if(verbose):
                print("\nmodel accuracy (loss value, accuracy):", file=f)
                print(accuracy, file=f)

        if(confusion_matrix):
            confusion_matrix = metrics.confusion_matrix(testing[1].argmax(
                axis=1), y_pred)
            metric_list.append(confusion_matrix)
            if(verbose):
                print("\nconfusion matrix:", file=f)
                print(confusion_matrix, file=f)

        if(classification_report):
            classification_report = metrics.classification_report(
                testing[1].argmax(axis=1), y_pred)
            metric_list.append(classification_report)
            if(verbose):
                print("\nclassification report:", file=f)
                print(classification_report, file=f)
    return metric_list
