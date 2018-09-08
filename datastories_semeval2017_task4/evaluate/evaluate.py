"""
Created by Florian Fricke.
"""

from sklearn import metrics
import matplotlib.pyplot as plt
import itertools
import numpy as np

def performance_analysis(testing, model, file_name="", file_information="", verbose=True, accuracy=True, confusion_matrix=True, plotting_confusion_matrix=True, classification_report=True, **kwargs):
    X_test_unprocessed = kwargs.get('X_test_unprocessed', False)
    
    with open('results_artificial_neural_network/{}.txt'.format(file_name), 'w') as f:
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

            if(plotting_confusion_matrix):
                import matplotlib as mpl
                mpl.rcParams.update(mpl.rcParamsDefault)
                from matplotlib import rcParams
                rcParams.update({'figure.autolayout': True})

                # Plot non-normalized confusion matrix
                plt.figure(dpi=600)
                plot_confusion_matrix(confusion_matrix, classes=['positiv', 'neutral', 'negativ'],
                                        title='Wahrheitsmatrix')
                plt.savefig(
                    'results_artificial_neural_network/{}_confusion_matrix_1.png'.format(file_name))

                # Plot normalized confusion matrix
                plt.figure(dpi=600)
                plot_confusion_matrix(confusion_matrix, classes=['positiv', 'neutral', 'negativ'], normalize=True,
                                        title='normalisierte Wahrheitsmatrix')
                plt.savefig(
                    'results_artificial_neural_network/{}_confusion_matrix_2.png'.format(file_name))


        if(classification_report):
            classification_report = metrics.classification_report(
                testing[1].argmax(axis=1), y_pred)
            metric_list.append(classification_report)
            if(verbose):
                print("\nclassification report:", file=f)
                print(classification_report, file=f)

        if(kwargs.get('save_pred', False) and X_test_unprocessed):
            with open('results_artificial_neural_network/predictions/{}_predictions.txt'.format(file_name), 'w', encoding="utf-8") as f:
                for i in range(len(y_pred)):
                    if y_pred[i] == 0:
                        pred = "negative"
                    elif y_pred[i] == 1:
                        pred = "neutral"
                    elif y_pred[i] == 2:
                        pred = "positive"
                    else:
                        pred=""
                    print("{}\t{}".format(
                        pred, X_test_unprocessed[i]), file=f)
    return metric_list

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Wahrheitsmatrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Tatsächliche Klasse')
    plt.xlabel('Vorhergesagte Klasse')
