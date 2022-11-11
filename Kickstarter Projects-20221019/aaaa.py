def multiclassROC(model, x_test=data.X_test, y_test=data.Y_test, ax=None):
    y = label_binarize(y_test, classes=[0, 1, 2, 3])
    n_classes = y.shape[1]
    pred = model.predict_proba(x_test)
    fpr, roc_auc, tpr = method_name(n_classes, pred, y)
    # Plot ROC curve
    if ax != None:
        ax.plot(fpr["micro"], tpr["micro"],
                label='micro-average ROC curve (area = {0:0.2f})'
                      ''.format(roc_auc["micro"]))
        for i in range(n_classes):
            ax.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                          ''.format(i, roc_auc[i]))

        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {model.__class__.__name__}')
        ax.legend(loc="lower right")
    else:
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]))
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                           ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model.__class__.__name__}')
        plt.legend(loc="lower right")
        plt.show()


def method_name(n_classes, pred, y):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return fpr, roc_auc, tpr

