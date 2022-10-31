import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 50, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd, flush=True)
    # Print New Line on Complete
    if iteration == total:
        print()
        
def printConfusionMatrix (y_true, y_pred):
    """
    Print the confusion matrix
    @params
        y_true  - true values
        y_pred  - predicted values
    """
    conf_stat = confusion_matrix(y_true=y_true, y_pred=y_pred)
    fig, ax = plt.subplots(figsize=(5,5), tight_layout=True)
    sns.heatmap(conf_stat, annot=True, fmt=".3f", 
            linewidths=.5, square = True, 
            cmap = 'Blues_r',cbar=False);
    ax.set_ylabel('True Label', fontsize=14);
    ax.set_xlabel('Predicted Label', fontsize=14);
    