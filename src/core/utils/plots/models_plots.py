from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, product

from sklearn.metrics import DetCurveDisplay, RocCurveDisplay, \
        ConfusionMatrixDisplay, PrecisionRecallDisplay, log_loss
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import DecisionBoundaryDisplay
from mlxtend.plotting import plot_decision_regions

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ConfigNameSpace import MAIN_TRAIN, MAIN_STAGE
train_backup_location = MAIN_TRAIN.backup_location
stage_backup_location = MAIN_STAGE.backup_location
backup_location_plots = MAIN_STAGE.backup_location + '/plots/'


N_SAMPLES = 1000

savefig = lambda name: plt.savefig(name, bbox_inches='tight')

def plot_boundary_surface(name:str, clf, X:np.array, y:np.array, 
                          labels:list, custom_name_png:str = '') -> None:
    '''
        The purpose of this method is to plot the decision boundary by 
        considering pairs of metrices.

        Parameters
        ----------
        name: str
            name of the Model

        clf: sklearn model
            Binary Decision Classifier

        X: np.array
            Matrix of metrices score

        y: np.array
            Array of class duplication (0 = No duplication, 1 = Yes Duplication)

        labels: list
            List of metrices names

        custom_name_png: str
            Custom part to name the png files

        Returns
        -------
        None
    '''
    
    pairs = list(filter(lambda x: x[0] != x[1], combinations(labels, 2)))
    model = deepcopy(clf)

    n_rows, n_cols = get_fig_dimensions(len(pairs))
    _, axs = plt.subplots(n_rows, n_cols, figsize=(25, 20))

    for pair, ax in zip(pairs, axs.flatten()):
        _X = X[:, [labels.index(pair[0]), labels.index(pair[1])]]
        model = model.fit(_X, y)

        disp = DecisionBoundaryDisplay.from_estimator(
            model, _X, response_method="predict",
            xlabel=pair[0], ylabel=pair[1],
            alpha=0.5, ax=ax)
        
        disp.ax_.set_xlim(-0.05, 1.05)
        disp.ax_.set_ylim(-0.05, 1.05)

        scatter = disp.ax_.scatter( _X[:, 0],  _X[:, 1], c=y, edgecolor="k")

    disp.ax_.legend(*scatter.legend_elements(), title="Duplication")
    plt.legend()
    
    savefig(backup_location_plots+f'{custom_name_png}{name}_boundary_surface.png')
    plt.close()

def plot_model_metrics_history(name:str, m_train:list, m_val:list, m_test:list, custom_name_png:str = '') -> None:
    '''
        The purpose of this method is to plot the history if the model
        during the KFold CV.

        Parameters
        ----------
        name: str
            name of the Model

        m_train: list
            List of model train accuracy

        m_val: list
            List of model validation accuracy

        m_test: list
            List of model test accuracy

        custom_name_png: str
            Custom part to name the png files

        Returns
        -------
        None
    '''

    n_iter = range(len(m_train))
    _, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(n_iter, m_train, 'g', label='Training accuracy')
    ax.plot(n_iter, m_val, 'b', label='Validation accuracy')
    ax.plot(n_iter, m_test, 'r', label='Testing accuracy')
    ax.set_title(name + ' - Training and Validation accuracy')
    ax.set_xlabel('N. Iterations')
    ax.set_ylabel('Accuracy')
    ax.legend()

    savefig(backup_location_plots+'{}_{}_metrics_history.png'.format(custom_name_png, name))
    plt.close()

def plot_model_metrics_summary(name:str, y_test:np.array,
        clf = None, X_train:np.array = None, y_train:np.array = None, 
        X_test:np.array = None, custom_name_png:str = '') -> None:
    '''
        The purpose of this method is to plot the summary of the model
        quality analysis: Confusion Matrix, Precision-Recall curve, Calibration curve.

        Parameters
        ----------
        name: str
            name of the Model

        y_test: np.array
            Array of test class duplication (0 = No duplication, 1 = Yes Duplication)

        y_pred: list
            List of predicted class duplication (0 = No duplication, 1 = Yes Duplication)

        clf: sklearn model
            Binary Decision Classifier

        X_train: np.array
            Matrix of metrices score

        y_train: np.array
            Array of train class duplication (0 = No duplication, 1 = Yes Duplication)

        X_test: np.array
            Matrix of metrices score

        custom_name_png: str
            Custom part to name the png files

        Returns
        -------
        None
    '''

    _, axs = plt.subplots(1, 3, figsize=(20, 7))

    clf.fit(X_train, y_train)
    ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, ax=axs[0])
    PrecisionRecallDisplay.from_estimator(clf, X_test, y_test, ax=axs[1])
    CalibrationDisplay.from_estimator(clf, X_test, y_test, ax=axs[2])
    
    axs[0].set_title(name + " - Confusion Matrix")
    axs[1].set_title(name + " - Precision Recall curve")
    axs[2].set_title(name + " - Model Calibration curve")
    axs[1].grid(linestyle="--")
    axs[2].grid(linestyle="--")
    
    savefig(backup_location_plots+f'{custom_name_png}{name}_model_summary.png')
    plt.close()

def plot_model_log_loss(model_name:str, train_loss:list, test_loss:list) -> None:
    plt.plot(range(1, len(train_loss) + 1), train_loss, marker='o', markersize=8, label='Train Log-Loss')
    plt.plot(range(1, len(test_loss) + 1), test_loss, marker='o', markersize=8, label='Validation Log-Loss')
    plt.title(f'Log-Loss of {model_name} Classifier')
    plt.xlabel('Iteration')
    plt.ylabel('Log-Loss')
    plt.legend()

    savefig(backup_location_plots+f'{model_name}_model_log_loss.png')
    plt.close()

def plot_model_comparison(y_test:list, clfs = None, X_train:np.array = None, 
        y_train:np.array = None, X_test:np.array = None, custom_name_png:str = '') -> None:
    '''
        The purpose of this method is to plot the summary of the (best) model
        quality analysis: Confusion Matrix, Precision-Recall curve, Calibration curve.

        Parameters
        ----------
        y_test: np.array
            Array of test class duplication (0 = No duplication, 1 = Yes Duplication)

        clfs_y_pred: dict
            Dictionary of y_pred labels for the duplication class

        clfs: dict
            Dictionary of best models

        X_train: np.array
            Matrix of metrices score

        y_train: np.array
            Array of train class duplication (0 = No duplication, 1 = Yes Duplication)

        X_test: np.array
            Matrix of metrices score

        custom_name_png: str
            Custom part to name the png files

        Returns
        -------
        None
    '''

    _, axs = plt.subplots(1, 2, figsize=(15, 7))
    for model_name, clf in clfs.items():
        clf.fit(X_train, y_train)
        RocCurveDisplay.from_estimator(clf, X_test, y_test, ax=axs[0], name=model_name)
        DetCurveDisplay.from_estimator(clf, X_test, y_test, ax=axs[1],name=model_name)

    axs[0].set_title("Receiver Operating Characteristic (ROC) curves")
    axs[1].set_title("Detection Error Tradeoff (DET) curves")
    axs[0].grid(linestyle="--")
    axs[1].grid(linestyle="--")
    plt.legend()

    savefig(backup_location_plots+f'{custom_name_png}models_comparison.png')
    plt.close()

def plot_models_decision_regions(clfs:dict, X_train:np.array = None, \
                                 y_train:np.array = None, custom_name_png:str = ''):
    """
    Plot decision regions for multiple classifiers.

    Note:
    This function plots decision regions for each classifier in the provided dictionary
    and saves the plot to a file.

    Parameters
    ----------
    clfs: dict
        Dictionary of classifiers with labels

    X_train: np.array 
        Features for training

    y_train: np.array
        Target labels for training

    custom_name_png: str 
        Custom name for the saved plot
    """
    n_rows, n_cols = get_fig_dimensions(len(clfs))
    _, axs = plt.subplots(n_rows, n_cols, figsize=(10,8))

    prd = product(range(n_rows), range(n_cols))
    figures_content = zip(list(clfs.values()), list(clfs.keys()), prd)

    for clf, lab, grd in figures_content:
        clf.fit(X_train, y_train)
        ax = axs[grd[0], grd[1]]
        ax.set_title(lab)
        _ = plot_decision_regions(X=X_train, y=y_train, clf=clf, legend=2, ax = ax)
        plt.title(lab)
    
    savefig(backup_location_plots+f'{custom_name_png}models_decision_regions.png')
    plt.close()

def get_fig_dimensions(n_plots:int):
    """
    Calculate the number of rows and columns for subplots.

    Note:
    This function calculates the number of rows and columns for a subplot layout
    based on the total number of plots.

    Parameters
    ----------
    n_plots: int 
        Number of plots.

    Returns
    -------
        tuple: 
        Number of rows and columns for subplots.
    """

    n_rows, n_cols = 1, 1
    while n_rows*n_cols < n_plots:
        if n_rows == n_cols:
            n_cols += 1
        elif n_cols-1 == n_rows:
            n_rows += 1
        else:
            n_rows += 1
            n_cols += 1
    return n_rows, n_cols

def compare_prediction_prob_distributions(n_prod_distr:int=3) -> None:
    """
    Compare the distribution of prediction probabilities among the training set, 
    test set, and multiple product distributions.

    This function loads prediction probabilities from the training set, test set, 
    and a specified number of product distributions.
    It performs Kolmogorov-Smirnov tests and chi-squared tests to assess the 
    similarity of distributions.

    Additionally, it generates a histogram plot for visual comparison.

    Parameters
    ----------
     n_prod_distr: int
        Number of Production prediction probabilities to take into account
    """
    import os
    import joblib
    import seaborn as sns
    from scipy.stats import ks_2samp, chi2_contingency
    from itertools import product

    true_train_preds = joblib.load(train_backup_location + 'pred_probs/true_train_probs.joblib')
    true_test_preds = joblib.load(train_backup_location + 'pred_probs/true_test_probs.joblib')
    train_test_probs = {'train_probs':true_train_preds, 'test_probs':true_test_preds}

    prod_probs_path = stage_backup_location + 'pred_probs/'
    prod_probs_files = [os.path.join(prod_probs_path, file) for file in os.listdir(prod_probs_path)]
    prod_probs_files_sorted = sorted(prod_probs_files, key=os.path.getmtime, reverse=True)

    dict_n_prod_preds = {f'prod_probs_{idx}':joblib.load(f) for idx, f in \
                            enumerate(prod_probs_files_sorted[:n_prod_distr])}

    # Generate combinations using itertools.product
    combinations = product(train_test_probs.keys(), dict_n_prod_preds.keys())

    # Kolmogorov-Smirnov test
    ks_statistic, ks_p_value = ks_2samp(true_train_preds, true_test_preds)
    print(f"KS Test - Train vs Test: Statistic={ks_statistic}, p-value={ks_p_value}")

    for i, j in combinations:
        i_probs = train_test_probs.get(i, dict_n_prod_preds[i])
        j_probs = dict_n_prod_preds.get(j, train_test_probs[j])
        ks_statistic, ks_p_value = ks_2samp(i_probs, j_probs)
        print(f"KS {i} vs {j}: Statistic={ks_statistic}, p-value={ks_p_value}")

    # Chi-squared test
    for prod_n, prod_preds in dict_n_prod_preds.items():
        _, chi2_p_value, _, _ = chi2_contingency(np.vstack([true_train_preds, true_test_preds, prod_preds]))
        print(f"Chi-squared Test (Train, Test, {prod_n}): p-value={chi2_p_value}")

    # Plotting the distributions
    plt.figure(figsize=(10, 6))
    sns.histplot(true_train_preds, kde=True, label="Train Set", color='green')
    sns.histplot(true_test_preds, kde=True, label="Test Set", color='orange')

    deep_blue_palette = sns.color_palette("deep", n_colors=len(dict_n_prod_preds))
    for idx, tmp in enumerate(dict_n_prod_preds.items()):
        prod_n, prod_preds = tmp
        sns.histplot(prod_preds, kde=True, label=prod_n, color=deep_blue_palette[idx])

    plt.title("Prediction Probabilities Distribution Comparison")
    plt.xlabel("Prediction Probabilities")
    plt.ylabel("Density")
    plt.legend()
    
    savefig(backup_location_plots+f'compare_prediction_prob_distributions.png')
    plt.close()

def plot_post_hypertuning(gp_opt:object, custom_name_png:str = '') -> None:
    
    from skopt.plots import plot_convergence
    
    plot_convergence(gp_opt)
    plt.savefig(backup_location_plots+f'{custom_name_png}gaussian_convergency.png', bbox_inches='tight')
    plt.close()