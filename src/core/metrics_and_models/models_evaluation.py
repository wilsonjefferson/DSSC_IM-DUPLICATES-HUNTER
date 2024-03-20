import pandas as pd
import numpy as np
import logging
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import log_loss
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_recall_fscore_support as score

from scipy import stats
from collections import OrderedDict

log = logging.getLogger('modelEvalLogger')


def baseline(model_type, params:dict, model_name:str, X:np.array, y:np.array) -> dict:
    '''
        The purpose of this method is to run the baseline model, no fine-tuning
        is performed neither KFold CV.

        Parameters
        ----------
        model_type: sklearn model
            Binary decision Model

        model_name: str
            Name of the model (BDT, BDF, ...)

        metrics: list
            List of metrices to evaluate under importance feature parameter

        X: np.array
            Matrix of metrices score

        y: np.array
            Array of class duplication (0 = No duplication, 1 = Yes Duplication)

        Returns
        -------
        model_perf_dict: dict
            Dictionary of model information
    '''

    model_perf_dict = {'Model':None, 'Model Type':None, 'Accuracy':None, 'Train_Log_Loss':None,
            'Validation_Log_Loss':None, 'Precision':None, 'Recall':None, 'F1':None, 'F2':None, 'ROC-AUC':None, 'MCC':None}
    log.debug('Baseline - split daataset in train/test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

    try:
        model = model_type() if params is None else model_type(**params)
    except Exception as e:
        log.exception(f'Class Name: {model_name}')
        raise e

    log.debug('Baseline - train model')
    model.fit(X_train, y_train)
    log.debug('Baseline - evaluate model')
    model_eval = eval(model, model_name, X_train, y_train, X_test, y_test)
    
    for k, v in zip(model_perf_dict.keys(), model_eval):
        model_perf_dict[k] = v

    return model_perf_dict

def k_fold(model_type, params:dict, model_name:str, X:np.array, y:np.array, dict_average_train:dict,
        n_splits:int = 10, n_repeats:int = 5) -> tuple:
    '''
        The purpose of this method is to run the KFold CV model, no fine-tuning.
        This is done to ensure a robust model.

        Parameters
        ----------
        model_type: sklearn model
            Binary decision Model

        model_name: str
            Name of the model (BDT, BDF, ...)

        metrics: list
            List of metrices to evaluate under importance feature parameter

        X: np.array
            Matrix of metrices score

        y: np.array
            Array of class duplication (0 = No duplication, 1 = Yes Duplication)

        dict_average_train: dict
            Dictionary of averaged eval values (i.e. accuracy, recall, precision, ...)

        n_splits: int
            Number of folds to split the trainset

        n_repeats: int
            Number of times to repeat KFold CV


        Returns
        -------
            tuple
                Best model from KFold CV and KFold CV history
    '''

    model_perf_dict = {'Model': [], 'Model Type': [], 'Accuracy': [], 'Train_Log_Loss': [],
            'Validation_Log_Loss': [], 'Precision': [], 'Recall': [], 'F1': [], 'F2':[], 'ROC-AUC': [], 'MCC':[]}
    
    log.debug('KFold Repetead %s'%n_repeats)
    model_base = model_type() if params is None else model_type(**params)
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)
    # RepeatedKFold provides indexes for train and validation sets
    for train_idx, valid_idx in rkf.split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        X_valid, y_valid = X[valid_idx], y[valid_idx]

        model = deepcopy(model_base) # in K Fold CV we need to train a new model each time
        model.fit(X_train, y_train)
        model_eval = eval(model, model_name, X_train, y_train, X_valid, y_valid)

        for k, v in list(zip(model_perf_dict.keys(), model_eval)):
            model_perf_dict[k].append(v)

    df_model = pd.DataFrame.from_dict(model_perf_dict)
    df_model.sort_values('Accuracy', ascending=False, ignore_index=True, inplace=True)

    # compute average on model eval values (accuracy, precision, recall, ...)
    log.debug('Compute average statistics')
    df_kflod_stats = average_eval(df_model.iloc[:, 2:], model_name)
    dict_average_train[model_name] = df_kflod_stats

    return df_model.iat[0, 0], df_model

def average_eval(dict_eval:dict, model_name:str) -> pd.DataFrame:
    df_kflod_stats = {}
    df_kflod_stats[model_name+"_mean"] = dict_eval.mean()
    df_kflod_stats[model_name+"_std.dev"] = dict_eval.std()
    df_kflod_stats[model_name+"_var"] = dict_eval.var()
    return pd.DataFrame.from_dict(df_kflod_stats)

def eval(model, model_name:str, X_train:np.array, y_train:np.array, 
        X_test:np.array, y_test:np.array) -> tuple:
    '''
        The purpose of this method is to evaluate the model on the testset.

        Parameters
        ----------
        model: sklearn model
            Binary decision Model

        model_name: str
            Name of the model (BDT, BDF, ...)

        metrics: list
            List of metrices to evaluate under importance feature parameter

        X_train: np.array
            Matrix of metrices score

        y_train: np.array
            Array of class duplication (0 = No duplication, 1 = Yes Duplication)

        X_test: np.array
            Matrix of metrices score

        y_test: np.array
            Array of test class duplication (0 = No duplication, 1 = Yes Duplication)

        Returns
        -------
            tuple
                Model information
    '''

    y_pred = model.predict(X_test)
    y_train_proba = model.predict_proba(X_train)[:,1]
    y_test_proba = model.predict_proba(X_test)[:,1]

    accuracy = accuracy_score(y_test, y_pred)*100

    train_log_loss = log_loss(y_train, y_train_proba)
    test_log_loss = log_loss(y_test, y_test_proba)

    precision, recall, f1_score, _ = score(y_test, y_pred, average='macro')

    f2_score = fbeta_score(y_test, y_pred, beta=2)
    r_a_score = roc_auc_score(y_train, y_train_proba)
    mcc_score = matthews_corrcoef(y_test, y_pred)

    return (model, model_name, accuracy, train_log_loss, test_log_loss, \
                precision, recall, f1_score, f2_score, r_a_score, mcc_score, y_pred)

def metrics_imp(model, model_name:str, X_train:np.array, y_train:np.array, metrics:list) -> pd.DataFrame:
    '''
        The purpose of this method is to determine the importances of the metrices.

        Parameters
        ----------
        model: sklearn model
            Binary decision Model

        model_type: str
            Name of the model (BDT, BDF, ...)

        metrics: list
            List of metrices to evaluate under importance feature parameter

        X: np.array
            Matrix of metrices score

        Returns
        -------
        df_metrics_imp: pd.Dataframe
            Ordered Dataframe with the importances score for each metrices
    '''

    if model_name == 'LRC':
        imp = [round(np.absolute(n), 3) for n in np.std(X_train, 0)*model.coef_[0]]
    elif model_name == 'SVM':
        from sklearn.inspection import permutation_importance
        result = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42)
        imp = result.importances_mean
    else:
        imp = [round(n, 3) for n in model.feature_importances_]

    df_metrics_imp = pd.DataFrame.from_dict({
            'Model Type': [model_name for _ in range(len(metrics))],
            'Metrics':metrics, 
            'Importance':imp})
    df_metrics_imp.sort_values('Importance', ascending=False, 
            ignore_index=True, inplace=True)
    return df_metrics_imp

def features_correlation_scores(X:np.array, y:np.array, metrics:list) -> pd.DataFrame:
    '''
        The purpose of this method is to compute the Pearson and Spearman
        correlations among the metrices and the duplication class.

        Parameters
        ----------
        X: np.array
            Matrix of metrices score
        
        y: np.array
            Array of class duplication (0 = No duplication, 1 = Yes Duplication)

        metrics:
            List of metrics

        Returns
        -------
            pd.Dataframe
                DataFrame with the measures of correlations
    '''

    log.debug('Compute correlation with Person and Sperman correlation scores')
    pers, spear = {}, {}
    for idx, m in enumerate(metrics):
        pers[m] = stats.pearsonr(X[:, idx], y) # pearson correlation score
        spear[m] = stats.spearmanr(X[:, idx], y) # spearman correlation score
    
    pers = OrderedDict(sorted(pers.items(), key=lambda kv: kv[1], reverse=True))
    spear = OrderedDict(sorted(spear.items(), key=lambda kv: kv[1], reverse=True))

    fcs = {'Metrics': [], 'Pearson Coeff.': [], 'Pearson p-value': [], 'Spearman Coeff.': [], 'Spearman p-value': []}
    for k in metrics:
        fcs['Metrics'].append(k)
        fcs['Pearson Coeff.'].append(round(pers[k][0], 2))
        fcs['Pearson p-value'].append(round(pers[k][1], 3))
        fcs['Spearman Coeff.'].append(round(spear[k][0], 2))
        fcs['Spearman p-value'].append(round(spear[k][1], 3))
    
    return pd.DataFrame.from_dict(fcs)
