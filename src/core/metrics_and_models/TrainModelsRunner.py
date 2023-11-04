import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import dump
import logging
from copy import deepcopy

from sklearn.model_selection import train_test_split
from typing import Callable, List

log = logging.getLogger('modelRunnerLogger')

from ConfigNameSpace import MAIN_STAGE
make_plots = MAIN_STAGE.make_plots
backup_location = MAIN_STAGE.backup_location

from core.metrics_and_models.models_evaluation import baseline, k_fold, metrics_imp, eval


class TrainModelsRunner:

    def __init__(self, dict_models:dict, df:pd.DataFrame, metrics:List[str], X:np.array, y:np.array, plot_name:str = '') -> None:

        self.plot_name = plot_name
        self.df = df
        self.metrics = metrics

        self.X = X
        self.y = y

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.20, random_state=42)

        self.dict_models = dict_models
        self.model_testing_dict = {'Model':[], 'Model Type':[], 'Accuracy':[], 'Train_Log_Loss':[],
            'Validation_Log_Loss':[], 'Precision':[], 'Recall':[], 'F1':[], 'F2':[], 'ROC-AUC':[], 'MCC':[]}
        self.dict_models_history = {}
        self.dict_average_train = {}
        self.dict_best_models = {}
        self.best_models_info_list = []
        self.clfs_y_pred = {}

        self.df_average_eval = None
        self.df_imp_metrics = None
        self.df_model_testing = None
        self.best_model = None

    def decorator_check_best_model(func:Callable):
        '''
            This decorator is used to check if the self.best_model exist and it 
            is different to None.

            Parameters
            ----------      
            func: object
                Class function

            Returns
            -------
            wrapper: object
                Nested function, it return the call to the function
        '''

        def wrapper(self, **args):
            if self.best_model is None:
                raise ValueError('Best model is None, execute self.fit method before.')
            return func(self, **args)
        return wrapper

    def decorator_check_clfs_y_pred(func:Callable):
        '''
            This decorator is used to check if the self.clfs_y_pred exist and it 
            is different to None.

            Parameters
            ----------      
            func: object
                Class function

            Returns
            -------
            wrapper: object
                Nested function, it return the call to the function
        '''
        def wrapper(self):
            if self.clfs_y_pred is None:
                raise ValueError('clfs_y_pred is None, execute self.fit method before.')
            return func(self)
        return wrapper

    def fit(self) -> tuple:
        '''
            Ths method is used to train the models, according the K-Fold cross validation
            and retrieve the best possible model.

            Parameters
            ----------      
            None

            Returns
            -------
            self.best_model: object
                best model according the train/test process
        '''

        metrics_imp_list = []
        for name, tmp in tqdm(self.dict_models.items(), desc='Fitting models'):
            model_type, params, metrics_idx, _ = tmp
            log.debug(f'model name: {name}\nmetrics: {metrics_idx}')
            metrics = [self.metrics[idx] for idx in metrics_idx]
            X_train = self.X_train[:, metrics_idx]
            X_test = self.X_test[:, metrics_idx]

            y_pred, df_metrics_imp = self._run(model_type, params, name,  metrics, X_train, X_test)
            
            self.clfs_y_pred[name] = y_pred
            log.debug(f'clfs_y_pred[{name}] = {y_pred}')

            metrics_imp_list.append(df_metrics_imp)
            log.debug(f'metrics_imp_list: {metrics_imp_list}')

        # concat average eval from different models in one bigger dataframe
        df_average_eval = pd.concat(self.dict_average_train.values(), axis=1).T
        df_average_eval['Model Type'] = df_average_eval.index
        df_average_eval_col = [df_average_eval.columns[-1]] + list(df_average_eval.columns[:-1])
        self.df_average_eval = df_average_eval[df_average_eval_col]

        # merge in a single dataframe all the metrics importancy dataframes from each model
        self.df_imp_metrics = pd.concat(metrics_imp_list)

        # retrieve the best model according the average KFold accuracy
        best_model_name = df_average_eval.sort_values('Accuracy', ascending=False)
        best_model_name.reset_index(inplace=True, drop=True)
        best_model_name = best_model_name.at[0, 'Model Type']
        best_model_name = best_model_name.split(' ')[0]

        best_model_type = self.dict_models[best_model_name][0]
        params = self.dict_models[best_model_name][1]

        best_model = best_model_type(**params)
        best_model.fit(self.X, self.y)

        self.best_model = {
            'model_name': best_model_name, 
            'clf': best_model, 
            'hyperparams': params, 
            'threshold': None}
        
        return self.best_model

    def _run(self, model_type, params:dict, model_name:str, metrics:List[str], X_train:np.array, X_test:np.array) -> tuple:
        '''
            This is a support method to train models, here are computed the models baselines, 
            the kfold cross validationa and metrics importances.

            Parameters
            ----------
            model_type: object
                model to train and test
            
            model_name: str
                model name
            
            metrics: list
                List of metrics to use to determine the importances

            X_train: np.array
                Raw data array composed by the scores from the metrics

            X_test: np.array
                Raw data array composed by the scores from the metrics

            Returns
            -------
            y_pred: list
                prediction from the give model on unseen data
        '''

        log.debug('Run %s baseline'%model_name)
        dict_baseline = baseline(model_type, params, model_name + ' (Baseline)', X_train, self.y_train)

        log.debug('Run %s kfold'%model_name)
        best_model, df_model = k_fold(model_type, params, model_name + ' (KFold)', X_train, self.y_train, self.dict_average_train)
        self.dict_best_models[model_name] = best_model

        # compute metrics importances
        log.debug('Compute %s metrics importances'%model_name)
        if model_name != 'NGB':
            df_metrics_imp = metrics_imp(best_model, model_name, X_train, self.y_train, metrics)
        else:
            df_metrics_imp = None

        self.best_models_info_list.append(dict_baseline)
        self.dict_models_history[model_name] = df_model
        self.best_models_info_list.append(df_model.iloc[0, :])
        
        # evaluate best model retrieved by KFold CV
        log.debug('Evaluate best model %s'%model_name)
        model = model_type() if params is None else model_type(**params)
        model.fit(X_train, self.y_train)
        model_eval = eval(model, model_name + ' (KFold)', 
                X_train, self.y_train, X_test, self.y_test)
        y_pred, model_eval = model_eval[-1], model_eval[:len(model_eval)-1]

        for k, v in list(zip(self.model_testing_dict.keys(), model_eval)):
            self.model_testing_dict[k].append(v)

        if make_plots:
            from core.utils.plots.models_plots import plot_model_metrics_summary

            # plot model metrics summary from estimators
            model_type = self.dict_models[model_name][0]
            params = self.dict_models[model_name][1]
            model = model_type() if params is None else model_type(**params)

            plot_model_metrics_summary(name=model_name, 
                                        y_test=self.y_test,
                                        clf=model, 
                                        X_train=self.X_train, 
                                        y_train=self.y_train, 
                                        X_test=self.X_test, 
                                        custom_name_png=self.plot_name)

        return y_pred, df_metrics_imp

    def optimize(self) -> tuple:
        """
        Perform Bayesian optimization for hyperparameter tuning using scikit-optimize (skopt).

        Returns:
            best_model: dict
                A dictionary containing information about the best model:
                - 'model_name': Name of the best model.
                - 'clf': Best classifier instance with optimized hyperparameters.
                - 'hyperparams': Dictionary of the best hyperparameters.
                - 'threshold': Decision threshold optimized for F1-Score.
        """
        from skopt import gp_minimize
        from skopt.utils import use_named_args
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import f1_score, make_scorer, precision_recall_curve
        import numpy as np

        model_name = self.best_model['model_name']
        model_type = self.dict_models[model_name][0]
        param_space = self.dict_models[model_name][3]
        
        if param_space is None:
            return None
                    
        log.info(f'best model type: {model_name}')

        @use_named_args(param_space)
        def objective_function(**params) -> float:

            clf = model_type(**params, random_state=42)
            scorer = make_scorer(f1_score, greater_is_better=True)
            f1 = np.mean(cross_val_score(clf, self.X_train, self.y_train, cv=5, scoring=scorer, n_jobs=-1))

            # The objective is to maximize the F1-Score (minimize the negative F1-Score)
            return -f1

        self.gp_opt = gp_minimize(objective_function, param_space, n_calls=100, n_random_starts=10, random_state=42)
        best_params = {param.name:self.gp_opt.x[idx] for idx, param in enumerate(param_space)}

        str_best_params = ''
        for param, val in best_params.items():
            str_best_params += f'{param} = {val}\n'
        
        log.info("Best score = %.4f" % self.gp_opt.fun)
        log.info(str_best_params)

        ### assess on performance improvement comparing with baseline ###
        # train default model
        default_params = self.dict_models[model_name][1]
        default_clf = model_type(**default_params, random_state=42)
        default_clf.fit(self.X_train, self.y_train)

        # make predictions on the test set
        y_pred_test = default_clf.predict(self.X_test)

        # evaluate the model on the test set
        f1_test = f1_score(self.y_test, y_pred_test)
        log.info(f"Default model - F1_Score on Test Set: {f1_test}")

        # retrain the model with the best hyperparameters
        best_clf = model_type(**best_params, random_state=42)
        best_clf.fit(self.X_train, self.y_train)

        # make predictions on the test set
        y_pred_test = best_clf.predict(self.X_test)

        # mvaluate the model on the test set
        f1_test = f1_score(self.y_test, y_pred_test)
        log.info(f"Bayesian Optimization - F1_Score on Test Set: {f1_test}")
        #################################################################

        best_clf = model_type(**best_params, random_state=42)
        best_clf.fit(self.X, self.y)
        
        self.best_model = {
            'model_name': model_name, 
            'clf': best_clf, 
            'hyperparams': best_params}
        
        return self.best_model

    @decorator_check_clfs_y_pred
    def plots(self) -> None:
        '''
            This method is used to plot some statistics during the train/test process.
            In particular, the models are compared using the ROC Curve, the Detection
            Error Tradeoff. Lastly, it is plotted the surface boudary from the best model, 
            to compare pairs of metrics.

            Parameters
            ----------      
            None

            Returns
            -------
            None
        '''
        from sklearn.decomposition import PCA
        from core.utils.plots.models_plots import (
            plot_models_decision_regions,
            plot_model_comparison, 
            plot_boundary_surface,
            plot_model_log_loss)
        
        pca = PCA(n_components = 2)
        X_train_pca = pca.fit_transform(self.X_train)

        clfs ={}
        for name, tmp in self.dict_models.items():
            model_type, params = tmp[0], tmp[1]
            clf = model_type() if params is None else model_type(**params)
            clfs[name] = clf

        plot_models_decision_regions(clfs, X_train_pca, self.y_train, self.plot_name)

        # plot model comparison from estimators
        clfs ={}
        for name, tmp in self.dict_models.items():
            model_type, params = tmp[0], tmp[1]
            clf = model_type() if params is None else model_type(**params)
            clfs[name] = clf
        
        plot_model_comparison(y_test=self.y_test, 
                                clfs=clfs,
                                X_train=self.X_train,
                                y_train=self.y_train,
                                X_test=self.X_test,
                                custom_name_png=self.plot_name)

        # plot decision boundary for the each model
        for name, model_tuple in self.dict_models.items():
            model_type, params = tmp[0], tmp[1]
            model = model_type() if params is None else model_type(**params)

            model_train_loss = self.dict_models_history[name]['Train_Log_Loss'].tolist()
            model_test_loss = self.dict_models_history[name]['Validation_Log_Loss'].tolist()
            plot_model_log_loss(name, train_loss=model_train_loss, test_loss=model_test_loss)
                                    
            plot_boundary_surface(name=name,
                                    clf=model,
                                    X=self.X_train,
                                    y=self.y_train, 
                                    labels=self.metrics,
                                    custom_name_png=self.plot_name)

        if not (self.gp_opt is None):
            from core.utils.plots.models_plots import plot_post_hypertuning
            plot_post_hypertuning(self.gp_opt)

    @decorator_check_best_model
    def store(self, location:str) -> None:
        '''
            This method store the best model in memory and also some useful information.

            Parameters
            ----------      
            location: str
                Location where save the best model and various information

            Returns
            -------
            None
        '''

        for name, model in self.dict_best_models.items():
            dump(model, location+f'models/model_{name}.joblib')

        self.df_model_testing = pd.DataFrame.from_dict(self.model_testing_dict)
        
        with pd.ExcelWriter(location + 'train_model_results.xlsx', engine='openpyxl') as writer:
            self.df_average_eval.to_excel(writer, sheet_name='Train - Average', index=False)
            self.df_model_testing.to_excel(writer, sheet_name='Test - Best Performance', index = False)
            self.df_imp_metrics.to_excel(writer, sheet_name='Features Selection', index = False)
            
            y_idx_failed = self.predict()
            self.df.iloc[y_idx_failed, :].to_excel(writer, 
                sheet_name='Wrong test - ' + \
                self.best_model['model_name'], index = False)

            for model_name, df in self.dict_models_history.items():
                df.to_excel(writer, sheet_name= model_name + ' - Fold History', index = False)

    @decorator_check_best_model
    def predict(self, X_test:np.array = None, y_test:np.array = None) -> np.array:
        '''
            This is a support method to train models, here are computed the models baselines, 
            the kfold cross validationa and metrics importances.

            Parameters
            ----------
            X_test: np.array
                Raw data array composed by the scores from the metrics

            y_test: np.array
                Raw data of the duplication

            Returns
            -------
            y_idx_failed: np.array
                Failed prediction from the give model on unseen data
        '''

        if not X_test and not y_test:
            X_test = self.X_test
            y_test = self.y_test

        # run prediction with the best possible model 
        # and retrieve the missclassified records
        model = self.best_model['clf']

        y_pred = model.predict(X_test)
        y_idx_failed = np.where(y_pred != y_test)[0] 

        return y_idx_failed
