import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split,learning_curve,cross_val_predict, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, get_scorer, confusion_matrix, ConfusionMatrixDisplay

from sklearn.inspection import permutation_importance

from sklearn.neural_network import MLPClassifier

class Experiment():

    X_train_raw=[]
    X_train=[]
    X_test_raw=[]
    X_test=[]
    y_train=[]
    y_test=[]

    dummy=0
    random_state=3

    feature_names=[]
    class_dist=[]

    scoring='accuracy'

    final_result=dict()

    def __init__(self,path,class_col,drop_cols=None,scoring='accuracy',random_state=3):
        """
        Initialize dataset from file path. 
        Splits into training and testing sets. 
        Drops any columns as specified to redice dimensionality.

        Parameters
        -----------
        path: path to csv data (file path or url)
        class_col: name of column with target classification (as it would appear in a pandas df)
        drop_cols(default=None): array of columns to drop
        scoring(default='accuracy'): scoring function to use
        random_state(default=3): my favorite number of the day

        Returns
        --------
        None 
        """
        df=pd.read_csv(path)

        self.class_dist=df.groupby([class_col]).agg({class_col:'count'})
        self.class_dist=self.class_dist.rename(columns={class_col:'count'})

        self.class_dist['percentage']=self.class_dist['count']/self.class_dist['count'].sum()
        print(self.class_dist)
        
        self.feature_names=df.columns.difference([class_col])
        if drop_cols:
            self.feature_names=self.feature_names.difference(drop_cols)
        y=df[class_col]
        X=df[self.feature_names]

        self.scoring=scoring
        self.random_state=random_state
        self.X_train_raw, self.X_test_raw, self.y_train, self.y_test = train_test_split(
            X.values, y.values, shuffle=True,random_state=self.random_state)

        return

    def preprocess(self,pipe):
        """
        Performs preprocessing steps on data.

        Parameters
        -----------
        pipe: Pipeline or object that can fit and transform
        
        Returns
        --------
        None
        """

        pipe.fit(self.X_train_raw)
        self.X_train=pipe.transform(self.X_train_raw)
        self.X_test=pipe.transform(self.X_test_raw)

        return

    def learning_cv(self,estimator,num_points=10,n_splits=5,plot=True):
        """
        Generates learning curve and general data for report.
        
        Parameters
        -----------
        estimator: an estimator
        num_points(default=10): Number of training sizes to test
        plot(default=True): whether or not a plot is generated

        Returns
        ---------
        train_sizes
        train_scores: averaged across kfolds
        test_scores: averaged across kfolds

        """
        try:
            estimator.set_params(**{'random_state':self.random_state})
        except ValueError as e:
            print(e)
        train_sizes, train_scores, test_scores, fit_times,score_times=learning_curve(estimator=estimator,
               X=self.X_train,
               y=self.y_train,
               train_sizes=np.linspace(0,1,num_points+1)[1:], 
               cv=StratifiedKFold(shuffle=True,random_state=self.random_state,n_splits=n_splits),
               scoring=self.scoring,
               n_jobs=-1,
               verbose=0,
               shuffle=True, 
               random_state=self.random_state,
               return_times=True,
               fit_params=None)

        y_pred=cross_val_predict(estimator=estimator,
            X=self.X_train,
            y=self.y_train,
            cv=StratifiedKFold(shuffle=True,random_state=self.random_state,n_splits=5),
            n_jobs=-1,method='predict')

        print(f'Fitting type(estimator)')
        print('--------------------------------------------------')
        print(f'Fit time: {np.mean(fit_times[-1,:]):0.3f}')
        print(f'Score time: {np.mean(score_times[-1,:]):0.3f}')
        print(f'Final scores ({self.scoring}):')
        print(f'\t Training {np.mean(train_scores[-1,:]):0.3f}')
        print(f'\t Validation {np.mean(test_scores[-1,:]):0.3f}')
        print(f'Classification report:')

        print(classification_report(self.y_train,y_pred,digits=3))
        
        ax=self.plot_learning(train_sizes,train_scores,test_scores,) if plot else None

        return {'train_sizes':train_sizes,'train_scores':train_scores,'test_scores':test_scores},ax

    def nn_learning_cv(self,nn_estimator,max_iter=200,n_splits=5,plot=True):
        nn_params=nn_estimator.get_params()
        nn_params['warm_start']=True
        nn_params['max_iter']=1
        nn_params['tol']=1e-10 #Set tolerance very low so continues to solve at higer iterations

        scoring_func=get_scorer(self.scoring)
        mlp_cv=[MLPClassifier(**nn_params) for i in range(n_splits)]

        skf=StratifiedKFold(shuffle=True,random_state=self.random_state,n_splits=n_splits)
        folds=[]
        for train_idx,test_idx in skf.split(self.X_train,self.y_train):
            this_fold={'X_train':self.X_train[train_idx],
                    'y_train':self.y_train[train_idx],
                    'X_test':self.X_train[test_idx],
                    'y_test':self.y_train[test_idx]}
            folds.append(this_fold)

        train_scores=np.zeros((max_iter,n_splits))
        test_scores=np.zeros((max_iter,n_splits))
        for n in range(max_iter):
            for i in range(n_splits):
                mlp_cv[i].fit(folds[i]['X_train'],folds[i]['y_train'])

                train_scores[n,i]=scoring_func(mlp_cv[i],folds[i]['X_train'],folds[i]['y_train'])
                test_scores[n,i]=scoring_func(mlp_cv[i],folds[i]['X_test'],folds[i]['y_test'])

        iter_range=np.arange(1,max_iter+1)
        ax = self.plot_learning(iter_range,train_scores,test_scores,) if plot else None

        ax.set_xlabel('Number of Iterations')

        return {'iter_range':np.arange(1,max_iter+1),'train_scores':train_scores,'test_scores':test_scores},ax

    def plot_learning(self,train_sizes,train_scores,test_scores,ax=None):
        """
        Plot learning curves based on pre-computed training and testing scores

        Parameters
        ----------
        train_sizes: array of training sizes used in learning_cv function
        train_scores: training scores returned from learning_cv function
        test_scores: testing scores returned from learning_cv fuction
        scoring: str, name of scoring function to show on y-axis label for clarity
        ax: Axes object, can be used in custom subplot. If no value is passed, function creates a new plot

        Returns
        -------
        ax: Axes object, can be used to continue formatting plot

        """
        if not ax:
            fig,ax=plt.subplots()
            
        means=np.mean(train_scores,axis=1)
        stds=np.std(train_scores,axis=1)

        ax.plot(train_sizes,means,color='steelblue',label='Training')
        ax.fill_between(train_sizes,means-stds,means+stds,alpha=0.2,color='steelblue')

        means=np.mean(test_scores,axis=1)
        stds=np.std(test_scores,axis=1)
        ax.plot(train_sizes,means,color='darksalmon',label='Validation')
        ax.fill_between(train_sizes,means-stds,means+stds,alpha=0.2,color='darksalmon')

        ax.set_ylabel(f'Score - {self.scoring}')
        ax.set_xlabel(f'Training size')

        ax.set_ylim([0.5,1.01])
        ax.set_yticks(np.arange(0.5,1.01,0.1))

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        return ax
    
    def tune_parameter(self,estimator,param_name,param_range,plot=True,semi_logx=False):

        print(f"Tuning {type(estimator)}")
        print('---------------------------------')
        print(f'Parameter Name: {param_name}')
        print(f'Parameter values: {param_range}')

        try:
            estimator.set_params(**{'random_state':self.random_state})
        except ValueError as e:
            print(e)
        gs=GridSearchCV(estimator=estimator,
                param_grid={param_name:param_range},
                scoring=self.scoring,
                n_jobs=-1,
                cv=StratifiedKFold(shuffle=True,random_state=self.random_state,n_splits=5),
                verbose=1,
                return_train_score=True,
                )
        gs.fit(self.X_train,self.y_train)

        ax=self.plot_validation(param_name,param_range,gs.cv_results_,semi_logx=semi_logx) if plot else None

        print(f'Best parameter: \n\t{gs.best_params_}')
        print(f'Best score: {gs.best_score_:0.3f}')

        return gs,ax

    def tune_multiple_parameters(self,estimator,param_grid):
        print(f"Grid search tuning {type(estimator)}")
        print('---------------------------------')
        for p in param_grid:
            print(f'Parameter Name: {p}')
            print(f'Parameter values: {param_grid[p]}')
        try:
            estimator.set_params(**{'random_state':self.random_state})
        except ValueError as e:
            print(e)
        gs=GridSearchCV(estimator=estimator,
                param_grid=param_grid,
                scoring=self.scoring,
                n_jobs=-1,
                cv=StratifiedKFold(shuffle=True,random_state=self.random_state,n_splits=5),
                verbose=1,
                return_train_score=True,
                )
        gs.fit(self.X_train,self.y_train)

        print(f'Best parameters: {gs.best_params_}')
        print(f'Best score: {gs.best_score_:0.3f}')

        return gs

    def plot_validation(self,param_name,param_range,results_dict,ax=None,semi_logx=False):
        
        """
        Plots validation curve for hyperparameter tuning.

        Parameters
        -----------
        param_name: Name of parameter being tuned
        param_range: values of parameter for x-axis
        results_dict: A dictionary with at least mean_train_score and mean_test_score. std_train_score and std_test_score optional.
        ax: Axes object
        semi_logx (default=False): set x scale

        """
        if not ax:    
            fig,ax=plt.subplots()

        #Plot Data
        best_index=np.argmax(results_dict['mean_test_score'])
        if semi_logx:  
            ax.semilogx(param_range,results_dict['mean_train_score'],color='steelblue',label='Training')  
            ax.semilogx(param_range,results_dict['mean_test_score'],color='darksalmon',label='Validation')
            
            ax.semilogx(param_range[best_index],results_dict['mean_test_score'][best_index],color='darksalmon',marker='o')
        else:
            ax.plot(param_range,results_dict['mean_train_score'],color='steelblue',label='Training')
            ax.plot(param_range,results_dict['mean_test_score'],color='darksalmon',label='Validation')
            ax.plot(param_range[best_index],results_dict['mean_test_score'][best_index],color='darksalmon',marker='o')
        
        if 'std_train_score' in results_dict.keys():    
            ax.fill_between(param_range,results_dict['mean_train_score']-results_dict['std_train_score'],
                                results_dict['mean_train_score']+results_dict['std_train_score'],
                                alpha=0.2,color='steelblue')

        
        if 'std_test_score' in results_dict.keys():
            ax.fill_between(param_range,results_dict['mean_test_score']-results_dict['std_test_score'],
                                results_dict['mean_test_score']+results_dict['std_test_score'],
                                alpha=0.2,color='darksalmon')

        # Make sure y-axis limits do not go out of range
        ax.get_ylim()
        y_lim=[0,1]
        y_lim[0]=max(ax.get_ylim()[0],0)
        y_lim[1]=min(ax.get_ylim()[1],1)
        ax.set_ylim(y_lim)
        ax.set_xlim((min(param_range),max(param_range)))

        #Show legend outside of plot
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        #Label axes
        ax.set_ylabel(f'Score - {self.scoring}')
        ax.set_xlabel(param_name)

        ax.set_ylim([0.5,1.01])
        ax.set_yticks(np.arange(0.5,1.01,0.1))

        return ax

    def final_test(self,alg_name,estimator):

        try:
            estimator.set_params(**{'random_state':self.random_state})
        except ValueError as e:
            print(e)
        start=time.time()
        estimator.fit(self.X_train,self.y_train)
        fit_time=time.time()-start
        score_func=get_scorer(self.scoring)

        y_pred=estimator.predict(self.X_test)
        score=score_func(estimator,self.X_test,self.y_test)
        cm=confusion_matrix(self.y_test,y_pred)
        cm_display = ConfusionMatrixDisplay(cm).plot()
        print(classification_report(self.y_test,y_pred,digits=3))
        cr=classification_report(self.y_test,y_pred,digits=3,output_dict=True)

        self.final_result[alg_name]={'score': score,
            'fit_time':fit_time,
            'params':estimator.get_params(),
            'conf_matr':cm,
            'class_rep':cr,
            'estimator':estimator}
        
        return self.final_result[alg_name]

    def perm_ft_importance(self,estimator,X=None,y=None):

        if not X:
            X=self.X_train
            y=self.y_train

        perm_importance = np.abs(permutation_importance(estimator, X, y,scoring=self.scoring,n_jobs=-1,random_state=self.random_state).importances_mean)

        sort_idxs = np.argsort(perm_importance)
        sort_idxs = sort_idxs[perm_importance[sort_idxs]>0]
        
        fig,ax=plt.subplots()
        ax.bar(self.feature_names[sort_idxs],perm_importance[sort_idxs])

        #shorten label names if needed
        ax.set_xticks(ticks=self.feature_names[sort_idxs],labels=[name[:20] for name in self.feature_names[sort_idxs]],rotation=90)

        #Make room for xlabels
        box = ax.get_position()
        ax.set_position([box.x0, box.y0+box.height*0.2, box.width*1.5, box.height*0.8])

        #Label axes
        ax.set_ylabel("Importance")
        ax.set_xlabel("Feature")

        print(self.feature_names.difference(self.feature_names[sort_idxs]))

        return perm_importance, ax

