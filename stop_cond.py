import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from experiment import Experiment

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
import mlrose_hiive as mlrose
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlrose_hiive as mlrose
import time

from sklearn.model_selection import train_test_split,learning_curve,cross_val_predict, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, get_scorer, confusion_matrix, ConfusionMatrixDisplay

from sklearn.inspection import permutation_importance

from sklearn.neural_network import MLPClassifier

from experiment import Experiment

exp=Experiment("WineQuality-RedWine.csv",'quality')
exp.preprocess(Pipeline(steps=[('ss',StandardScaler())]))

#Encode to make a binary problem
exp.y_train=np.apply_along_axis((lambda y: y>5),axis=0,arr=exp.y_train)
exp.y_test=np.apply_along_axis((lambda y: y>5),axis=0,arr=exp.y_test)


n_splits=5
nn_params={
    'algorithm':'random_hill_climb',
    'restarts':0,
    'max_iters':np.inf
}
scoring_func=get_scorer(exp.scoring)
mlp_cv=[mlrose.NeuralNetwork(hidden_nodes=[4],
                 activation='relu',
                 bias=True,
                 is_classifier=True,
                 learning_rate=0.1,
                 early_stopping=False,
                 clip_max=1e+10,
                 random_state=None,
                 curve=True,**nn_params)]*n_splits

skf=StratifiedKFold(shuffle=True,random_state=exp.random_state,n_splits=n_splits)
folds=[]
for train_idx,test_idx in skf.split(exp.X_train,exp.y_train):
    this_fold={'X_train':exp.X_train[train_idx],
            'y_train':exp.y_train[train_idx],
            'X_test':exp.X_train[test_idx],
            'y_test':exp.y_train[test_idx]}
    folds.append(this_fold)

train_scores=np.zeros((n_splits))
test_scores=np.zeros((n_splits))
fitness_curves=None

for i in range(n_splits):
    mlp_cv[i].fit(folds[i]['X_train'],folds[i]['y_train'])

    train_scores[i]=scoring_func(mlp_cv[i],folds[i]['X_train'],folds[i]['y_train'])
    test_scores[i]=scoring_func(mlp_cv[i],folds[i]['X_test'],folds[i]['y_test'])

    plt.plot(mlp_cv[i].fitness_curve[:,0])
    
    

print(np.mean(train_scores))
print(np.mean(test_scores))

np.mean([nn.fitness_curve[:,1][-1] for nn in mlp_cv])

