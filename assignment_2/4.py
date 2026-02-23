from sklearn.svm import LinearSVC
from sklearn.metrics import hinge_loss
import time

#We are going to investigate the performance of solving primal and dual problems for linear classification. 
def Primal_Dual_Comparison(X_train, y_train):
    results = {}

    Primal_model = LinearSVC(loss="hinge", dual=False)
    start = time.time()
    Primal_model.fit(X_train, y_train)
    Primal_time = time.time() - start
    Primal_score = Primal_model.decision_function(X_train)
    Primal_loss = hinge_loss(y_train, Primal_score)
    results["Primal"] = {"time":Primal_time, "loss":Primal_loss}
    
    Dual_model = LinearSVC(loss="hinge", dual=True)
    start = time.time()
    Dual_model.fit(X_train, y_train)
    Dual_time = time.time() - start
    Dual_score = Primal_model.decision_function(X_train)
    Dual_loss = hinge_loss(y_train, Dual_score)
    results["Dual"] = {"time":Dual_time, "loss":Dual_loss}
    
    return results
    
# You may use the 9 datasets in the previous task. The easiest way to reuse a dataset is to keep all data in a file. 
# You may use the hinge loss and the default value for the regularization parameter. 
# For each dataset, compare the time costs and loss convergences of training a linear SVC by solving the primal and dual problems.