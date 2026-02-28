from sklearn.svm import LinearSVC
from sklearn.metrics import hinge_loss
import time
import pandas as pd
import matplotlib.pyplot as plt
from LinearlySeperableDataGeneration import make_classification

#We are going to investigate the performance of solving primal and dual problems for linear classification. 
def Primal_Dual_Comparison(X_train, y_train):
    results = {}

    Primal_model = LinearSVC(loss="squared_hinge", dual=False, max_iter=10000)
    start = time.time()
    Primal_model.fit(X_train, y_train)
    Primal_time = time.time() - start
    Primal_score = Primal_model.decision_function(X_train)
    Primal_loss = hinge_loss(y_train, Primal_score)
    
    Dual_model = LinearSVC(loss="squared_hinge", dual=True, max_iter=10000)
    start = time.time()
    Dual_model.fit(X_train, y_train)
    Dual_time = time.time() - start
    Dual_score = Dual_model.decision_function(X_train)
    Dual_loss = hinge_loss(y_train, Dual_score)
    
    return results
    
dimensions = [100, 500, 800]
size = [10000, 30000, 50000]

result_list = []
for i, d in enumerate(dimensions):
    for j, n in enumerate(size):
        X_train, X_test, y_train, y_test, a = make_classification(d, n, 100, 7)
        result = Primal_Dual_Comparison(X_train, y_train)
        result_list.append(
        {
            "n":n,
            "d":d,
            "Primal_Time":result["Primal"]["time"],
            "Primal_Loss":result["Primal"]["loss"],
            "Primal_Iter":result["Primal"]["iter"],
            "Dual_Time":result["Dual"]["time"],
            "Dual_Loss":result["Dual"]["loss"],
            "Dual_Iter":result["Dual"]["iter"]
        }
    )

result_dataframe = pd.DataFrame(result_list, columns=["n","d","Primal_Time","Primal_Loss", "Primal_Iter","Dual_Time","Dual_Loss", "Dual_Iter"])
print(result_dataframe)

plt.figure()
for d in dimensions:
    subset = result_dataframe[result_dataframe["d"] == d]
    plt.plot(subset["n"], subset["Primal_Time"], marker='o', label=f"d={d}")
plt.xlabel("Number of samples (n)")
plt.ylabel("Time (seconds)")
plt.title("Primal Time vs n for different dimensions")
plt.legend()
plt.show()

plt.figure()
for d in dimensions:
    subset = result_dataframe[result_dataframe["d"] == d]
    plt.plot(subset["n"], subset["Dual_Time"], marker='o', label=f"d={d}")
plt.xlabel("Number of samples (n)")
plt.ylabel("Time (seconds)")
plt.title("Dual Time vs n for different dimensions")
plt.legend()
plt.show()

plt.figure()
for d in dimensions:
    subset = result_dataframe[result_dataframe["d"] == d]
    plt.plot(subset["n"], subset["Primal_Loss"], marker='o', label=f"d={d}")
plt.xlabel("Number of samples (n)")
plt.ylabel("Loss")
plt.title("Primal Loss vs n for different dimensions")
plt.legend()
plt.show()

plt.figure()
for d in dimensions:
    subset = result_dataframe[result_dataframe["d"] == d]
    plt.plot(subset["n"], subset["Dual_Loss"], marker='o', label=f"d={d}")
plt.xlabel("Number of samples (n)")
plt.ylabel("Loss")
plt.title("Dual Loss vs n for different dimensions")
plt.legend()
plt.show()

plt.figure()
for n in size:
    subset = result_dataframe[result_dataframe["n"] == n]
    plt.plot(subset["d"], subset["Primal_Time"], marker='o', label=f"n={n}")
plt.xlabel("Dimension (d)")
plt.ylabel("Time (seconds)")
plt.title("Primal Time vs d for different sample sizes")
plt.legend()
plt.show()

plt.figure()
for n in size:
    subset = result_dataframe[result_dataframe["n"] == n]
    plt.plot(subset["d"], subset["Dual_Time"], marker='o', label=f"n={n}")
plt.xlabel("Dimension (d)")
plt.ylabel("Time (seconds)")
plt.title("Dual Time vs d for different sample sizes")
plt.legend()
plt.show()

plt.figure()
for n in size:
    subset = result_dataframe[result_dataframe["n"] == n]
    plt.plot(subset["d"], subset["Primal_Loss"], marker='o', label=f"n={n}")
plt.xlabel("Dimension (d)")
plt.ylabel("Loss")
plt.title("Primal Loss vs d for different sample sizes")
plt.legend()
plt.show()

plt.figure()
for n in size:
    subset = result_dataframe[result_dataframe["n"] == n]
    plt.plot(subset["d"], subset["Dual_Loss"], marker='o', label=f"n={n}")
plt.xlabel("Dimension (d)")
plt.ylabel("Loss")
plt.title("Dual Loss vs d for different sample sizes")
plt.legend()
plt.show()