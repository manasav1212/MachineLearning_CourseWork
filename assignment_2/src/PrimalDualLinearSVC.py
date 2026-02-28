from sklearn.svm import LinearSVC
from sklearn.metrics import hinge_loss
import time
import pandas as pd
import matplotlib.pyplot as plt
from LinearlySeperableDataGeneration import make_classification
from sklearn.preprocessing import StandardScaler

#We are going to investigate the performance of solving primal and dual problems for linear classification. 
def Primal_Dual_Comparison(X_train, y_train):
    #scaler = StandardScaler()
    #X_train = scaler.fit_transform(X_train)
    results = {}

    Primal_model = LinearSVC(loss="squared_hinge", dual=False, C=0.1, max_iter=30000, tol=1e-3)
    start = time.time()
    Primal_model.fit(X_train, y_train)
    Primal_time = time.time() - start
    Primal_score = Primal_model.decision_function(X_train)
    Primal_loss = hinge_loss(y_train, Primal_score)
    results["Primal"] = {"time":Primal_time, "loss":Primal_loss, "iter": Primal_model.n_iter_}
    
    Dual_model = LinearSVC(loss="squared_hinge", dual=True, C=0.1, max_iter=30000, tol=1e-3)
    start = time.time()
    Dual_model.fit(X_train, y_train)
    Dual_time = time.time() - start
    Dual_score = Dual_model.decision_function(X_train)
    Dual_loss = hinge_loss(y_train, Dual_score)
    results["Dual"] = {"time":Dual_time, "loss":Dual_loss, "iter": Dual_model.n_iter_}
    
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

#Time vs n for different d
d_colors = {100: "blue", 500: "green", 800: "red"}
n_colors = {10000: "blue", 30000: "green", 50000: "red"}

plt.figure()

for d in dimensions:
    subset = result_dataframe[result_dataframe["d"] == d].sort_values("n")
    
    plt.plot(subset["n"], subset["Primal_Time"], marker='o', linestyle='-', color=d_colors[d], label=f"d={d} (Primal)")
    plt.plot(subset["n"], subset["Dual_Time"], marker='o', linestyle='--', color=d_colors[d], label=f"d={d} (Dual)")

plt.xlabel("Number of samples (n)")
plt.ylabel("Time (seconds)")
plt.title("Training Time vs n (Primal vs Dual)")
plt.legend()
plt.show()


# Time vs d for different n
plt.figure()

for n in size:
    subset = result_dataframe[result_dataframe["n"] == n].sort_values("d")
    
    # Primal
    plt.plot(subset["d"], subset["Primal_Time"],marker='o', color=n_colors[n], linestyle='-',label=f"n={n} (Primal)")
    # Dual
    plt.plot(subset["d"], subset["Dual_Time"],marker='o', color=n_colors[n], linestyle='--',label=f"n={n} (Dual)")

plt.xlabel("Dimension (d)")
plt.ylabel("Time (seconds)")
plt.title("Training Time vs d (Primal vs Dual)")
plt.legend()
plt.show()

plt.figure()

for d in dimensions:
    subset = result_dataframe[result_dataframe["d"] == d].sort_values("n")
    
    plt.plot(subset["n"], subset["Primal_Loss"], marker='o', linestyle='-', color=d_colors[d], label=f"d={d} (Primal)")
    plt.plot(subset["n"], subset["Dual_Loss"], marker='o', linestyle='--', color=d_colors[d], label=f"d={d} (Dual)")

plt.xlabel("Number of samples (n)")
plt.ylabel("Time (seconds)")
plt.title("Loss vs n (Primal vs Dual)")
plt.legend()
plt.show()


# Time vs d for different n
plt.figure()

for n in size:
    subset = result_dataframe[result_dataframe["n"] == n].sort_values("d")
    
    # Primal
    plt.plot(subset["d"], subset["Primal_Loss"],marker='o', color=n_colors[n], linestyle='-',label=f"n={n} (Primal)")
    # Dual
    plt.plot(subset["d"], subset["Dual_Loss"],marker='o', color=n_colors[n], linestyle='--',label=f"n={n} (Dual)")

plt.xlabel("Dimension (d)")
plt.ylabel("Time (seconds)")
plt.title("Loss vs d (Primal vs Dual)")
plt.legend()
plt.show()