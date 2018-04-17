from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#list of classifier
# key : name
# value : parameters and svr

classifier_list = [

    {   "name":"svm_linear",
        "parameters" : {'C': [0.1, 0.5, 1, 5, 10, 50, 100]},
        "svr" : SVC(kernel='linear')
    },

    {   "name":"svm_polynomial",
        "parameters" : {'C': [0.1, 1, 3], 'degree': [4, 5, 6], 'gamma': [0.1, 1]},
        "svr" : SVC(kernel='poly')
    },

    {   "name":"svm_rbf",
        "parameters" : {'C': [0.1, 0.5, 1, 5, 10, 50, 100], 'gamma': [0.1, 0.5, 1, 3, 6, 10]},
        "svr" : SVC(kernel='rbf')
    },

    {   "name":"logistic",
        "parameters" : {'C': [0.1, 0.5, 1, 5, 10, 50, 100]},
        "svr" : LinearSVC()
    },

    {   "name":"knn",
        "parameters" : {'n_neighbors': range(1, 50), 'leaf_size': range(5, 60, 5)},
        "svr" : KNeighborsClassifier()
    },

    {   "name":"decision_tree",
        "parameters" : {'max_depth': range(1, 50), 'min_samples_split': range(2, 10, 1)},
        "svr" : DecisionTreeClassifier()

    },

    {  "name":"random_forest",
        "parameters" : {'max_depth': range(1, 50), 'min_samples_split': range(2, 10, 1)},
        "svr" : RandomForestClassifier()
    }

]

