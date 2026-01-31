"""Central configuration for the botnet detection pipeline."""

import os

# Paths
DATA_FOLDER = "data"
LOG_FOLDER = "logs"
PLOTS_FOLDER = "plots"

# Train/test
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Plot colors
COLOR_BACKGROUND = "#3498db"
COLOR_BOTNET = "#e74c3c"
COLOR_NORMAL = "#2ecc71"

# Feature lists
MAIN_FEATURES = [
    'Dur', 'Proto', 'Dir', 'State',
    'SrcAddr', 'DstAddr',
    'SportRange', 'DportRange',
    'TotPkts', 'TotBytes', 'SrcBytes',
    'BytesPerSecond', 'PktsPerSecond',
    'BytePktRatio',
    'SrcAddrEntropy', 'DstAddrEntropy',
    'DurCategory',
    'Botnet'
]

NUMERIC_FEATURES = [
    'Dur', 'TotPkts', 'TotBytes', 'SrcBytes',
    'BytesPerSecond', 'PktsPerSecond', 'BytePktRatio'
]

CATEGORICAL_COLS = ['Proto', 'Dir', 'State', 'SportRange', 'DportRange', 'DurCategory']

IP_COLS = ['SrcAddr', 'DstAddr']

COLS_TO_DROP = ['sTos', 'dTos', 'StartTime']

# Duration binning
DUR_BINS = [0, 1, 10, 60, float('inf')]
DUR_LABELS = ['very_short', 'short', 'medium', 'long']

# CTU-13 scenario files
SCENARIO_FILES = [
    ('1-Neris', '1-Neris-20110810.binetflow.csv'),
    ('2-Neris', '2-Neris-20110811.binetflow.csv'),
    ('3-Rbot', '3-Rbot-20110812.binetflow.csv'),
    ('4-Rbot', '4-Rbot-20110815.binetflow.csv'),
    ('5-Virut', '5-Virut-20110815-2.binetflow.csv'),
    ('6-Menti', '6-Menti-20110816.binetflow.csv'),
    ('7-Sogou', '7-Sogou-20110816-2.binetflow.csv'),
    ('8-Murlo', '8-Murlo-20110816-3.binetflow.csv'),
    ('9-Neris', '9-Neris-20110817.binetflow.csv'),
    ('10-Rbot', '10-Rbot-20110818.binetflow.csv'),
    ('11-Rbot', '11-Rbot-20110818-2.binetflow.csv'),
    ('12-NsisAy', '12-NsisAy-20110819.binetflow.csv'),
    ('13-Virut', '13-Virut-20110815-3.binetflow.csv'),
]

# Model hyperparameter grids
def get_model_grids():
    """Returns model hyperparameter grids for GridSearchCV."""
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    return {
        "RandomForest": {
            "model": RandomForestClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                'clf__n_estimators': [50, 100],
                'clf__max_depth': [5, 10],
                'clf__min_samples_split': [2, 5],
                'clf__min_samples_leaf': [1, 2]
            }
        },
        "DecisionTree": {
            "model": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                'clf__max_depth': [5, 10, None],
                'clf__min_samples_split': [2, 5],
                'clf__min_samples_leaf': [1, 2]
            }
        },
        "NaiveBayes": {
            "model": GaussianNB(),
            "param_grid": {
                'clf__var_smoothing': [1e-9, 1e-8, 1e-7]
            }
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "param_grid": {
                'clf__n_neighbors': [3, 5, 7],
                'clf__weights': ['uniform', 'distance']
            }
        },
        "SVM": {
            "model": SVC(random_state=RANDOM_STATE, probability=True),
            "param_grid": {
                'clf__C': [0.1, 1.0, 10.0],
                'clf__kernel': ['linear', 'rbf']
            }
        },
        "LogisticRegression": {
            "model": LogisticRegression(solver='liblinear', random_state=RANDOM_STATE),
            "param_grid": {
                'clf__C': [0.01, 0.1, 1, 10],
                'clf__penalty': ['l1', 'l2']
            }
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                'clf__n_estimators': [50, 100],
                'clf__learning_rate': [0.01, 0.1],
                'clf__max_depth': [3, 5]
            }
        }
    }
