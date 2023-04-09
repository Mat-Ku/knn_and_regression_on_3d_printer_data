###################
# 0. LOAD LIBRARIES
###################

import numpy as np
import pandas as pd
from knn_algorithm import KNN


##############
# 1. LOAD DATA
##############

df = pd.read_csv('*') # insert path to dataset


####################
# 2. PREPROCESS DATA
####################

# Correct misspelled column name
df.columns = ['tension_strength' if name == 'tension_strenght' else name for name in df.columns]

# Turn 'layer_height' and 'elongation' into mm
df.layer_height = df.layer_height * 100
df.elongation = df.elongation * 100

# Convert categorical into numerical values
# infill_pattern: 'grid' = 0; 'honeycomb' = 1
df.infill_pattern = [0 if pattern == 'grid' else 1 for pattern in df.infill_pattern]
# material: 'abs' [Acrylonitrile Butadiene Styrene] = 0; 'pla' [Polylactic Acid] = 1
df.material = [0 if mat == 'abs' else 1 for mat in df.material]

# Normalize data
df_norm = (df - np.min(df)) / (np.max(df) - np.min(df))


#####################
# 3. TRAIN-TEST-SPLIT
#####################
# input variables = layer_height, wall_thickness, infill_density, infill_pattern,
#                   nozzle_temperature, bed_temperature, print_speed, fan_speed
# output variables = roughness, elongation, tension_strength
# target variable = material

# Create different dataframes for materials
# serves for assuring to have an equal number of instances of each material in training and test data
df_norm_abs = df_norm[df_norm.material == 0]
df_norm_pla = df_norm[df_norm.material == 1]

# Take all but the last 5 instances of both material dataframes as training data
X_train =  pd.concat([df_norm_abs[:-5], df_norm_pla[:-5]])
y_train = X_train.material.values
X_train = X_train.drop(['material'], axis=1)
X_train = np.asarray(X_train)

# Take the last 5 instances of both material dataframes as test data
X_test = pd.concat([df_norm_abs[-5:], df_norm_pla[-5:]])
y_test = X_test.material.values
X_test = X_test.drop(['material'], axis=1)
X_test = np.asarray(X_test)


###############
# 4. PREDICTION
###############

# Determine k according to rule of thumb as square-root of size of dataset
k = int(np.round(np.sqrt(len(df))))

# Assure that k is an odd number
if k % 2 == 0:
    k = k + 1
else:
    pass

# Fit and predict
classifier = KNN(k)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test, X_train)


############
# 5. RESULTS
############

accuracy = np.sum(np.asarray(y_pred) == np.asarray(y_test)) / len(y_test)
print(f'Accuracy: {accuracy * 100} %')