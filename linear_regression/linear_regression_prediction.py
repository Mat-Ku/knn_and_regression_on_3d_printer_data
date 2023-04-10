###################
# 0. LOAD LIBRARIES
###################

import numpy as np
import pandas as pd
from linear_regression_algorithm import mse, LinReg


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


#########################
# 3. PREDICTING ROUGHNESS
#########################
# Roughness is given in µm

# 3.1 TRAIN-TEST-SPLIT
# input variables = layer_height, wall_thickness, infill_density, infill_pattern,
#                   nozzle_temperature, bed_temperature, print_speed, fan_speed
# target variable = roughness
# omitted variables = tension_strength, elongation, material

# Create different dataframes for materials
# serves for assuring to have an equal number of instances of each material in training and test data
df_norm_abs = df_norm[df_norm.material == 0]
df_norm_pla = df_norm[df_norm.material == 1]

# Take all but the last 5 instances of both material dataframes as training data
X_train =  pd.concat([df_norm_abs[:-5], df_norm_pla[:-5]])
y_train = np.asarray(X_train.roughness.values)
X_train = X_train.drop(['material', 'roughness', 'tension_strength', 'elongation'], axis=1)
X_train = np.asarray(X_train)

# Take the last 5 instances of both material dataframes as test data
X_test = pd.concat([df_norm_abs[-5:], df_norm_pla[-5:]])
y_test = np.asarray(X_test.roughness.values)
X_test = X_test.drop(['material', 'roughness', 'tension_strength', 'elongation'], axis=1)
X_test = np.asarray(X_test)


# 3.2 PREDICTION

model = LinReg(n = 1000, alpha = 0.001)  # 1000 iterations and a learning rate of 0.001
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
error = mse(y_test, y_pred)
print(f'The fitted linear regression model yielded an MSE of {np.round(error, 2)} µm in terms of roughness on the test data.')


##########################
# 4. PREDICTING ELONGATION
##########################
# Elongation is given in mm with respect to original size

# 4.1 TRAIN-TEST-SPLIT
# input variables = layer_height, wall_thickness, infill_density, infill_pattern,
#                   nozzle_temperature, bed_temperature, print_speed, fan_speed
# target variable = elongation
# omitted variables = tension_strength, roughness, material

# Take all but the last 5 instances of both material dataframes as training data
X_train =  pd.concat([df_norm_abs[:-5], df_norm_pla[:-5]])
y_train = np.asarray(X_train.elongation.values)
X_train = X_train.drop(['material', 'roughness', 'tension_strength', 'elongation'], axis=1)
X_train = np.asarray(X_train)

# Take the last 5 instances of both material dataframes as test data
X_test = pd.concat([df_norm_abs[-5:], df_norm_pla[-5:]])
y_test = np.asarray(X_test.elongation.values)
X_test = X_test.drop(['material', 'roughness', 'tension_strength', 'elongation'], axis=1)
X_test = np.asarray(X_test)

# 4.2 PREDICTION

model = LinReg(n = 1000, alpha = 0.001)  # 1000 iterations and a learning rate of 0.001
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
error = mse(y_test, y_pred)
print(f'The fitted linear regression model yielded an MSE of {np.round(error, 2)} mm in terms of elongation on the test data.')


################################
# 5. PREDICTING TENSION STRENGTH
################################
# Tension strength is given in MPa (Megapascals)

# 5.1 TRAIN-TEST-SPLIT
# input variables = layer_height, wall_thickness, infill_density, infill_pattern,
#                   nozzle_temperature, bed_temperature, print_speed, fan_speed
# target variable = tension_strength
# omitted variables = roughness, elongation, material

# Take all but the last 5 instances of both material dataframes as training data
X_train =  pd.concat([df_norm_abs[:-5], df_norm_pla[:-5]])
y_train = np.asarray(X_train.tension_strength.values)
X_train = X_train.drop(['material', 'roughness', 'tension_strength', 'elongation'], axis=1)
X_train = np.asarray(X_train)

# Take the last 5 instances of both material dataframes as test data
X_test = pd.concat([df_norm_abs[-5:], df_norm_pla[-5:]])
y_test = np.asarray(X_test.tension_strength.values)
X_test = X_test.drop(['material', 'roughness', 'tension_strength', 'elongation'], axis=1)
X_test = np.asarray(X_test)

# 5.2 PREDICTION

model = LinReg(n = 1000, alpha = 0.001)  # 1000 iterations and a learning rate of 0.001
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
error = mse(y_test, y_pred)
print(f'The fitted linear regression model yielded an MSE of {np.round(error, 2)} MPa in terms of tension strength on the test data.')