import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Activation, Conv2D, AveragePooling2D, Input, Softmax
from tensorflow.keras.models import load_model, Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
import pickle
import pandas as pd
import numpy as np
from numpy.random import seed; seed(111)
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.random import set_seed; set_seed(111)
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import normalize
from six import StringIO  
from sklearn.tree import export_graphviz, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind
from IPython.display import Image  
import pydotplus
import time
from pathlib import Path
import sys
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Larger matplotlib plots
plt.rcParams['figure.figsize'] = [12,8]

# Use only needed GPU mem
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# Load training data
aging_dir = Path('/home/ubuntu/a/aging')

with open(aging_dir/'aging-cytof-data.obj', 'rb') as f:
    allData = pickle.load(f)
samples = allData["samples"]
cyto_data = allData['expr_list']
cyto_data = cyto_data[ :, :int(10e3)]
markers = allData['marker_names']

# Check for selected phenotyp
phenotype = sys.argv[1]
if phenotype not in samples.columns:
    print('phenotype not found')
    exit(1)
print(samples[phenotype].describe())

# Create output dirs
result_dir = Path('result')
if not result_dir.exists():
    result_dir.mkdir()
out_dir = result_dir/phenotype
if not out_dir.exists():
    out_dir.mkdir()

# Create train and validation data sets
x = []
y = []
for i, row in samples.iterrows():
    if math.isnan(row[phenotype]):
        continue
    phenval = row[phenotype]
    x.append(cyto_data[i])
    y.append(phenval)
x = np.asarray(x)
y_raw = np.asarray(y)

x_train, x_valid, y_train, y_valid = train_test_split(x, y_raw, train_size=0.9)

y_train = y_train.reshape(-1,1)
y_valid = y_valid.reshape(-1,1)
scaler = MinMaxScaler()
scaler.fit(y_train)
y_train = scaler.transform(y_train).reshape(1,-1)[0]
y_valid = scaler.transform(y_valid).reshape(1,-1)[0]
print(x_train.shape, x_valid.shape)

print(f"""
Data dimensionality
train = {x_train.shape}
valid = {x_valid.shape}
""".strip())

# Create model
model = Sequential([
    Input(shape=x[0].shape),
    Conv2D(3, kernel_size = (1, x.shape[2]), activation=None),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(6, kernel_size = (1,1), activation=None),
    BatchNormalization(),
    Activation('relu'),
    AveragePooling2D(pool_size = (x.shape[1], 1)),
    Flatten(),
    Dense(3, activation=None),
    BatchNormalization(),
    Activation('relu'),
    Dense(1, activation=None),
])
model.compile(
    loss='mean_absolute_error',
    optimizer='adam',
    metrics=['mean_absolute_error']
)
print(model.summary())

# Train model
model_store = out_dir/'best_model.hdf5'
checkpointer = ModelCheckpoint(
    filepath=model_store, 
    monitor='val_loss', verbose=0, 
    save_best_only=True
)
# Early stopping is assuming data normalized to [0-1] range
early_stopper = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=50,
    min_delta = 0.01,
)
st = time.time()
model.fit([x_train], y_train,
    batch_size=60,
    epochs=500, 
    verbose=1,
    callbacks=[checkpointer, early_stopper],
    validation_data=([x_valid], y_valid))
rt = time.time()-st
print(f'Runtime: {rt}')


# Plot model training history
history = model.history.history
plt.plot(pd.Series(history['mean_absolute_error']))
plt.plot(pd.Series(history['val_mean_absolute_error']))
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig(out_dir/'train-history.png')

# Get final validation predictions
y_scores = model.predict([x_valid])
y_scores = y_scores.reshape(y_scores.shape[0])
vals_true = pd.Series(scaler.inverse_transform(y_valid.reshape(-1,1)).reshape(1,-1)[0])
vals_pred = pd.Series(scaler.inverse_transform(y_scores.reshape(-1,1)).reshape(1,-1)[0])
errors = vals_pred - vals_true
errors.describe()

# Plot distribution of errors
fig, ax = plt.subplots()
ax.hist(errors, bins=np.linspace(min(errors),max(errors),num=20))
ax.set_title('Error Distribution')
ax.set_xlabel('Predicted value - True value')
plt.savefig(out_dir/'errors-dist.png')

# Plot True vs Predicted values
plt.plot(vals_true, vals_pred,'b.')
plt.axis('square')
limits = (min(min(vals_true), min(vals_pred)), max(max(vals_true), max(vals_pred)))
limits = (limits[0] - 0.1*(limits[1] - limits[0]), limits[1] + 0.1*(limits[1] - limits[0]))
plt.xlim(limits)
plt.ylim(limits)
plt.xlabel('True value')
plt.ylabel('Predicted value')
plt.title('Prediction vs True')
plt.plot(limits,limits,'--k')
coef = np.polyfit(vals_true, vals_pred, 1)
poly1d_fn = np.poly1d(coef)
plt.plot(limits, poly1d_fn(limits), '--b')
plt.savefig(out_dir/'pred-vs-true.png')

# Export
out = {
    'history': history,
    'y_valid': y_valid,
    'y_scores': y_scores,
    'vals_pred': vals_pred,
    'vals_true': vals_true,
    'scaler': scaler,
}
with open(out_dir/'results.pkl','wb') as wf:
    pickle.dump(out, wf)