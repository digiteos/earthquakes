###############################################################################################################
#################### Modelo de predicción de terremotos con Machine Learning y red neuronal ###################
###############################################################################################################

# Python 3.8.7
# Librerías requeridas: Numpy 1.19.5, Pandas 1.2.0, Sklearn 0.0, Keras 2.4.3, Tensorflow 2.4.0, Datetime 4.3

# Dashboard en Power BI con información de Terremotos
	# https://app.powerbi.com/view?r=eyJrIjoiMDQzMTI5MWItMzAyZi00MzRkLTkxMDEtYjUwMzRjZmEyODY3IiwidCI6IjNhY2M3NWRiLTNhOTQtNDFmOS04N2M3LWIwNjE3MGRlZjEwYiJ9&pageName=ReportSectione26908533edc15cb8d45

# Referencias
	# Earthquakes magnitude predication using artificial neural network in northern Red Sea area
	#	https://www.sciencedirect.com/science/article/pii/S1018364711000280
# Fuente de datos de terremotos:
	# De 1900 a 1969 | Tableau Resources - https://public.tableau.com/en-us/s/resources
	# De 1970 a 2019 | Incorporated Research Institutions for Seismology (IRIS) - https://www.iris.edu/hq/

# Última fecha de actualización: 21/01/2021
# Github: https://github.com/digiteos/earthquakes

##############################################################################################################


# 1) Importando librerías de Python (Numpy y Pandas)

import numpy as np
import pandas as pd

# 2) Cargando y leyendo el set de datos históricos (el archivo csv tiene ; como separador)

data = pd.read_csv("EarthQuakes-Data-1900-2019.csv", sep=';')
data.columns

# 3) Visualizando encabezado de tabla, cantidad de registros y tipos de datos
 
print(data.head())
print(data.shape)
print(data.dtypes)

# 4) Convirtiendo la fecha/hora a formato Unix (solo admite datos a partir de 1970) para que pueda ser procesada por la red neuronal y visualizando el nuevo dataset

import datetime
import time

timestamp = []
for d, t in zip(data['Date'], data['Time']):
    try:
        ts = datetime.datetime.strptime(d+' '+t, '%d/%m/%Y %H:%M:%S')
        timestamp.append(time.mktime(ts.timetuple()))
    except ValueError:
        # print('ValueError')
        timestamp.append('ValueError')
timeStamp = pd.Series(timestamp)
data['Timestamp'] = timeStamp.values
final_data = data.drop(['Date', 'Time'], axis=1)
final_data = final_data[final_data.Timestamp != 'ValueError']
final_data = final_data

# Aquí se cambia el tipo de dato de la nueva columna Timestamp a formato float para que pueda ser procesado por la red neuronal

final_data['Timestamp'] = final_data['Timestamp'].astype(float)

final_data.head()
print(data.dtypes)
print(data.head())

# 5) Para crear el modelo de predicción, se dividen los datos en X e Y respectivamente, siendo las entradas para el modelo.
		# Entradas: Tiempo (Timestamp), Latitud y Longitud
		# Salidas:  Magnitud y Profundidad
	# Se dividen los datos en 80% para entreamiento y 20% para prueba

X = final_data[['Timestamp', 'Latitude','Longitude']]
y = final_data[['Depth', 'Magnitude']]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, X_test.shape)

# 6) Creando el modelo de red neuronal
		# Capa 1: 16 nodos
		# Capa 2: 16 nodos
		# Capa 3: 2  nodos
		# Funciones de activación: Relu y Softmax

from keras.models import Sequential
from keras.layers import Dense

def create_model(neurons, activation, optimizer, loss):
    model = Sequential()
    model.add(Dense(neurons, activation=activation, input_shape=(3,)))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(2, activation='softmax'))
    
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    return model

# 7) Definiendo parámetros para encontrar el mejor ajuste
		# Activación: Sigmoid, Relu 
		# Optimización:	SGD, Aladelta
		# Pérdida: Squared Hinge

from keras.wrappers.scikit_learn import KerasClassifier

model = KerasClassifier(build_fn=create_model, verbose=0)

neurons = [16]
batch_size = [10]
epochs = [10]
activation = ['sigmoid', 'relu']
optimizer = ['SGD', 'Adadelta']
loss = ['squared_hinge']

param_grid = dict(neurons=neurons, batch_size=batch_size, epochs=epochs, activation=activation, optimizer=optimizer, loss=loss)


# 8)  Ahora necesitamos encontrar el mejor ajuste del modelo, obtendremos la puntuación media y la desviación estándar

from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# 9) Los parámetros del modelo con mejor ajuste se utilizan para calcular la puntuación con los datos de entrenamiento y los datos de prueba

model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(3,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='SGD', loss='squared_hinge', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=10, epochs=20, verbose=1, validation_data=(X_test, y_test))

[test_loss, test_acc] = model.evaluate(X_test, y_test)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

# 10) En las pruebas realizadas se obtuve una precisión de 86%.
