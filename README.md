# Trabajo de Fin de Grado

En este repositorio puede consultarse el código desarrollado para el
Trabajo de Fin de Grado, que lleva por título _Detección de arritmias
en electrocardiogramas usando aprendizaje profundo en el contexto del
Internet Of Things_. El modelo desarrollado consiste en una red neuronal
convolucional, y ha sido desarrollado utilizando el _framework_ Keras.

## Estructura de directorios

### `data_reader`

En este directorio se encuentra el código destinado a la lectura y
preprocesamiento de datos. Dentro del archivo `__init__.py` se
recogen las dos clases implementadas para esta tarea. La primera de
ellas, `DataReader`, permite la lectura de los datos de
electrocardiogramas desde archivo. La segunda, `DataGenerator`,
se utiliza para preparar los datos y servirlos como entrada a la
red neuronal.

### `cnn_model`

En esta carpeta se recoge el código implementado para definir la
arquitectura del modelo, así como el proceso de entrenamiento
y test. Este directorio contiene los siguientes archivos:

1. `models.py`: En este archivo se define la estructura de la
red neuronal. Se ha implementado utilizando la clase `Sequential`
que pertenece a Keras.
