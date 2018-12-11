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
arquitectura del modelo, así como el proceso de entrenamiento y
test. Este directorio contiene los siguientes archivos:

1. `models.py`: En este archivo se define la estructura de la red
neuronal. Se ha implementado utilizando la clase `Sequential` que
pertenece a Keras.
2. `train.py`: Script que define el método de entrenamiento de la red.
3. `test.py`: Script que permite la evaluación de un modelo ya
entrenado, cargándolo desde memoria y tomando el subconjunto de test
correspondiente.
4. Carpeta `saved_models`: En ella se encuentran los modelos que han
sido entrenados para los cinco subconjuntos de validación durante la
etapa de experimentación del desarrollo del trabajo.
5. `write_answers.py`: Este archivo se utiliza para guardar en fichero
las predicciones hechas por un modelo, para poder utilizarlas en otros
puntos en caso de que sea necesario. Actúa de la misma manera que el
archivo `test.py`, pero guarda las predicciones hechas en un archivo
externo.

### `utils`

En esta carpeta se encuentra el código que se ha utilizado para crear
la función de puntuación utilizada en la competición, para poder
ser utilizada durante el proceso de entrenamiento de la red.

### `dataset`

En esta carpeta se encuentra la información referente al conjunto de
datos con los que se ha trabajado. Se ha evitado subir el conjunto
de datos completo, dado que es muy pesado y no tiene mucho sentido
tenerlo aquí almacenado. Para poder ejecutar el modelo, previamente
hay que descargar el conjunto de datos y descomprimirlo dentro de una
carpeta llamada `data` dentro de este directorio.
