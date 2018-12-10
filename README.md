# Trabajo de Fin de Grado

En este repositorio puede consultarse el código desarrollado para el
Trabajo de Fin de Grado, que lleva por título _Detección de arritmias
en electrocardiogramas usando aprendizaje profundo en el contexto del
Internet Of Things_. El modelo desarrollado consiste en una red neuronal
convolucional, y ha sido desarrollado utilizando el _framework_ Keras.

## Estructura de directorios

### `data\_reader`

En este directorio se encuentra el código destinado a la lectura y
preprocesamiento de datos. Dentro del archivo \_\_init\_\_.py se
recogen las dos clases implementadas para esta tarea. La primera de
ellas, `DataReader`, permite la lectura de los datos de
electrocardiogramas desde archivo. La segunda, `DataGenerator`,
se utiliza para preparar los datos y servirlos como entrada a la
red neuronal.
