# Cómo ejecutar el laboratorio

## Instalar dependencias

- `pip install numpy`
- `pip install matplotlib`

## Abrir Jupyter en el navegador

Ejecutar jupyter notebook localmente: `jupyter notebook`, luego abrir el
archivo `Tarea1AA.ipynb` y ejecutar todos los bloques secuencialmente.


## Ver partidas

### Guía para interpretar salida del programa
Al final de cada entrenamiento se muestra la siguiente información:
  - Pesos finales obtenidos
  - Cantidad de partidas ganas, empatadas y perdidas
  - Una o dos gráficas con la evolución del porcentaje de partidas ganadas

Luego de entrenar los algoritmos se realiza un campeonato entre los tres.
Al final del campeonato entre los tres algoritmos se muestra el resultado de cada partida.

### Cómo ver historico jugadas

Para ver cada jugada que se realiza en una partida, se debe descomentar
el código dentro de la función jugar y se van a imprimir los tableros
por cada jugada.
