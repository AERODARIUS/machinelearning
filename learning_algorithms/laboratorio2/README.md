# Cómo ejecutar el laboratorio

## Instalar dependencias

- `pip install pandas`
- `pip install sklearn`
- `pip install matplotlib`

## Ejecutar los siguentes archivos

- `python parte1.py` Entrena el algoritmo ID3 implementado por nosotros, y luego lo pone a prueba
- `python parte2.py` Ejecuta el algoritmo de sklearn, lo pone a prueba y luego compara con nuestro algoritmo

Ambos archivos son independientes uno del otro, esto quiere decir que se pueden ejecutar en cualquier orden.

### Uso avanzado
Ambos programas pueden recibir parámetros, para ver más información sobre cada ejercicio. A continuación se explica en más detalle.


#### Estructura general del comando
`[parte1.py | parte2.py] [-h | --help] [-v | --verbose] [-m | --multi] [-t | --tree]`

#### Argumentos opcionales
`-h, --help` *Muestra el mensaje de ayuda*

`--verbose, -v` Imprime en la consola el avance del entrenamiento y pruebas, admás muestra los gráficos.

`--multi, -m` Por defecto se realiza un único experimento entrenando con un 80% de los datos y realizando un test con el 20% restante. Al utilizar esta opción se realizan varias preubas con varias combinaciones.

`--tree, -t` Imprime el árbol generado en la consola.


#### Ejemplos de su uso
- `python parte1.py`
- `python parte1.py -vt`
- `python parte1.py -v -t`
- `python parte2.py -t -m`
- `python parte2.py -vtm`
- `python parte1.py -tree`
- `python parte1.py -multi`
- `python parte2.py -verbose`

