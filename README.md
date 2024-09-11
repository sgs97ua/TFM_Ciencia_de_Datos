Trabajo final de master de ciencia de datos que combina dos modelos con el objetivo de generar un grafo de conocimiento.

Los modelos que han sido elegidos para llevar acabo esta tarea son:

* [ReFinED](https://github.com/amazon-science/ReFinED): Modelo de enlazado de entidades (entity linking(EL)) que enlaza las menciones de entidades en los documentos con su correspondiente entidad en Wikipedia o Wikidata.
* [RebEL](https://github.com/Babelscape/rebel): Modelo secuencia-secuencia basada en BART que realiza una extracción de relaciones para más de 200 tipos diferentes de relaciones.


En este repositorio se encuentran *Jupyter notebooks* que se han utilizado para realización de diferentes pruebas con diferentes modelos y scripts de *Python* en los que se encuentran implementados el código que constituye nuestra herramienta.

A continuación, enumeramos los ficheros que constituyen este fichero y el propósito de cada uno de estos.

- [combination.ipynb](./combination.ipynb): Notebook utilizado como banco de pruebas para elaborar la estrategia 3.
- [docker-compose.yml](./docker-compose.yml): Archivo de configuración utilizado por Docker Compose para definir y administrar la base de datos Neo4j.
- [evaluation.ipynb](./evaluation.ipynb): Notebook utilizado para evaluar las estrategias planteadas para combinar los modelos.
- [kgbuilder.py](./kgbuilder.py): Módulo de python que contiene la implementación de la estrategia seleccionada.
- [llm.ipynb](./llm.ipynb): Notebook utilizado para probar el uso de llm en la tarea de generación de grafos de conocimiento apoyándose en ontologías.
- [main.py](./main.py): Script de python que implementa la herramienta comentada en la memoria.
- [properties.json](./properties.json): Diccionario de propiedades de Wikidata utilizado para mejorar el proceso de extracción de información de la base de conocimiento.
- [pruebas.ipynb](./pruebas.ipynb): Notebook utilizado para replicar los resultados de evaluación de los papers de ambos modelos.
- [rebel_structure.txt](./rebel_structure.txt): Fichero de texto que detalla la estructura del modelo RebEL.
- [refined_model_structure.txt](./refined_model_structure.txt) Fichero texto que detalla la estructura de los componentes del modelo ReFinED.
- [requirements.txt](./requirements.txt): Listado de dependencias de python.
- [test_sample.json](./test_sample.json): Ejemplo de estructura de las muestras utilizadas para la evaluación.
- [texto_test_tfm](./texto_test_tfm): Textos de ejemplos utilizados para mostrar el funcionamiento de la herramienta.



Vídeo demostración de la herramienta [URL](https://unialicante-my.sharepoint.com/:v:/g/personal/sgs97_mscloud_ua_es/ERc1LeAUw1dLucjaGNmra6IBWK_K77Ab5_CqedHWpo5XPw?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=Pv0EU9)

