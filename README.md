# Telecom X - Challenge Parte 2: Predicción de Churn de Clientes

## Propósito del Análisis

El objetivo principal de este proyecto es **predecir la probabilidad de que un cliente de TelecomX cancele su servicio (Churn)**. A través de un análisis exhaustivo de las características de los clientes, buscamos identificar los factores más relevantes que influyen en esta decisión para, posteriormente, proponer estrategias de retención efectivas.

## Estructura del Proyecto

Este proyecto se organiza de la siguiente manera:

-   `nombre_del_cuaderno.ipynb`: El cuaderno principal de Google Colab que contiene todo el código para la carga de datos, preprocesamiento, análisis, modelado y evaluación.
-   `datos_tratados.csv`: El conjunto de datos utilizado para el análisis, previamente tratado y cargado desde el repositorio de GitHub.
-   Cualquier carpeta con visualizaciones generadas durante el EDA.

## Proceso de Preparación de Datos

La preparación de los datos fue una etapa crucial para asegurar que el conjunto de datos fuera adecuado para el modelado predictivo:

1.  **Carga de Datos:** El conjunto de datos `datos_tratados.csv` fue cargado directamente desde un repositorio público de GitHub.
2.  **Limpieza Inicial:** Se eliminó la columna `customerID` por ser un identificador único que no aporta valor predictivo.
3.  **Clasificación de Variables:** Se identificaron variables numéricas (como `tenure`, `Charges_Monthly`, `Charges_Total`, `Cuentas_Diarias`) y variables categóricas (columnas de tipo `object` como `MultipleLines`, `InternetService`, `Contract`, `PaymentMethod`, etc.).
4.  **Codificación de Variables Categóricas:** Las variables categóricas se transformaron a formato numérico utilizando **One-Hot Encoding**. Se aplicó `drop_first=True` para evitar la multicolinealidad.
5.  **Verificación y Manejo del Desbalance de Clases:** Se analizó la distribución de la variable objetivo (`Churn`), detectando un desbalance significativo. Para abordarlo, se aplicó la técnica de sobremuestreo **SMOTE (Synthetic Minority Over-sampling Technique)** a los datos para igualar el número de instancias en las clases minoritaria y mayoritaria.
6.  **Estandarización de Variables Numéricas:** Las características numéricas se **estandarizaron** utilizando `StandardScaler`. Este paso fue necesario para los modelos sensibles a la escala (como Regresión Logística) para asegurar que ninguna característica dominara el cálculo de distancias o coeficientes debido a su magnitud. Las variables booleanas generadas por One-Hot Encoding no fueron escaladas.
7.  **División en Conjuntos de Entrenamiento y Prueba:** El conjunto de datos procesado y balanceado fue dividido en conjuntos de entrenamiento (70%) y prueba (30%) utilizando `train_test_split` con `stratify` para mantener la proporción de clases en ambos conjuntos.

## Modelado Predictivo

Se entrenaron dos modelos de clasificación para predecir el Churn:

1.  **Regresión Logística:** Un modelo lineal que es sensible a la escala de los datos, por lo que se benefició de la estandarización previa.
2.  **Random Forest:** Un modelo basado en árboles que no es sensible a la escala de los datos.

Ambos modelos fueron entrenados con los conjuntos de datos de entrenamiento balanceados y escalados.

## Evaluación del Rendimiento del Modelo

El rendimiento de los modelos se evaluó en el conjunto de prueba utilizando métricas clave para problemas de clasificación, incluyendo Exactitud (Accuracy), Precisión (Precision), Recall (Sensibilidad), F1-score y la Matriz de Confusión.

-   Ambos modelos mostraron un buen desempeño, con métricas superiores al 80% para ambas clases después del balanceo con SMOTE.
-   El modelo de **Random Forest** demostró ser ligeramente superior en las métricas de Precision, Recall y F1-score para la clase minoritaria (Churn), indicando una mejor capacidad para identificar correctamente a los clientes que cancelarán con menos falsos positivos y negativos.

A continuación, se presentan las métricas clave obtenidas en el conjunto de prueba:

| Métrica   | Regresión Logística | Random Forest |
| :-------- | :------------------ | :------------ |
| Exactitud | 0.8092              | 0.8410        |
| Precisión | 0.8116              | 0.8385        |
| Recall    | 0.8051              | 0.8447        |
| F1-score  | 0.8083              | 0.8416        |

## Análisis Exploratorio de Datos (EDA) e Insights

Durante el EDA, se visualizaron relaciones clave entre algunas variables y Churn:

-   **Boxplot de Tenure vs Churn:** Mostró que los clientes que cancelan tienden a tener un tiempo de contrato significativamente menor.
-   **Boxplot de Charges_Total vs Churn:** Indicó que los clientes que cancelan generalmente tienen gastos totales más bajos.
-   **Gráfico de Barras de Proporción de Churn por Tipo de Contrato:** Evidenció claramente que los contratos "Month-to-month" tienen una tasa de cancelación mucho más alta que los contratos a más largo plazo.

## Importancia de las Variables

El análisis de la importancia de las variables (coeficientes para Regresión Logística e importancia de características para Random Forest) destacó los factores más influyentes:

-   **Variables clave identificadas por ambos modelos:** Tenure, Variables de Gasto (Charges_Monthly, Charges_Total, Cuentas_Diarias), Tipo de Servicio de Internet (Fiber optic), Método de Pago (Electronic check) y Tipo de Contrato.
-   Estas variables son cruciales para entender por qué los clientes cancelan.

### Coeficientes del Modelo de Regresión Logística (Ordenados por Valor Absoluto)

| Feature                               |   Coefficient |
| :------------------------------------ | -------------:|
| InternetService_Fiber optic           |       4.68527 |
| Charges_Monthly                       |      -2.18569 |
| Cuentas_Diarias                       |      -2.18569 |
| tenure                                |      -1.8269  |
| StreamingTV_Yes                       |       1.81888 |
| StreamingMovies_Yes                   |       1.66342 |
| PaymentMethod_Electronic check        |       1.46352 |
| MultipleLines_Yes                     |       1.38006 |
| PaymentMethod_Mailed check            |       0.97906 |
| Charges_Total                         |       0.972913 |
| PaymentMethod_Credit card (automatic) |       0.848051 |
| DeviceProtection_Yes                  |       0.783015 |
| OnlineBackup_Yes                      |       0.779651 |
| MultipleLines_NoService               |      -0.702543 |
| Contract_Two year                     |      -0.598454 |
| OnlineBackup_NoService                |      -0.552593 |
| DeviceProtection_NoService            |      -0.552593 |
| TechSupport_NoService                 |      -0.552593 |
| StreamingTV_NoService                 |      -0.552593 |
| StreamingMovies_NoService             |      -0.552593 |
| InternetService_No                    |      -0.552593 |
| OnlineSecurity_NoService              |      -0.552593 |
| OnlineSecurity_Yes                    |       0.489556 |
| TechSupport_Yes                       |       0.407757 |
| PhoneService                          |       0.295452 |
| Contract_One year                     |      -0.263138 |
| Dependents                            |      -0.253502 |
| gender                                |      -0.200856 |
| Partner                               |      -0.145092 |
| SeniorCitizen                         |      -0.056225 |
| PaperlessBilling                      |       0.008082 |

### Importancia de Características del Modelo Random Forest (Ordenada)

| Feature                               |   Importance |
| :------------------------------------ | -------------:|
| tenure                                |      0.144563 |
| Charges_Total                         |      0.136324 |
| Cuentas_Diarias                       |      0.106967 |
| Charges_Monthly                       |      0.106479 |
| PaymentMethod_Electronic check        |      0.092041 |
| InternetService_Fiber optic           |      0.05098  |
| Contract_Two year                     |      0.036877 |
| Partner                               |      0.032354 |
| Dependents                            |      0.029361 |
| gender                                |      0.022262 |
| MultipleLines_Yes                     |      0.022215 |
| Contract_One year                     |      0.019544 |
| PaymentMethod_Credit card (automatic) |      0.017213 |
| StreamingTV_Yes                       |      0.016972 |
| PaperlessBilling                      |      0.016523 |
| OnlineBackup_Yes                      |      0.015825 |
| StreamingMovies_Yes                   |      0.01535  |
| PaymentMethod_Mailed check            |      0.014472 |
| TechSupport_Yes                       |      0.014431 |
| DeviceProtection_Yes                  |      0.014106 |
| OnlineSecurity_Yes                    |      0.013916 |
| SeniorCitizen                         |      0.012779 |
| DeviceProtection_NoService            |      0.007461 |
| MultipleLines_NoService               |      0.006696 |
| PhoneService                          |      0.006374 |
| OnlineSecurity_NoService              |      0.005964 |
| InternetService_No                    |      0.004967 |
| TechSupport_NoService                 |      0.004803 |
| OnlineBackup_NoService                |      0.004712 |
| StreamingMovies_NoService             |      0.004272 |
| StreamingTV_NoService                 |      0.003195 |

## Conclusión Estratégica y Estrategias de Retención

El análisis confirma que el **ciclo de vida del cliente (tenure y tipo de contrato)**, las **características del servicio (Fibra Óptica)** y el **comportamiento de pago (Cheque Electrónico)** son los principales impulsores de la cancelación.

Para abordar estos factores, se proponen estrategias como:

-   Programas de fidelización temprana para clientes nuevos.
-   Incentivos para migrar a contratos a largo plazo.
-   Investigación y mejora de la experiencia con el servicio de Fibra Óptica.
-   Optimización del proceso de pago electrónico.
-   Implementación de un sistema de alerta temprana basado en los modelos predictivos para contactar proactivamente a clientes de alto riesgo.

## Instrucciones para Ejecutar el Cuaderno

Para ejecutar este cuaderno y replicar el análisis:

1.  Abre el cuaderno en Google Colab.
2.  Ejecuta las celdas de código secuencialmente.
3.  Asegúrate de tener conexión a internet para descargar los datos desde la URL de GitHub.
4.  Instala las bibliotecas necesarias ejecutando las celdas con comandos `%pip install`. Las principales bibliotecas utilizadas son `pandas`, `scikit-learn`, `imblearn`, `matplotlib` y `seaborn`.
