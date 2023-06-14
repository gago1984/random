# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

# -*- coding: utf-8 -*-
"""
Grupo: Alta Data
Integrantes: Mariano Papaleo, Gaston Ariel Sanchez, Juan Pablo Aquilante

"""
#Carga de archivos

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

df = pd.read_csv('/home/clinux01/Descargas/mnist_desarrollo.csv')
df_test = pd.read_csv('/home/clinux01/Descargas/mnist_test.csv')
df_binario_test = pd.read_csv('/home/clinux01/Descargas/mnist_binario_test.csv')

#%%
# =============================================================================
#Ejercicio 1
#Realizar un análisis exploratorio de los datos. Ver, entre otras cosas,
#cantidad de datos, cantidad y tipos de atributos, cantidad de clases de la
#variable de interés (el dígito) y otras características que consideren
#relevantes. ¿Cuáles parecen ser atributos relevantes? ¿Cuáles no? Se
#pueden hacer gráficos para abordar estas preguntas.
# =============================================================================

#La primera columna indica el digito, las demas son los pixeles de la imagen.renombramos columnas:
#Cada pixel será representado de la forma i-j: indicando fila y columna
cols = ["digito"]
for i in range(28):
    for j in range(28):
        elem = str(i) + "-" + str(j)
        cols.append(elem)


df = df.rename(columns=dict(zip(df.columns, cols)))
#%%

#Exploracion de datos
def graficar(df,fila):
    plt.imshow(np.array(df.iloc[fila,1:]).reshape(28,28),cmap='Greys')
    numero = df.iloc[fila,0]
    plt.title(f'Numero: {numero}')
    plt.show()

#Las proporciones de los dígitos en todo el dataset
cant_de_imgs_por_num = df["digito"].value_counts().sort_index()
porc_de_imgs_por_num = round(cant_de_imgs_por_num / len(df) * 100,2)
proporcion_digitos = pd.DataFrame({'cant': cant_de_imgs_por_num,'% cant.':porc_de_imgs_por_num})
proporcion_digitos.index.name = 'Dígito'
proporcion_digitos = proporcion_digitos.sort_values(by='cant')
#%% vemos la distribucion de los digitos en el dataset

# Calcular las ocurrencias de cada dígito
ocurrencias = df['digito'].value_counts()
# Calcular el total de ocurrencias
total_ocurrencias = ocurrencias.sum()
# Calcular los porcentajes de ocurrencia
porcentajes = (ocurrencias / total_ocurrencias) * 100
# Crear la figura y el eje del gráfico
fig, ax = plt.subplots()
digitos = [str(d) for d in ocurrencias.index]
plt.bar(digitos,ocurrencias.values)
#Agregar etiquetas de texto en cada barra
for i in range(len(ocurrencias)):
    ax.text(i , ocurrencias.values[i],
            f"{porcentajes.values[i]:.2f}%",
            ha='center',
            va='top',
            rotation=60,
            c="azure")
# Configurar etiquetas y título del gráfico
ax.set_xlabel('Dígitos')
ax.set_ylabel('Ocurrencias')
ax.set_title('Ocurrencias de dígitos')
plt.savefig('./data/Ocurrencias_digitos.png')
plt.show()
#%%
df_sin_label = np.array(df.iloc[:,1:])
imgs = df_sin_label.reshape(-1,28, 28)

# Calcular el promedio de cada pixel
matriz_prom = np.mean(imgs, axis=0)


plt.imshow(matriz_prom, cmap='hot')
plt.colorbar()
plt.title('Mapa de calor de dispersión')
plt.show()

#%%
# =============================================================================
# Ejercicio 2
#Construir un dataframe con el subconjunto que contiene solamente los
#dígitos 0 y 1.
# =============================================================================

con_0s_y_1s = df[ (df["digito"]==0) | (df["digito"]==1) ]
#Inicializo imagenes promedio para 0s y 1s
con_0s= df[(df["digito"]==0)]
con_1s= df[(df["digito"]==1)]

imgs_con_0 = np.array(con_0s.iloc[:,1:])
ceros = imgs_con_0.reshape(-1,28, 28)
prom_ceros = np.mean(ceros, axis=0)
prom_ceros_dif = prom_ceros

imgs_con_1 = np.array(con_1s.iloc[:,1:])
unos = imgs_con_1.reshape(-1,28, 28)
prom_unos = np.mean(unos, axis=0)
prom_unos_dif = prom_unos

plt.imshow(prom_ceros, cmap='hot')
plt.colorbar()
plt.title('Imagen promedio para el digito 0')
plt.axis('off')
plt.show()

plt.imshow(prom_unos, cmap='hot')
plt.colorbar()
plt.title('Imagen promedio para el digito 1')
plt.axis('off')
plt.show()
#%% Matriz diferencial unos

umbralCeros = 50

for i in range(len(prom_unos_dif)):
    for j in range(len(prom_unos_dif)):
        if(prom_ceros[i][j]>umbralCeros):
            prom_unos_dif[i][j]=0

plt.figure(figsize=(6, 4))
plt.imshow(prom_unos_dif, cmap='hot')
plt.colorbar()
plt.title('Imagen promedio (1) sin representativos del 0')
plt.show()

umbralUnos=200

for i in range(0,len(prom_unos_dif)):
    for j in range(0,len(prom_unos_dif[0])):
        if(prom_unos[i][j]<umbralUnos):
            prom_unos_dif[i][j]=0

plt.imshow(prom_unos_dif, cmap='hot')
plt.colorbar()
plt.title('Imagen promedio (1) diferencial')
plt.show()
#busco las columnas relevantes
array_unos_dif=prom_unos_dif.flatten()
pixeles_sign_unos=np.argwhere(array_unos_dif>0)+1

#%% Matriz diferencial ceros

imgs_con_0 = np.array(con_0s.iloc[:,1:])
ceros = imgs_con_0.reshape(-1,28, 28)
prom_ceros = np.mean(ceros, axis=0)
prom_ceros_dif = prom_ceros

imgs_con_1 = np.array(con_1s.iloc[:,1:])
unos = imgs_con_1.reshape(-1,28, 28)
prom_unos = np.mean(unos, axis=0)
prom_unos_dif = prom_unos

umbralUnos=50

for i in range(len(prom_ceros_dif)):
    for j in range(len(prom_ceros_dif)):
        if(prom_unos[i][j]>umbralUnos):
            prom_ceros_dif[i][j]=0

plt.figure(figsize=(6, 4))
plt.imshow(prom_ceros_dif, cmap='hot')
plt.colorbar()
plt.title('Imagen promedio (0) sin representativos del 1')
plt.show()

umbralCeros=120

for i in range(0,len(prom_ceros_dif)):
    for j in range(0,len(prom_ceros_dif[0])):
        if(prom_ceros[i][j]<umbralCeros):
            prom_ceros_dif[i][j]=0

plt.imshow(prom_ceros_dif, cmap='hot')
plt.colorbar()
plt.title('Imagen promedio (0) diferencial')
plt.show()

array_ceros_dif=prom_ceros_dif.flatten()
pixeles_sign_ceros=np.argwhere(array_ceros_dif>0)+1            

#%%
# =============================================================================
# Ejercicio 3
#Para este subconjunto de datos, ver cuántas muestras se tienen y
#determinar si está balanceado entre las clases.
# =============================================================================

#Graficamos una imagen al azar del subconjunto de datos generado
fila = np.random.randint(0, len(con_0s_y_1s))
graficar(con_0s_y_1s,fila)

# Devuelve ceros y unos como esperabamos

# Esto calcula para cada pixel la suma total de los valores que tienen a lo largo de todas las imagenes
# Arma un dataframe cuya primera columna es el pixel, y la segunda es el valor total sumado

def suma_columnas(df):
    suma_columna = []
    a = pd.DataFrame()
    for i in range(len(df.columns)-1):
        suma_columna.append(df.iloc[1:,i].sum())
    a['pixel'] = df.columns
    a = a.drop(0)
    a['suma_de_color'] = suma_columna
    return a

columnas = suma_columnas(df)
columnas_ceros_y_unos = suma_columnas(con_0s_y_1s)

# Dataframe de pixeles que tienen unicamente el valor 0 a lo largo de todas las imagenes
sub = columnas[columnas['suma_de_color'] == 0]
print("Proporcion de pixeles que tienen unicamente el valor 0 a lo largo de todas las imagenes del dataset")
len(sub)/(len(df.columns)-1) * 100 # 66/784*100

#Vemos cuantas muestras se tienen
cant_de_imgs_por_num = con_0s_y_1s["digito"].value_counts().sort_index()
porc_de_imgs_por_num = round(cant_de_imgs_por_num / len(con_0s_y_1s) * 100,2)
cant = pd.DataFrame({'cantidad': cant_de_imgs_por_num})
porcentajes = pd.DataFrame({'% subconj':porc_de_imgs_por_num,'% dataset original':round(cant_de_imgs_por_num / len(df) * 100,2)})
cant.index.name = 'Dígito'
tabla = pd.concat([cant, porcentajes], axis=1)


#%%
# =============================================================================
# Ejercicio 4
#Ajustar un modelo de knn considerando pocos atributos, por ejemplo 3.
#Probar con distintos conjuntos de 3 atributos y comparar resultados.
#Analizar utilizando otras cantidades de atributos.
# =============================================================================
#%%

# Elegimos 3 atributos(pixeles) aleatoriamente de nuestra lista de pixeles que distinguen al CERO del UNO

filas = pixeles_sign_ceros.shape[0]
filas_aleatorias = np.random.choice(filas, size=3, replace=False)
atributos_aleatorios_ceros = pixeles_sign_ceros[filas_aleatorias] 
print(atributos_aleatorios_ceros)

X = con_0s_y_1s.iloc[:,np.squeeze(atributos_aleatorios_ceros)]
Y = con_0s_y_1s.digito

Nrep = 5
valores_n = [5,10,15,20]

resultados_test = np.zeros((Nrep, len(valores_n)))
resultados_train = np.zeros((Nrep, len(valores_n)))
cms = []


for i in range(Nrep):
    j=0
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
    while j < len(valores_n):
        k = valores_n[j]
        model = KNeighborsClassifier(n_neighbors = k)
        model.fit(X_train, Y_train) 
        Y_pred = model.predict(X_test)
        Y_pred_train = model.predict(X_train)
        cm = metrics.confusion_matrix(Y_test, Y_pred)
        acc_test = metrics.accuracy_score(Y_test, Y_pred)
        acc_train = metrics.accuracy_score(Y_train, Y_pred_train)
        resultados_test[i, j] = acc_test
        resultados_train[i, j] = acc_train
        cms.append(cm)
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                        display_labels=model.classes_)
        disp.plot()
        print("Exactitud del modelo:", metrics.accuracy_score(Y_test, Y_pred))
        print("Precisión del modelo: ", metrics.precision_score(Y_test, Y_pred, pos_label=1))
        print("Sensitividad del modelo: ", metrics.recall_score(Y_test, Y_pred, pos_label=1))
        print("F1 Score del modelo: ", metrics.f1_score(Y_test, Y_pred, pos_label=1))
        j=j+1

#%%
# Promedio de resultados de train y test

promedios_train = np.mean(resultados_train, axis = 0) 
promedios_test = np.mean(resultados_test, axis = 0) 
#%%
# Grafico de la exactitud

plt.figure(figsize=(4,3),dpi=100)
plt.plot(valores_n, promedios_train, label = 'Train')
plt.plot(valores_n, promedios_test, label = 'Test')
plt.legend()
plt.title('Exactitud del modelo de knn')
plt.xlabel('Cantidad de vecinos')
plt.ylabel('Exactitud (accuracy)')

#%%

# Elegimos 3 atributos(pixeles) aleatoriamente de nuestra lista de pixeles que distinguen al UNO del CERO

filas = pixeles_sign_unos.shape[0]
filas_aleatorias = np.random.choice(filas, size=3, replace=False)
atributos_aleatorios_unos = pixeles_sign_unos[filas_aleatorias] 
print(atributos_aleatorios_unos)

X = con_0s_y_1s.iloc[:,np.squeeze(atributos_aleatorios_unos)]
Y = con_0s_y_1s.digito

Nrep = 5
valores_n = [5,10,15,20]

resultados_test = np.zeros((Nrep, len(valores_n)))
resultados_train = np.zeros((Nrep, len(valores_n)))
cms = []


for i in range(Nrep):
    j=0
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
    while j < len(valores_n):
        k = valores_n[j]
        model = KNeighborsClassifier(n_neighbors = k)
        model.fit(X_train, Y_train) 
        Y_pred = model.predict(X_test)
        Y_pred_train = model.predict(X_train)
        cm = metrics.confusion_matrix(Y_test, Y_pred)
        acc_test = metrics.accuracy_score(Y_test, Y_pred)
        acc_train = metrics.accuracy_score(Y_train, Y_pred_train)
        resultados_test[i, j] = acc_test
        resultados_train[i, j] = acc_train
        cms.append(cm)
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                        display_labels=model.classes_)
        disp.plot()
        print("Exactitud del modelo:", metrics.accuracy_score(Y_test, Y_pred))
        print("Precisión del modelo: ", metrics.precision_score(Y_test, Y_pred, pos_label=1))
        print("Sensitividad del modelo: ", metrics.recall_score(Y_test, Y_pred, pos_label=1))
        print("F1 Score del modelo: ", metrics.f1_score(Y_test, Y_pred, pos_label=1))
        j=j+1

#%%
# Promedio de resultados de train y test

promedios_train = np.mean(resultados_train, axis = 0) 
promedios_test = np.mean(resultados_test, axis = 0) 
#%%
# Grafico de la exactitud

plt.figure(figsize=(4,3),dpi=100)
plt.plot(valores_n, promedios_train, label = 'Train')
plt.plot(valores_n, promedios_test, label = 'Test')
plt.legend()
plt.title('Exactitud del modelo de knn')
plt.xlabel('Cantidad de vecinos')
plt.ylabel('Exactitud (accuracy)')

#%%
y=2
promedios_por_atributos_train = []
promedios_por_atributos_test = []
while y < 12:
    filas = pixeles_sign_ceros.shape[0]
    filas_aleatorias = np.random.choice(filas, size=y, replace=False)
    atributos_aleatorios_ceros = pixeles_sign_ceros[filas_aleatorias] 
    print(atributos_aleatorios_ceros)
    
    X = con_0s_y_1s.iloc[:,np.squeeze(atributos_aleatorios_ceros)]
    Y = con_0s_y_1s.digito
    
    Nrep = 5
    valores_n = [12]
    
    resultados_test = np.zeros((Nrep, len(valores_n)))
    resultados_train = np.zeros((Nrep, len(valores_n)))
    
    
    for i in range(Nrep):
        j=0
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
        while j < len(valores_n):
            k = valores_n[j]
            model = KNeighborsClassifier(n_neighbors = k)
            model.fit(X_train, Y_train) 
            Y_pred = model.predict(X_test)
            Y_pred_train = model.predict(X_train)
            acc_test = metrics.accuracy_score(Y_test, Y_pred)
            acc_train = metrics.accuracy_score(Y_train, Y_pred_train)
            resultados_test[i, j] = acc_test
            resultados_train[i, j] = acc_train
            j=j+1
    
    promedios_train = np.mean(resultados_train, axis = 0) 
    promedios_test = np.mean(resultados_test, axis = 0) 
    promedios_por_atributos_train.append(promedios_train)
    promedios_por_atributos_test.append(promedios_test)
    y=y+1

h = [2,3,4,5,6,7,8,9,10,11]

plt.figure(figsize=(4,3),dpi=100)
plt.plot(h, promedios_por_atributos_train , label = 'Train')
plt.plot(h, promedios_por_atributos_test , label = 'Test')
plt.legend()
plt.title('Exactitud del modelo de knn')
plt.xlabel('Cantidad de atributos')
plt.ylabel('Exactitud (accuracy)')


#%%
# =============================================================================
# Ejercicio 5
# Para comparar modelos, utilizar validación cruzada. Comparar modelos
# con distintos atributos y con distintos valores de k (vecinos). Para el análisis
# de los resultados, tener en cuenta las medidas de evaluación (por ejemplo,
# la exactitud) y la cantidad de atributos.
# =============================================================================
X = con_0s_y_1s.iloc[:,[490,462,380]]
Y = con_0s_y_1s.digito

# CROSS VALIDATION CON KNN Y TREE CON CROSS_VAL_SCORE

clf = DecisionTreeClassifier(random_state=42)
knn = KNeighborsClassifier(n_neighbors=12)

k_folds = KFold(n_splits = 5)

scores = cross_val_score(knn, X, Y, cv = k_folds)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))

scores = cross_val_score(clf, X, Y, cv = k_folds)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))

#%%

# CROSS VALIDATION CON KNN CON CROSS_VALIDATE

cv_results = cross_validate(knn, X, Y, cv=10,return_train_score=True)
sorted(cv_results.keys())
print('Test score:', cv_results['test_score'])
print('Train score:', cv_results['train_score'])
print('Promedio Test score: ', np.mean(cv_results['test_score']))
print('Promedio Train score: ', np.mean(cv_results['train_score']))

#%%

# CROSS VALIDATION CON KFOLD.SPLIT (FALLO)

kf = KFold(n_splits=5)

# Inicializar una lista para almacenar los resultados de precisión
accuracy_scores = []

for train_index, test_index in kf.split(X):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    # Inicializar y ajustar el clasificador KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, Y_train)
    
    # Realizar predicciones en el conjunto de prueba
    Y_pred = knn.predict(X_test)
    
    # Calcular la precisión y agregarla a la lista de resultados
    accuracy = accuracy_score(Y_test, Y_pred)
    accuracy_scores.append(accuracy)

# Calcular el promedio de los resultados de precisión
average_accuracy = sum(accuracy_scores) / len(accuracy_scores)

# Imprimir el resultado final
print("Precisión promedio:", average_accuracy)
#%%
# =============================================================================
# Ejercicio 6
# Trabajar nuevamente con el dataset de todos los dígitos. Ajustar un
# modelo de árbol de decisión. Analizar distintas profundidades.
# =============================================================================
import time

#Sin definir profundidad(sin prepruning)

X = df.iloc[:,1:]
Y = df['digito']
clf = DecisionTreeClassifier(criterion = "entropy")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
Y_pred_train = clf.predict(X_train)
acc_test = metrics.accuracy_score(Y_test, Y_pred)
acc_train = metrics.accuracy_score(Y_train, Y_pred_train)
print("Criterio: entropy")
print("Test:",acc_test)
print("Train:",acc_train)
print("Profundidad:",clf.tree_.max_depth) # Por lo general entre 20 y 22
#%%
def entrenar_y_graficar(X,Y,criterio,Nrep,k,nombre_archivo):
    valores_k = range(4,k+1)
    resultados_test = np.zeros( (Nrep,k))
    resultados_train = np.zeros( (Nrep,k))

    for i in range(Nrep):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
        for j in valores_k:
            model = DecisionTreeClassifier(criterion = criterio,max_depth = j)
            model.fit(X_train, Y_train)
            Y_pred = model.predict(X_test)
            Y_pred_train = model.predict(X_train)
            acc_test = metrics.accuracy_score(Y_test, Y_pred)
            acc_train = metrics.accuracy_score(Y_train, Y_pred_train)
            resultados_test[i,j-1] = acc_test
            resultados_train[i,j-1] = acc_train
    
    promedios_train = np.mean(resultados_train, axis = 0) #A lo largo de cada columna
    promedios_test = np.mean(resultados_test, axis = 0)
    
#    plt.plot(valores_k, promedios_train,marker="o",label = 'Train',drawstyle="steps-post")
#    plt.plot(valores_k, promedios_test, marker="o",label = 'Test',drawstyle="steps-post")
#    plt.legend()
#    title = "Accuracy segun profundidad, criterio:" + criterio
#    plt.title(title)
#    plt.xlabel('Profundidad')
#    plt.ylabel('Exactitud (accuracy)')
#    archive = "./data/" + nombre_archivo + ".png"
#    plt.savefig(archive)
#    plt.show()
# Iniciar el contador de tiempo
start_time = time.time()

entrenar_y_graficar(X, Y, "entropy",3, 20, "entropy_k_20_n_5reps_")
entrenar_y_graficar(X, Y, "gini",3, 20, "entropy_k_20_5reps_") #21 minutos de ejecucion

end_time = time.time()
execution_time = end_time - start_time
print(f"Tiempo de ejecución: {execution_time} segundos")
#%% Analizamos distintas profundidades
def entrenar_hasta_prof_k(X,Y,criterio,k,nombre_archivo):
    valores_k = range(4,k+1)
    clfs = []
    #Particionamos el conjunto de entrenamiento en 30% test y 70% train
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
    for d in valores_k:
        clf = DecisionTreeClassifier(criterion = criterio,max_depth = d)
        clf.fit(X_train, Y_train)
        clfs.append(clf)
    #node_counts = [clf.tree_.node_count for clf in clfs]
    #depth = [clf.tree_.max_depth for clf in clfs]
    train_scores = [clf.score(X_train, Y_train) for clf in clfs]
    test_scores = [clf.score(X_test, Y_test) for clf in clfs]
    fig, ax = plt.subplots()
    ax.set_xlabel("profundidad")
    ax.set_ylabel("accuracy")
    title = "Accuracy segun profundidad, criterio:" + criterio
    ax.set_title(title)
    ax.plot(valores_k, train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(valores_k, test_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
#    archive = "./data/" + nombre_archivo + ".png"
#    plt.savefig(archive)
    plt.show()
    
X = df.iloc[:,1:]
Y = df['digito']

# Iniciar el contador de tiempo
start_time = time.time()
# Código que deseas medir

entrenar_hasta_prof_k(X,Y,"gini",20,"clf_hasta_20k_gini")
entrenar_hasta_prof_k(X,Y,"entropy",20,"clf_hasta_20k_entropy") #3 min de ejecucion
# Finalizar el contador de tiempo y calcular la duración
end_time = time.time()
execution_time = end_time - start_time

print(f"Tiempo de ejecución: {execution_time} segundos")
#%%
#Al parecer 12 es la profundidad optima
start_time = time.time()

clf = DecisionTreeClassifier(criterion = "entropy",max_depth = 12)
k_folds = KFold(n_splits = 10)
scores = cross_val_score(clf, X, Y, cv = k_folds)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))

end_time = time.time()
execution = end_time - start_time
execution_minutos = int(execution_time // 60)
execution_segundos = np.round(execution_time % 60,4)

print(f"Tiempo de ejecución: {execution_minutos} minutos {execution_segundos} segundos")
#%%
# =============================================================================
# Ejercicio 8
# Les daremos un conjunto de test el día de la entrega, para que puedan evaluar
# sus modelos y reportar la perfomance
# =============================================================================

# VEAMOS QUE TAL DECISIONTREECLASSIFIER PREDICE EL DF_TEST

cols = ["digito"]
for i in range(28):
    for j in range(28):
        elem = str(i) + "-" + str(j)
        cols.append(elem)


df_test = df_test.rename(columns=dict(zip(df_test.columns, cols)))

# Primero corregimos el nombre de las columnas del df_test para que coincida con el df original

#%%
""" PREDICCIÓN DF_TEST """

X = df.iloc[:,1:]
Y = df['digito']
clf = DecisionTreeClassifier(criterion = "entropy",max_depth=12)
clf.fit(X, Y)
Y_pred_train = clf.predict(X)
acc_train = metrics.accuracy_score(Y, Y_pred_train)
print("Predicción de digitos en el df_test con DecisionTreeClassifier, Criterio: entropy")
print("Train:",acc_train)

X_test = df_test.iloc[:,1:]
Y_test = df_test['digito']

Y_pred = clf.predict(X_test)
acc_test = metrics.accuracy_score(Y_test, Y_pred)

print("Test:",acc_test)

#%%

# VEAMOS QUE TAL KNN PREDICE EL DF_BINARIO_TEST

df_binario_test = df_binario_test.drop('Unnamed: 0',axis=1)

cols = ["digito"]
for i in range(28):
    for j in range(28):
        elem = str(i) + "-" + str(j)
        cols.append(elem)


df_binario_test = df_binario_test.rename(columns=dict(zip(df_binario_test.columns, cols)))

# Primero corregimos el nombre de las columnas del df_binario_test para que coincida con el df original

#%%
""" PREDICCIÓN DF_BINARIO_TEST """
""" Pixeles que distinguen al cero """

filas = pixeles_sign_ceros.shape[0]
filas_aleatorias = np.random.choice(filas, size=7, replace=False)
atributos_aleatorios_ceros = pixeles_sign_ceros[filas_aleatorias] 
print(atributos_aleatorios_ceros)

X = con_0s_y_1s.iloc[:,np.squeeze(atributos_aleatorios_ceros)]
Y = con_0s_y_1s.digito

k = 10 # Cantidad de vecinos optima

model = KNeighborsClassifier(n_neighbors = k)
model.fit(X, Y)
Y_pred_train = model.predict(X)
acc_train = metrics.accuracy_score(Y, Y_pred_train)
print("KNeighborsClassifier con 10 vecinos y 7 atributos, predicción de ceros en el df_binario_test")
print("Train:",acc_train)

X_test = df_binario_test.iloc[:,np.squeeze(atributos_aleatorios_ceros)]
Y_test = df_binario_test['digito']

Y_pred = model.predict(X_test)
acc_test = metrics.accuracy_score(Y_test, Y_pred)

print("Test:",acc_test)


#%%
""" PREDICCIÓN DF_BINARIO_TEST """
""" Pixeles que distinguen al uno """

filas = pixeles_sign_unos.shape[0]
filas_aleatorias = np.random.choice(filas, size=7, replace=False)
atributos_aleatorios_unos = pixeles_sign_unos[filas_aleatorias] 
print(atributos_aleatorios_unos)

X = con_0s_y_1s.iloc[:,np.squeeze(atributos_aleatorios_unos)]
Y = con_0s_y_1s.digito

k = 10 # Cantidad de vecinos optima

model = KNeighborsClassifier(n_neighbors = k)
model.fit(X, Y) 
Y_pred_train = model.predict(X)
acc_train = metrics.accuracy_score(Y, Y_pred_train)
print("KNeighborsClassifier con 10 vecinos y 7 atributos, predicción de unos en el df_binario_test")
print("Train:",acc_train)

X_test = df_binario_test.iloc[:,np.squeeze(atributos_aleatorios_unos)]
Y_test = df_binario_test['digito']

Y_pred = model.predict(X_test)
acc_test = metrics.accuracy_score(Y_test, Y_pred)

print("Test:",acc_test)
