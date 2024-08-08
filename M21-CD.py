#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd

# Cargar los datos
data = pd.read_csv("recursos_humanos.csv")

# Mostrar las primeras filas para entender la estructura del dataframe
data.head(), data.info()


# In[8]:


# Recodificación de variables categóricas
data_encoded = pd.get_dummies(data, columns=['sales', 'salary'], drop_first=True)

# Análisis exploratorio: Verificar el balance de la variable objetivo 'left'
balance = data['left'].value_counts(normalize=True)

# Mostrar la distribución de la variable 'left'
balance


# In[10]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Separar las características y la variable objetivo
X = data_encoded.drop('left', axis=1)
y = data_encoded['left']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Estandarizar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Evaluar diferentes valores de k
k_values = range(1, 21)
cv_scores = []

# Cross-validation para cada valor de k
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# Determinar el valor óptimo de k
optimal_k = k_values[np.argmax(cv_scores)]
optimal_k, max(cv_scores)


# In[11]:


# Entrenar el modelo con el k óptimo
knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train_scaled, y_train)

# Predecir en el conjunto de prueba
y_pred = knn.predict(X_test_scaled)

# Generar la matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Crear un mapa de calor
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión para KNN (k=optimal_k)')
plt.xlabel('Predicción')
plt.ylabel('Verdad')
plt.show()


# In[12]:


from sklearn.metrics import roc_curve, roc_auc_score

# Obtener las probabilidades predichas
y_prob = knn.predict_proba(X_test_scaled)[:, 1]

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

# Graficar la curva ROC
plt.plot(fpr, tpr, label=f'ROC curve (area = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC para KNN (k=optimal_k)')
plt.legend(loc='best')
plt.show()


# In[ ]:




