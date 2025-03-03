# Importando as bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Carregando o dataset Wine
data = load_wine()
X = data.data
y = data.target

# Dividindo os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizando as features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Implementando o classificador KNN com k = 5
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Calculando as métricas
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Exibindo as métricas
print(f"Acurácia: {accuracy:.4f}")
print("Matriz de Confusão:\n", conf_matrix)
print(f"Precisão: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Visualizando a Matriz de Confusão com Seaborn
plt.figure(figsize=(7,7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
plt.title("Matriz de Confusão")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()

# (Opcional) Plotando a Curva ROC e calculando a AUC para cada classe
# Binarizando as classes para multi-classe ROC
y_bin = label_binarize(y_test, classes=[0, 1, 2])
fpr, tpr, _ = roc_curve(y_bin[:,0], knn.predict_proba(X_test)[:,0])
roc_auc = auc(fpr, tpr)

# Plotando a Curva ROC para a classe 1
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'Classe 1 (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC para a Classe 1')
plt.legend(loc='lower right')
plt.show()

# Validação Cruzada com KNN para avaliar a média da acurácia
cv_scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print(f"Acurácia média da validação cruzada (10 folds): {cv_scores.mean():.4f}")

# Testando a acurácia para diferentes valores de K (K = 1 a 20)
k_values = range(1, 21)
accuracies = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

# Plotando a Acurácia vs. Número de Vizinhos (K)
plt.figure(figsize=(8,6))
plt.plot(k_values, accuracies, marker='o')
plt.xlabel('Número de Vizinhos (k)')
plt.ylabel('Acurácia')
plt.title('Acurácia vs. Número de Vizinhos (k)')
plt.grid(True)
plt.show()
