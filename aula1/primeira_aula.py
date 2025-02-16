from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Carrega o dataset Iris
iris = load_iris()

# Divide os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)

# Cria o modelo KNN
knn = KNeighborsClassifier(n_neighbors=3)

# Treina o modelo
knn.fit(X_train, y_train)

# Faz previsões
y_pred = knn.predict(X_test)

# Avalia o modelo
accuracy = knn.score(X_test, y_test)
print(f"Acurácia: {accuracy}")

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Cria dados sintéticos
X, y = make_blobs(n_samples=300, centers=4, random_state=42)

# Cria o modelo KMeans
kmeans = KMeans(n_clusters=4)

# Treina o modelo
kmeans.fit(X)

# Obtém os центróides dos clusters
centroids = kmeans.cluster_centers_

# Imprime os центróides
print(f"Centróides: {centroids}")


from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Cria dados sintéticos
X, y = make_blobs(n_samples=300, centers=4, random_state=42)

# Cria o modelo KMeans
kmeans = KMeans(n_clusters=4)

# Treina o modelo
kmeans.fit(X)

# Obtém os центróides dos clusters
centroids = kmeans.cluster_centers_

# Imprime os центróides
print(f"Centróides: {centroids}")


import numpy as np

# Define o ambiente
n_states = 6
n_actions = 4
Q = np.zeros((n_states, n_actions))

# Define os parâmetros de aprendizado
learning_rate = 0.8
discount_factor = 0.95
epsilon = 0.1

# Define as recompensas
rewards = np.array([[-1, -1, -1, -1],
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1],
                    [-1, -1, -1, 100]])

# Treina o agente
for i in range(1000):
    state = np.random.randint(0, n_states)
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.randint(0, n_actions)
    else:
        action = np.argmax(Q[state])
    next_state = np.random.randint(0, n_states)
    Q[state, action] = Q[state, action] + learning_rate * (rewards[state, action] + discount_factor * np.max(Q[next_state]) - Q[state, action])

# Imprime a tabela Q
print(f"Tabela Q: {Q}")