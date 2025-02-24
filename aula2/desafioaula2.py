# Função para calcular a média de uma lista
def calcular_media(lista):
    return sum(lista) / len(lista)

# Função para calcular a regressão linear simples
def regressao_linear(x, y):
    # 2. Calcular as médias de x e y
    media_x = calcular_media(x)
    media_y = calcular_media(y)

    # 3. Inicializar as variáveis para os somatórios
    somatorio_xy = 0
    somatorio_xx = 0

    # 4. Utilizar um loop para calcular os somatórios
    for i in range(len(x)):
        somatorio_xy += (x[i] - media_x) * (y[i] - media_y)
        somatorio_xx += (x[i] - media_x) ** 2

    # 5. Calcular beta1 (coeficiente angular) e beta0 (coeficiente linear)
    beta_1 = somatorio_xy / somatorio_xx
    beta_0 = media_y - beta_1 * media_x

    # 6. Retornar os coeficientes
    return beta_0, beta_1

# 1. Definir as listas x e y
x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

# Chamar a função de regressão linear
beta_0, beta_1 = regressao_linear(x, y)

# 7. Imprimir os resultados
print(f"Coeficiente linear (β0): {beta_0:.2f}")
print(f"Coeficiente angular (β1): {beta_1:.2f}")

# Exemplo de previsão de y para um valor de x
x_novo = 6
y_estimado = beta_0 + beta_1 * x_novo
print(f"Para x = {x_novo}, o valor estimado de y é: {y_estimado:.2f}")
