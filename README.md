<div align="center">

# Laboratório de Machine Learning

**Da regressão linear à avaliação de classificadores**

![Python](https://img.shields.io/badge/Python-20232a?style=flat-square) ![scikit-learn](https://img.shields.io/badge/scikit%2Dlearn-20232a?style=flat-square) ![NumPy](https://img.shields.io/badge/NumPy-20232a?style=flat-square) ![Pandas](https://img.shields.io/badge/Pandas-20232a?style=flat-square) ![Matplotlib](https://img.shields.io/badge/Matplotlib-20232a?style=flat-square)

[Sobre](#sobre) · [Como executar](#como-executar) · [Documentação](DOCUMENTACAO.md)

</div>

---

## Sobre

Repositório de estudos com exemplos de aprendizado de máquina organizados por aula. Os exercícios combinam implementação, métricas e visualizações para explorar o comportamento dos modelos.

## Destaques

- Regressão linear manual e com statsmodels.
- Classificação com KNN e árvores de decisão.
- Agrupamento com KMeans e exemplo de tabela Q.
- Métricas, matriz de confusão, curva ROC e validação cruzada.

## Como executar

Clone o repositório:

```sh
git clone https://github.com/Kakykyyy/kaykysantos.4na.git
cd kaykysantos.4na
```

Requer Python 3. Para os exercícios com gráficos, utilize um ambiente com suporte a janelas.

```sh
python -m venv .venv
# Ative o ambiente virtual antes de continuar.
python -m pip install numpy pandas scikit-learn statsmodels matplotlib seaborn
python aula2/desafioaula2.py
```

Para ativar o ambiente virtual, use `.venv\\Scripts\\Activate.ps1` no PowerShell ou `source .venv/bin/activate` no Linux/macOS.

## Organização

```text
aula1/ # KNN, KMeans e tabela Q
aula2/ # Regressão e classificação
aula3/ # Métricas e experimentos com Wine
```

## Roteiro de estudos

1. Comece por `aula2/desafioaula2.py` para acompanhar o cálculo da regressão.
2. Explore `aula2/segundoaula2.py` e `aula2/terceiro.py` para comparar classificadores no Iris.
3. Execute `aula3/classifcacao.py` e `aula3/regressao.py` para conhecer as métricas.
4. Avance para `aula3/desafio3.py` para experimentar KNN no Wine.

## Documentação técnica

O [guia técnico](DOCUMENTACAO.md) reúne configuração, comportamento dos dados, limitações conhecidas e verificações manuais. As instruções foram elaboradas a partir da leitura do código; a execução dos projetos não foi validada nesta revisão.

## Autor

Desenvolvido por [Kayky Erick · @Kakykyyy](https://github.com/Kakykyyy).
