# Estudos de aprendizado de máquina

Exercícios em Python organizados por aula, cobrindo regressão, classificação, agrupamento e avaliação de modelos. Os dados são exemplos sintéticos ou conjuntos incluídos no scikit-learn.

## Ambiente

Requer Python 3 e, para os gráficos, ambiente capaz de abrir janelas. As versões das bibliotecas não estão fixadas; os comandos abaixo são uma preparação sugerida.

```sh
git clone https://github.com/Kakykyyy/kaykysantos.4na.git
cd kaykysantos.4na
python -m venv .venv
```

Ative com `.venv\Scripts\Activate.ps1` no PowerShell ou `source .venv/bin/activate` no Linux/macOS.

```sh
python -m pip install numpy pandas scikit-learn statsmodels matplotlib seaborn
python aula2/desafioaula2.py
```

## Roteiro dos arquivos

| Arquivo | Conteúdo |
|---|---|
| `aula1/primeira_aula.py` | KNN no Iris, KMeans em dados sintéticos e atualização de tabela Q |
| `aula2/desafioaula2.py` | Regressão linear simples calculada manualmente |
| `aula2/primeiro_exemploaula2.py` | Regressão OLS com statsmodels e gráfico |
| `aula2/segundoaula2.py` | KNN no Iris, relatório e matriz de confusão |
| `aula2/terceiro.py` | Árvore de decisão no Iris e visualização |
| `aula3/classifcacao.py` | Acurácia, precisão, recall, F1 e matriz de confusão |
| `aula3/regressao.py` | MAE, MSE, RMSE e R² |
| `aula3/desafio3.py` | KNN no Wine, normalização, ROC, validação cruzada e comparação de K |
| `aula3/notebook.ipynb` | Notebook disponível para exploração |

Para executar outro exercício, substitua o caminho após `python`. O nome `classifcacao.py` está grafado dessa forma no repositório.

## Uso e interpretação

Leia o script, execute-o e observe as métricas no terminal. Nos exercícios com `plt.show()`, feche cada janela para permitir que a execução avance. Não há serviço web, configuração de banco nem treinamento de produção.

Algumas divisões de dados usam `random_state=42`; outras operações são aleatórias e podem variar. No desafio Wine, a validação cruzada recebe `X` sem o mesmo pipeline de normalização usado na separação treino/teste; seus resultados não são diretamente equivalentes. A curva ROC usa a primeira classe binarizada, apesar do rótulo visual “Classe 1”.

## Verificação manual

Execute primeiro `aula2/desafioaula2.py` e depois os exemplos de métricas em `aula3/`. Confirme a saída no terminal e a abertura dos gráficos nos demais exercícios. Não há testes automatizados nem versões de dependências homologadas nesta documentação.

## Licença

Não há arquivo LICENSE na árvore analisada.

---
Documentação baseada na leitura da branch principal em 2 de novembro de 2026. Os procedimentos não foram executados nesta revisão documental.
