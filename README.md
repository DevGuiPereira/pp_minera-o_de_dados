# 🎬 Projeto Prático: Previsão de Churn em Streaming

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![Status](https://img.shields.io/badge/Status-Concluído-success?style=for-the-badge)
![Lib](https://img.shields.io/badge/Lib-Scikit--Learn-orange?style=for-the-badge)

## 📄 Sobre o Projeto

Este repositório contém o **Projeto Prático (PP)** da disciplina de **Mineração de Dados**. 

O objetivo foi desenvolver um pipeline completo de Ciência de Dados para analisar o comportamento de clientes de uma plataforma de streaming e prever a probabilidade de **Churn** (cancelamento da assinatura).

### 🎯 Objetivo de Negócio
Identificar precocemente clientes com alto risco de cancelamento para que o time de marketing possa realizar ações preventivas de retenção.

---

## 👥 Integrantes do Grupo

| Nome | GitHub |
|------|--------|
| Kawan Silva | [@ksilva-kwn](https://github.com/ksilva-kwn) |
| Guilherme Pereira | [@DevGuiPereira](https://github.com/DevGuiPereira) |
| Bruno Rezebde | [@BVRezende](https://github.com/BVRezende) |
| Pedro Teixeira | [@PedroTeixeira027](https://github.com/PedroTeixeira027) |
| Pedro Vargas | [@PedroAngeloVargas](https://github.com/PedroAngeloVargas) |

---

## 🗂️ Sobre os Dados

O dataset utilizado (`dados_streaming_ficticio.csv`) simula uma base de assinantes com as seguintes variáveis:

* **Target (Alvo):** `Churn` (0 = Não cancelou, 1 = Cancelou).
* **Variáveis Numéricas:** Idade, Meses como Cliente, Fatura Mensal, Num Chamados Suporte.
* **Variáveis Categóricas:** Gênero, Plano (Básico, Padrão, Premium), Atraso Pagamento.

---

## ⚙️ Pipeline de Desenvolvimento

O projeto foi dividido em dois scripts principais para modularizar o processo:

### 1. ETL e Pré-processamento (`etl.py`)
Nesta etapa, tratamos a qualidade dos dados brutos:
* **Limpeza de Ruído:** Remoção de idades negativas (inconsistentes).
* **Imputação de Dados:** Preenchimento de valores nulos (`NaN`) utilizando a **mediana** das colunas (para evitar sensibilidade a outliers).
* **Seleção de Features:** Remoção de colunas irrelevantes para o modelo, como o `ID_Cliente`.
* **Exportação:** Geração do arquivo tratado `dados_streaming_limpos.csv`.

### 2. Modelagem e Machine Learning (`modelagem.py`)
Utilizamos a biblioteca `scikit-learn` para treinar e avaliar modelos preditivos:
* **Feature Engineering:** Transformação de variáveis categóricas (`Genero`, `Plano`) em numéricas ordinais.
* **Algoritmos Testados:**
    1.  **Regressão Logística:** Como baseline para entender a linearidade dos dados.
    2.  **Random Forest Classifier:** Para capturar relações não-lineares complexas.
* **Avaliação:** Uso de Matriz de Confusão para verificar falsos positivos e falsos negativos.
* **Persistência:** O modelo final treinado foi salvo no arquivo `modelo_churn.pkl`.

---

## 📊 Principais Resultados

A análise da **Importância das Variáveis** (Feature Importance) do Random Forest indicou que os fatores que mais influenciam o cancelamento são:
1.  **Atraso no Pagamento:** Clientes que atrasam faturas têm alta correlação com churn.
2.  **Número de Chamados ao Suporte:** Alto volume de reclamações indica insatisfação.
3.  **Fatura Mensal:** Valores mais altos podem gerar maior sensibilidade ao preço.

*(Obs: Os gráficos gerados pelo script `modelagem.py` mostram as matrizes de confusão detalhadas).*

---

## 🚀 Como Executar o Projeto

Siga os passos abaixo para rodar a análise na sua máquina:

### 1. Instalar Dependências
Certifique-se de ter o Python instalado e rode:
```bash
pip install -r requirements.txt
