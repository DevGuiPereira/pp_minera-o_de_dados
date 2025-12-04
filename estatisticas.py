import pandas as pd
import matplotlib.pyplot as plt

# Carregar a base já tratada pelo ETL
df = pd.read_csv("dados_streaming_limpos.csv")

print("\n===== VISÃO GERAL DA BASE =====")
print("Dimensões (linhas, colunas):", df.shape)
print("\nTipos de dados:")
print(df.dtypes)
print("\nValores nulos por coluna:")
print(df.isnull().sum())

print("\n===== ESTATÍSTICAS DESCRITIVAS =====")
print(df.describe())

print("\nMediana:")
print(df.median(numeric_only=True))

print("\n===== FREQUÊNCIA DAS VARIÁVEIS CATEGÓRICAS =====")
colunas_categoricas = df.select_dtypes(include=['object']).columns
for col in colunas_categoricas:
    print(f"\nFrequência de '{col}':")
    print(df[col].value_counts())

print("\n===== MATRIZ DE CORRELAÇÃO (NUMÉRICAS) =====")
print(df.corr(numeric_only=True))

# Heatmap de correlação (salvo como arquivo)
plt.figure(figsize=(10, 6))
plt.imshow(df.corr(numeric_only=True), cmap="viridis", interpolation="nearest")
plt.title("Matriz de Correlação")
plt.colorbar()
plt.savefig("correlacao.png")
plt.close()

print("\nHeatmap de correlação salvo como 'correlacao.png'.")

# Se existir a coluna Churn, análise adicional
if "Churn" in df.columns:
    print("\n===== ANÁLISE POR CHURN =====")
    print(df.groupby("Churn").mean(numeric_only=True))

    # gráfico simples da distribuição
    plt.figure(figsize=(5, 4))
    df["Churn"].value_counts().plot(kind="bar")
    plt.title("Distribuição de Churn")
    plt.xlabel("Churn (0 = não, 1 = sim)")
    plt.ylabel("Quantidade de clientes")
    plt.tight_layout()
    plt.savefig("distribuicao_churn.png")
    plt.close()

    print("\nGráfico de distribuição salvo como 'distribuicao_churn.png'.")