import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

df = pd.read_csv("dados_streaming_limpos.csv", decimal=",")

genero_ordem = {"M": 1, "F": 2, "Outro": 3}

plano_ordem = {"Básico": 1, "Padrão": 2, "Premium": 3, "Premium_Plus": 4}

df["Genero_Ordem"] = df["Genero"].map(genero_ordem)
df["Plano_Ordem"] = df["Plano"].map(plano_ordem)

df_novo = df[
    [
        "Idade",
        "Genero_Ordem",
        "Meses_Como_Cliente",
        "Fatura_Mensal",
        "Plano_Ordem",
        "Num_Chamados_Suporte",
        "Atraso_Pagamento",
        "Churn",
    ]
]

Y = df_novo["Churn"]
X = df_novo.drop("Churn", axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)

modelo_log = LogisticRegression(max_iter=1000)
modelo_rf = RandomForestClassifier(n_estimators=1000)

modelo_log.fit(X_train, Y_train)
modelo_rf.fit(X_train, Y_train)

previsoes_log = modelo_log.predict(X_test)
previsoes_rf = modelo_rf.predict(X_test)

log = accuracy_score(Y_test, previsoes_log)
rf = accuracy_score(Y_test, previsoes_rf)

print("Acurácia Regressão: ", log)
print("Acurácia Random: ", rf)

log_matriz = confusion_matrix(Y_test, previsoes_log)
rf_matriz = confusion_matrix(Y_test, previsoes_rf)

print(log_matriz)
print(rf_matriz)

def plotar_matriz(modelo, nome_modelo, X_teste, Y_teste): #função para plotar a matriz de confusão
    # Faz as previsões
    y_pred = modelo.predict(X_teste)
    # Gera a matriz matemática
    cm = confusion_matrix(Y_teste, y_pred)
    
    # Cria o desenho
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Matriz de Confusão: {nome_modelo}')
    plt.xlabel('O que o Robô previu')
    plt.ylabel('O que aconteceu de verdade')
    plt.show()

# Chamando a função para os dois modelos
plotar_matriz(modelo_log, "Regressão Logística", X_test, Y_test)
plotar_matriz(modelo_rf, "Random Forest", X_test, Y_test)



plt.figure(figsize=(10, 6))
importancias = modelo_rf.feature_importances_
colunas = X.columns

# Criando um DataFrame só para organizar o gráfico
df_importancia = pd.DataFrame({'Variavel': colunas, 'Importancia': importancias})
df_importancia = df_importancia.sort_values(by='Importancia', ascending=False)

sns.barplot(x='Importancia', y='Variavel', data=df_importancia, palette='viridis')
plt.title('O que mais causa Churn (Cancelamento)?')
plt.xlabel('Peso da Importância')
plt.ylabel('Variáveis')
plt.show()

print("Classificação log: \n", classification_report(Y_test, previsoes_log))
print("Classificação rf: \n", classification_report(Y_test, previsoes_rf))

joblib.dump(modelo_rf, "modelo_churn.pkl")
