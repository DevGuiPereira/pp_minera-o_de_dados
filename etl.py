import pandas as pd
import numpy as np

df = pd.read_csv('dados_streaming_ficticio.csv')
df.loc[df['Idade'] < 0, 'Idade'] = np.nan

mediana_idade = df['Idade'].median()
mediana_fatura = df['Fatura_Mensal'].median()

df['Idade'] = df['Idade'].fillna(mediana_idade)
df['Fatura_Mensal'] = df['Fatura_Mensal'].fillna(mediana_fatura)
df['Fatura_Mensal'] = df['Fatura_Mensal'].round(2)

if 'ID_Cliente' in df.columns:
    df = df.drop(columns=['ID_Cliente'])

df.to_csv('dados_streaming_limpos.csv', index=False, float_format='%.2f')

print("Limpeza concluída!")
print(df.head())
print("\nVerificação de nulos (deve ser tudo zero):")
print(df.isnull().sum())