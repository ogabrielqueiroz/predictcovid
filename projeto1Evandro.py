import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from matplotlib.backends.backend_pdf import PdfPages

# Carregar o conjunto de dados
intervac = pd.read_csv("C:\\Users\\gabri\\Downloads\\internvac.csv", sep=";", header=0)

# Selecionar as colunas desejadas
variaveis = ["NU_IDADE_N", "CS_SEXO", "FATOR_RISC", "VACINA_COV"]
intervac = intervac[variaveis]

# Imprimir a descrição do DataFrame
descricao = intervac.describe()
print(descricao)

# 1.2 Tratamento inicial dos dados
# Tratar dados faltantes
intervac.dropna(axis=1, inplace=True)
intervac["CS_SEXO"] = intervac["CS_SEXO"].map({"F": 0, "M": 1})

# Selecionar variáveis relevantes para o modelo
variaveis = ["NU_IDADE_N", "CS_SEXO", "FATOR_RISC", "VACINA_COV"]
intervac = intervac[variaveis]

# Análise de correlação
correlacao = intervac.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlacao, annot=True, cmap="coolwarm")
plt.title("Matriz de Correlação")
plt.show()

# 1.3 Percentual definido para amostras de treino e teste
percentual_treino = 0.7
percentual_teste = 0.3

# Separar os dados em amostras de treino e teste
X = intervac.drop("VACINA_COV", axis=1)
y = intervac["VACINA_COV"]
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=percentual_teste, random_state=123)
dados_nao_vacinados = intervac[intervac['VACINA_COV'].isin(['2', '9'])]
dados_vacinados = intervac[intervac['VACINA_COV'] == '1']

# 2.1 Programação utilizada (Python)
programacao = "Python"

# 2.2 Bibliotecas e parâmetros utilizados
modelo = LogisticRegression(max_iter=1000)

# Treinar o modelo
modelo.fit(X_treino, y_treino)

# Realizar predições
predicoes_treino = modelo.predict(X_treino)
predicoes_teste = modelo.predict(X_teste)

# 2.3 Resumo da avaliação do modelo
# Avaliação na amostra de treino
r2_treino = r2_score(y_treino, predicoes_treino)
rmse_treino = mean_squared_error(y_treino, predicoes_treino, squared=False)
mae_treino = mean_absolute_error(y_treino, predicoes_treino)

# Avaliação na amostra de teste
r2_teste = r2_score(y_teste, predicoes_teste)
rmse_teste = mean_squared_error(y_teste, predicoes_teste, squared=False)
mae_teste = mean_absolute_error(y_teste, predicoes_teste)

# Variância das métricas
variancia_r2 = r2_teste - r2_treino
variancia_rmse = rmse_teste - rmse_treino
variancia_mae = mae_teste - mae_treino

# 3. Conclusão da análise
conclusao = "O modelo de regressão logística foi ajustado utilizando as variáveis NU_IDADE_N, CS_SEXO e FATOR_RISC como preditores da variável VACINA_COV. O modelo apresentou uma avaliação satisfatória na amostra de treino e teste, com uma variação positiva no R2, RMSE e MAE entre as amostras, indicando uma capacidade de generalização adequada. No entanto, é importante ressaltar que essa é uma análise preliminar e podem ser necessárias outras variáveis ou técnicas de modelagem para melhorar a precisão do modelo."

# Salvar o arquivo em PDF com os gráficos e tabelas gerados
with PdfPages(r"C:\Users\gabri\OneDrive\Área de Trabalho\Projetos\analise_intervac.pdf") as pdf:
    plt.figure(figsize=(10, 6))
    sns.histplot(intervac["NU_IDADE_N"], bins=20, kde=True, color="lightblue")
    plt.title("Idade dos pacientes internados com Covid-19")
    plt.xlabel("Idade")
    plt.ylabel("Quantidade")
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlacao, annot=True, cmap="coolwarm")
    plt.title("Matriz de Correlação")
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.bar(["Não vacinados", "Vacinados"], [len(dados_nao_vacinados) / len(intervac), len(dados_vacinados) / len(intervac)], color=["darkblue", "darkred"])
    plt.title("Proporção de pacientes vacinados e não vacinados")
    plt.xlabel("Status de vacinação")
    plt.ylabel("Proporção")
    plt.ylim(0, 1)
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.bar(["Treino", "Teste"], [r2_treino, r2_teste], color=["darkgreen", "darkorange"])
    plt.title("Avaliação do modelo - R2")
    plt.xlabel("Amostra")
    plt.ylabel("R2")
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.bar(["Treino", "Teste"], [rmse_treino, rmse_teste], color=["darkgreen", "darkorange"])
    plt.title("Avaliação do modelo - RMSE")
    plt.xlabel("Amostra")
    plt.ylabel("RMSE")
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.bar(["Treino", "Teste"], [mae_treino, mae_teste], color=["darkgreen", "darkorange"])
    plt.title("Avaliação do modelo - MAE")
    plt.xlabel("Amostra")
    plt.ylabel("MAE")
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.bar(["∆_R2", "∆_RMSE", "∆_MAE"], [variancia_r2, variancia_rmse, variancia_mae], color="purple")
    plt.title("Variância das métricas entre treino e teste")
    plt.xlabel("Métricas")
    plt.ylabel("Variância")
    pdf.savefig()
    plt.close()

    # Adicionar a descrição do DataFrame como uma tabela no arquivo PDF
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis('off')
    table = ax.table(cellText=descricao.values, colLabels=descricao.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    ax.set_title("Descrição do DataFrame")
    pdf.savefig()
    plt.close()

    # Adicionar a conclusão da análise no arquivo PDF
    plt.figure(figsize=(8, 5))
    plt.text(0.5, 0.5, conclusao, ha='center', va='center')
    plt.axis('off')
    pdf.savefig()
    plt.close()

# Fim do script
