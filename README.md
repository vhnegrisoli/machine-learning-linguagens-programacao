# Projeto: Machine Learning: Linguagens de Programacao 2004-2001

Projeto de Data Science e Machine Learning de análise de linguagens de programação de 2004 a 2021 obtidos a partir do seguinte dataset do [Kaggle](https://www.kaggle.com/muhammadkhalid/most-popular-programming-languages-since-2004).

### Tecnologias

* Python 3
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn

### Algoritmos

* Regressão Linear

Inicialmente, serão visualizados dados de séries temporais e regressão linear.

### Tratamento de dados

```python
df = pd.read_csv('Most Popular Programming Languages from 2004 to 2021 V4.csv')

def createDataFrameFor(df, colunas, colunaAtual):
    return pd.DataFrame(
        {
            'Date': df.Date,
            'Timestamp': map(lambda i : datetime.strptime(df["Date"][i], '%B %Y'), range(len(df.Date))),
            'Language': colunas[colunaAtual],
            'Value': df[df.columns[colunaAtual]]
        }
    )

colunas = df.columns

dados_tratados = createDataFrameFor(df, colunas, 1)

for coluna in range(1, len(colunas)):
    dados_tratados = pd.concat([dados_tratados, createDataFrameFor(df, colunas, coluna)])

dados_tratados.reset_index(drop=True, inplace=True)

dados_tratados['UnixTime'] = list(map(lambda i: (pd.to_datetime([dados_tratados['Timestamp'][i]]).astype(int) / 10**9)[0], range(len(dados_tratados['Date']))))

```

### Visualização dos dados

```python
  df_java = dados_tratados[dados_tratados['Language'] == 'Java']
sns.regplot(x="UnixTime", y="Value", data= df_java)
plt.gcf().set_size_inches(16, 6)
plt.ylabel('% de uso da linguagem Java')
plt.xlabel('Anos em Unix Time de 2004 a 2021')
plt.show()

X = df_java.UnixTime.values.reshape(-1, 1)
y = df_java.Value.values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

plt.scatter(X_test, y_test,  color='blue')
plt.plot(X_test, y_pred, color='red', linewidth=3)

plt.gcf().set_size_inches(16, 6)

plt.ylabel('% de uso da linguagem Java')
plt.xlabel('Anos em Unix Time de 2004 a 2021')

plt.show()
```

### Plots gerados

* Regressão Linear com Seaborn (SNS):

![Seaborn](https://github.com/vhnegrisoli/machine-learning-linguagens-programacao/blob/master/imgs/Figure_1.png)

* Regressão Linear com Scikit-Learn (LinearRegression) e Matplotlib:

![Matplotlib ScikitLearn](https://github.com/vhnegrisoli/machine-learning-linguagens-programacao/blob/master/imgs/Figure_2.png)

### Autor

* Victor Hugo Negrisoli
* Desenvolvedor Back-End Sênior | Analista de Dados
