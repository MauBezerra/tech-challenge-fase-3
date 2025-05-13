# Previsão de Evasão Acadêmica

Este projeto utiliza machine learning para prever a evasão acadêmica de estudantes com base em dados históricos e socioeconômicos. Inclui um pipeline de modelagem em Python e uma interface interativa via Streamlit para facilitar a análise e a previsão.

## Funcionalidades
- Treinamento e avaliação de modelos de classificação (Random Forest)
- Interface web para inserção de dados e previsão de evasão
- Análise de desempenho do modelo (AUC ROC, matriz de confusão, etc.)
- Suporte a múltiplas nacionalidades, qualificações e variáveis relevantes do estudante

## Estrutura do Projeto
```
modelo_preditivo_binario/
├── app_streamlit.py         # Interface web (Streamlit)
├── pipeline_modelo.py      # Pipeline de modelagem e avaliação
├── dados/
│   └── StudentsPrepared.xlsx # Base de dados de estudantes
├── modelo_evasao.joblib    # Modelo treinado (gerado após execução do pipeline)
├── requirements.txt        # Dependências do projeto
└── README.md               # Este arquivo
```

## Instalação
1. Clone este repositório:
   ```bash
   git clone <url-do-repositorio>
   cd modelo_preditivo_binario
   ```
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## Como usar
### 1. Treinar/Re-treinar o modelo
Execute o pipeline de modelagem para treinar ou atualizar o modelo:
```bash
python3 pipeline_modelo.py
```
O arquivo `modelo_evasao.joblib` será gerado/atualizado.

### 2. Rodar a interface web
Execute o app Streamlit:
```bash
python3 -m streamlit run app_streamlit.py
```
Acesse [http://localhost:8501](http://localhost:8501) no navegador.

### 3. Inserir dados e obter previsão
Preencha o formulário com os dados do estudante e clique em "Prever Evasão" para obter a probabilidade de evasão ou graduação.

## Observações
- O arquivo `StudentsPrepared.xlsx` deve estar presente na pasta `dados/`.
- O modelo considera variáveis como nacionalidade, qualificação anterior, notas, situação financeira, entre outras.
- Certifique-se de que as versões das dependências sejam compatíveis com seu ambiente Python.

Desenvolvido por Mauricio Bezerra
