# 1. CONFIGURAÇÃO INICIAL DO APP
# -------------------------------
# Importações necessárias para o funcionamento da aplicação
import streamlit as st  # Framework para criar aplicações web
import pandas as pd     # Manipulação de dados
import joblib          # Para carregar o modelo treinado
import numpy as np      # Operações numéricas
import os              # Para manipulação de caminhos

# Configurações iniciais da página
st.set_page_config(
    page_title="Previsão de Evasão Acadêmica",  # Título exibido na aba do navegador
    layout="centered"                          # Layout centralizado
)
st.title("Previsão de Evasão Acadêmica de Estudantes")  # Título principal da aplicação

# 2. CARREGAMENTO DO MODELO
# -------------------------------
# Função com cache para carregar o modelo apenas uma vez
# @st.cache_resource evita recarregar o modelo a cada interação do usuário
@st.cache_resource
def load_model():
    """
    Carrega o modelo treinado salvo no arquivo 'modelo_evasao.joblib'
    Retorna:
        O modelo de machine learning treinado e pronto para previsões
    """
    return joblib.load(os.path.join(os.path.dirname(__file__), 'modelo_evasao.joblib'))

# Carrega o modelo uma vez e reutiliza para todas as interações
clf = load_model()

# 3. INTERFACE DE ENTRADA DE DADOS
# -------------------------------
# Função que cria o formulário para coleta de dados do estudante
def user_input_features():
    """
    Cria a interface para entrada de dados do estudante usando widgets do Streamlit
    Retorna:
        DataFrame pandas com os dados formatados para o modelo
    """
    st.header("Insira os dados do estudante:")
    
    # Dados demográficos
    estado_civil = st.selectbox('Estado Civil', ['Solteiro', 'Casado', 'Divorciado', 'Viúvo'])
    curso = st.selectbox('Curso', [
        'Design de Animação e Multimédia',
        'Turismo',
        'Design de Comunicação',
        'Jornalismo e Comunicação',
        'Serviço Social (prestação nocturna)',
        'Gestão (presencial noturno)',
        'Enfermagem',
        'Serviço Social',
        'Gestão de Publicidade e Marketing',
        'Ensino Básico',
        'Enfermagem Veterinária',
        'Equincultura',
        'Higiene Oral',
        'Gestão',
        'Agronomia',
        'Tecnologias de Produção de Biocombustíveis',
        'Engenharia Informática'
    ])
    qualificacao_anterior = st.selectbox('Qualificação Anterior', [
    'Ensino Secundário',
    'Ensino Básico (3º Ciclo)',
    'Curso Técnico Superior Profissional',
    'Curso de Especialização Tecnológica',
    '11º Ano de Escolaridade - Não Concluído',
    'Ensino Superior - Licenciatura',
    'Ensino Superior - Licenciatura (1º Ciclo)',
    'Ensino Superior - Mestrado',
    'Outro - 11º Ano de Escolaridade',
    'Ensino Superior - Mestrado (2º Ciclo)',
    '10º Ano de Escolaridade - Não Concluído',
    'Frequência do Ensino Superior',
    '12º Ano de Escolaridade - Não Concluído',
    'Ensino Básico (2º Ciclo)',
    'Ensino Superior - Doutoramento',
    '10º Ano de Escolaridade'
])
    qualificacao_anterior_grau = st.number_input('Grau da Qualificação Anterior', min_value=0.0, max_value=20.0, value=12.0)
    nacionalidade = st.selectbox('Nacionalidade', [
    'Português', 'Romeno', 'Espanhol', 'Brasileiro', 'Santomense', 'Ucraniano',
    'Holandês', 'Moçambicano', 'Angolano', 'Mexicano', 'Italiano', 'Cabo-verdiano',
    'Turco', 'Moldávia (República da)', 'Guineense', 'Colombiano', 'Alemão',
    'Cubano', 'Russo', 'Inglês', 'Lituano'])
    nota_admissao = st.number_input('Nota de Admissão', min_value=0.0, max_value=20.0, value=10.0)
    necessidades_especiais = st.selectbox('Necessidades Especiais', [0, 1])
    devedor = st.selectbox('Devedor', [0, 1])
    mensalidades_em_dia = st.selectbox('Mensalidades em Dia', [0, 1])
    genero = st.selectbox('Gênero', ['Masculino', 'Feminino'])
    bolsista = st.selectbox('Bolsista', [0, 1])
    international = st.selectbox('International', [0, 1])
    uc1_creditado = st.number_input('UC 1º Semestre Creditado', min_value=0, max_value=10, value=5)
    uc1_inscrito = st.number_input('UC 1º Semestre Inscrito', min_value=0, max_value=10, value=5)
    uc1_avaliacoes = st.number_input('UC 1º Semestre Avaliações', min_value=0, max_value=10, value=5)
    uc1_aprovado = st.number_input('UC 1º Semestre Aprovado', min_value=0, max_value=10, value=5)
    uc1_grau = st.number_input('UC 1º Semestre Grau', min_value=0.0, max_value=20.0, value=10.0)
    uc1_sem_avaliacoes = st.number_input('UC 1º Semestre Sem Avaliações', min_value=0, max_value=10, value=0)
    uc2_creditado = st.number_input('UC 2º Semestre Creditado', min_value=0, max_value=10, value=5)
    uc2_inscrito = st.number_input('UC 2º Semestre Inscrito', min_value=0, max_value=10, value=5)
    uc2_avaliacoes = st.number_input('UC 2º Semestre Avaliações', min_value=0, max_value=10, value=5)
    uc2_aprovado = st.number_input('UC 2º Semestre Aprovado', min_value=0, max_value=10, value=5)
    uc2_grau = st.number_input('UC 2º Semestre Grau', min_value=0.0, max_value=20.0, value=10.0)
    uc2_sem_avaliacoes = st.number_input('UC 2º Semestre Sem Avaliações', min_value=0, max_value=10, value=0)
    taxa_desemprego = st.number_input('Taxa de Desemprego (%)', min_value=0.0, max_value=100.0, value=10.0)
    taxa_inflacao = st.number_input('Taxa de Inflação (%)', min_value=0.0, max_value=100.0, value=5.0)
    pib = st.number_input('PIB', min_value=0.0, max_value=10.0, value=1.0)
    
    # Organiza os dados em um dicionário
    data = {
        'EstadoCivil': estado_civil,
        'Curso': curso,
        'QualificacaoAnterior': qualificacao_anterior,
        'QualificacaoAnteriorGrau': qualificacao_anterior_grau,
        'Nacionalidade': nacionalidade,
        'NotaAdmissao': nota_admissao,
        'NecessidadesEspeciais': necessidades_especiais,
        'Devedor': devedor,
        'MensalidadesEmDia': mensalidades_em_dia,
        'Genero': genero,
        'Bolsista': bolsista,
        'International': international,
        'UnidadesCurriculares1SemestreCreditado': uc1_creditado,
        'UnidadesCurriculares1SemestreInscrito': uc1_inscrito,
        'UnidadesCurriculares1SemestreAvaliacoes': uc1_avaliacoes,
        'UnidadesCurriculares1SemestreAprovado': uc1_aprovado,
        'UnidadesCurriculares1SemestreGrau': uc1_grau,
        'UnidadesCurriculares1SemestreSemAvaliacoes': uc1_sem_avaliacoes,
        'UnidadesCurriculares2SemestreCreditado': uc2_creditado,
        'UnidadesCurriculares2SemestreInscrito': uc2_inscrito,
        'UnidadesCurriculares2SemestreAvaliacoes': uc2_avaliacoes,
        'UnidadesCurriculares2SemestreAprovado': uc2_aprovado,
        'UnidadesCurriculares2SemestreGrau': uc2_grau,
        'UnidadesCurriculares2SemestreSemAvaliacoes': uc2_sem_avaliacoes,
        'TaxaDesemprego': taxa_desemprego,
        'TaxaInflacao': taxa_inflacao,
        'PIB': pib
    }
    
    # Converte o dicionário em um DataFrame pandas
    return pd.DataFrame([data])

# 4. EXECUÇÃO DA PREVISÃO
# -------------------------------
# Chama a função de entrada de dados
df_input = user_input_features()

# Botão para disparar a previsão
if st.button('Prever Evasão'):
    # Realiza a previsão usando o modelo carregado
    pred = clf.predict(df_input)[0]
    proba = clf.predict_proba(df_input)[0,1]
    
    # Exibe o resultado da previsão
    if pred == 1:
        st.error(f"Probabilidade de evasão: {proba:.2%} (Desistente)")
    else:
        st.success(f"Probabilidade de evasão: {proba:.2%} (Graduado)")
