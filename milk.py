import streamlit as st
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from datetime import date
from io import StringIO

# Configuração da página
st.set_page_config(page_title="Sistema de Análise e Previsão de Séries Temporais", 
                   layout="wide", initial_sidebar_state="expanded")

# Título da aplicação com ícone
st.title("📈 Sistema de Análise e Previsão de Séries Temporais")

# Botão para abrir o menu de explicação
if st.button("ℹ️ Como funciona este relatório?"):
    with st.expander("Sobre o Relatório e o Funcionamento", expanded=True):
        st.markdown("""
            ### 📄 Explicação do Relatório de Análise e Previsão de Séries Temporais

            Este sistema permite realizar uma **análise detalhada e previsões de séries temporais** com base em dados históricos mensais. Utilizando técnicas de decomposição e o modelo **SARIMAX**, o aplicativo analisa padrões sazonais e tendências e gera previsões para períodos futuros definidos pelo usuário. 

            ### 🎯 Objetivo do Relatório
            O objetivo deste relatório é fornecer uma visão clara das **tendências passadas** e uma previsão confiável para o futuro. Isso é especialmente útil para negócios ou setores que dependem de ciclos sazonais ou tendências de longo prazo, como **vendas mensais, produção agrícola, indicadores econômicos** e outros tipos de séries temporais.

            ### 📝 Guia de Uso - Filtros e Parâmetros

            - **Upload de Dados (Arquivo CSV)**: Carregue um arquivo CSV contendo a série temporal que você deseja analisar. A série deve ser composta por valores mensais.
            - **Período Inicial da Série**: Define a data de início da série temporal. Esse valor é utilizado para construir o eixo de tempo do relatório.
            - **Meses para Previsão**: Define o número de meses para os quais o modelo irá gerar previsões (1 a 48 meses).

            ### 🔧 Configurações do Modelo SARIMAX

            - **AR Order (p)**: Número de termos autoregressivos.
            - **I Order (d)**: Grau de diferenciação para tornar os dados estacionários.
            - **MA Order (q)**: Número de termos de média móvel.
            - **Seasonal AR Order (P)**: Número de termos autoregressivos para a parte sazonal.
            - **Seasonal I Order (D)**: Grau de diferenciação sazonal.
            - **Seasonal MA Order (Q)**: Número de termos de média móvel para a sazonalidade.
            - **Seasonal Periodicity (s)**: Define o período de sazonalidade (12 para dados mensais).

            ### 📊 Resultados do Relatório

            - **Decomposição da Série Temporal**: Gráfico que mostra tendência, sazonalidade e resíduo da série.
            - **Previsão da Série Temporal**: Gráfico da série histórica e a previsão para o futuro.
            - **Tabela de Dados da Previsão**: Valores previstos em tabela, com opção de download.

            ### 💾 Baixar os Dados
            Clique em **"Baixar Previsão"** para salvar os dados de previsão em CSV.

            ### 📘 Dicas para Uso Ideal

            - **Dados de Qualidade**: Use dados mensais consistentes.
            - **Ajuste dos Parâmetros**: Experimente diferentes combinações para obter a melhor previsão.
            - **Interpretação**: Utilize a decomposição para identificar fatores sazonais e tendências.
        """, unsafe_allow_html=True)

# Barra lateral com upload de arquivo e parâmetros
with st.sidebar:
    st.header("Configurações")
    uploaded_file = st.file_uploader("Escolha o arquivo CSV:", type=['csv'])
    if uploaded_file is not None:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        data = pd.read_csv(stringio, header=None)
        
        # Configurações do período
        data_inicio = date(2000, 1, 1)
        periodo = st.date_input("Período Inicial da Série", data_inicio)
        periodo_previsao = st.number_input("Meses para Previsão", min_value=1, max_value=48, value=12)
        
        # Configurações do modelo SARIMAX
        st.subheader("Configurações do Modelo SARIMAX")
        p = st.number_input("AR Order (p)", min_value=0, max_value=5, value=2)
        d = st.number_input("I Order (d)", min_value=0, max_value=2, value=0)
        q = st.number_input("MA Order (q)", min_value=0, max_value=5, value=0)
        P = st.number_input("Seasonal AR Order (P)", min_value=0, max_value=5, value=0)
        D = st.number_input("Seasonal I Order (D)", min_value=0, max_value=2, value=1)
        Q = st.number_input("Seasonal MA Order (Q)", min_value=0, max_value=5, value=1)
        s = st.number_input("Seasonal Periodicity (S)", min_value=1, max_value=12, value=12)
        
        processar = st.button("🔍 Processar Análise e Previsão")

# Processamento dos dados
if uploaded_file is not None and processar:
    try:
        # Configuração da série temporal
        ts_data = pd.Series(data.iloc[:, 0].values, index=pd.date_range(start=periodo, periods=len(data), freq='M'))
        
        # Decomposição da série
        decomposicao = seasonal_decompose(ts_data, model='additive')
        fig_decomposicao = decomposicao.plot()
        fig_decomposicao.set_size_inches(5, 4)

        # Configuração e ajuste do modelo SARIMAX
        modelo = SARIMAX(ts_data, order=(p, d, q), seasonal_order=(P, D, Q, s))
        modelo_fit = modelo.fit(disp=False)
        previsao = modelo_fit.forecast(steps=periodo_previsao)

        # Gráfico de previsão
        fig_previsao, ax = plt.subplots(figsize=(5, 4))
        ax = ts_data.plot(ax=ax, label='Série Histórica')
        previsao.plot(ax=ax, style='r--', label='Previsão')
        plt.legend()

        # Exibir gráficos em containers
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Decomposição da Série Temporal")
            st.pyplot(fig_decomposicao)
        
        with col2:
            st.write("### Previsão da Série Temporal")
            st.pyplot(fig_previsao)

        st.write("### Dados da Previsão")
        st.dataframe(previsao, height=300)
        csv_previsao = previsao.to_csv(index=True)
        st.download_button(label="📥 Baixar Previsão", data=csv_previsao, 
                           file_name="previsao.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Erro ao processar os dados: {e}")

# Rodapé com informações do desenvolvedor
st.write("---")
st.markdown(
    """
    <div style='text-align: center; margin-top: 20px; line-height: 1.2;'>
        <p style='font-size: 16px; font-weight: bold; margin: 0;'>Projeto: Sistema de Análise e Previsão de Séries Temporais</p>
        <p style='font-size: 14px; margin: 5px 0;'>Desenvolvido por:</p>
        <p style='font-size: 20px; color: #4CAF50; font-weight: bold; margin: 0;'>Cláudio Ferreira Neves</p>
        <p style='font-size: 16px; color: #555; margin: 0;'>Especialista em Análise de Dados, RPA e AI</p>
        <p style='font-size: 14px; margin: 10px 0 5px 0;'>Ferramentas utilizadas: Python, Streamlit, Pandas, StatsModels, Matplotlib</p>
        <p style='font-size: 12px; color: #777; margin: 0;'>© 2024</p>
    </div>
    """,
    unsafe_allow_html=True
)
