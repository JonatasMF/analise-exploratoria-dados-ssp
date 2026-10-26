import streamlit as st

st.set_page_config(
    page_title="Painel de dados da Segurança Pública - SP",
    page_icon="🚔",
    layout="wide",
)

st.title("Painel de Dados da Segurança Pública - SP")
st.caption("Análise exploratória de indicadores de criminalidade e violência no estado de São Paulo")

st.markdown(
    """
    Este painel reúne análises de dados públicos sobre segurança pública no Estado de São Paulo,
    com foco em ocorrências criminais e mortes decorrentes da ação policial. A ideia é apresentar
    informações de forma clara, visual e acessível para apoiar a leitura de tendências e padrões.
    """
)

st.subheader("📌 Sobre o projeto")

st.markdown(
    """
    O projeto foi desenvolvido como parte da disciplina de Projeto Integrador e tem como objetivo
    explorar os dados disponíveis para entender melhor a distribuição e a evolução dos indicadores
    de segurança no estado. A análise combina visualizações, filtros interativos e contexto
    temporal para facilitar a interpretação das informações.
    """
)

st.markdown(
    """
    ### Conteúdo disponível
    - Análise de ocorrências criminais
    - Comparações por município e período
    - Visualizações de tendência e distribuição geográfica
    - Indicadores relacionados a mortes decorrentes da ação policial
    """
)

st.markdown("---")

st.info("Use o menu lateral para navegar entre as páginas do painel e explorar os indicadores disponíveis.")