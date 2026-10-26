import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

CAMINHO_PARQUET = "Data/Criminais/SPDadosCriminais_CONSOLIDADO_reduzido.parquet"

# Só as colunas realmente usadas no dashboard.
# Adicione outras aqui se for usar em mais gráficos futuramente.
COLUNAS_NECESSARIAS = [
    "DATA_OCORRENCIA_BO",
    "NATUREZA_APURADA",
    "NOME_MUNICIPIO_CIRCUNCRIÇÃO",
    "NUM_BO",
    "DESC_PERIODO",
    "LATITUDE",
    "LONGITUDE",
]

DATA_INICIO = pd.Timestamp("2022-01-01")
DATA_FIM = pd.Timestamp("2026-06-30")


@st.cache_data
def carregar_dados():
    df = pd.read_parquet(
        CAMINHO_PARQUET,
        columns=COLUNAS_NECESSARIAS,
        engine="pyarrow",
        filters=[
            ("DATA_OCORRENCIA_BO", ">=", DATA_INICIO),
            ("DATA_OCORRENCIA_BO", "<=", DATA_FIM),
        ],
    )
    df["DATA_OCORRENCIA_BO"] = pd.to_datetime(df["DATA_OCORRENCIA_BO"])

    for col in ["NATUREZA_APURADA", "NOME_MUNICIPIO_CIRCUNCRIÇÃO"]:
        df[col] = df[col].astype("category")

    df["ANO_MES"] = df["DATA_OCORRENCIA_BO"].dt.to_period("M").dt.to_timestamp()
    df["ANO"] = df["DATA_OCORRENCIA_BO"].dt.year.astype("int16")
    df["MES"] = df["DATA_OCORRENCIA_BO"].dt.month.astype("int8")

    return df


df = carregar_dados()

st.title("📊 Análise Exploratória de Dados Criminais - SP")
st.caption("Dados registrados entre Jan/2022 à Jun/2026")

# ============================================================================
# FILTROS (sidebar) — valem para as duas abas
# ============================================================================
MESES = {
    1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril", 5: "Maio", 6: "Junho",
    7: "Julho", 8: "Agosto", 9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro",
}


def filtrar(dados, cidades=None, naturezas=None, anos=None, meses=None):
    """Aplica só os filtros informados. Lista vazia = sem filtro naquele campo."""
    if cidades:
        dados = dados[dados["NOME_MUNICIPIO_CIRCUNCRIÇÃO"].isin(cidades)]
    if naturezas:
        dados = dados[dados["NATUREZA_APURADA"].isin(naturezas)]
    if anos:
        dados = dados[dados["ANO"].isin(anos)]
    if meses:
        dados = dados[dados["MES"].isin(meses)]
    return dados


with st.sidebar:
    st.title("🔎 Filtros")

    lista_cidades = sorted(df["NOME_MUNICIPIO_CIRCUNCRIÇÃO"].dropna().unique())
    sel_cidades = st.multiselect("Cidade", lista_cidades, placeholder="Todas")

    lista_naturezas = sorted(df["NATUREZA_APURADA"].dropna().unique())
    sel_naturezas = st.multiselect("Tipo de ocorrência", lista_naturezas, placeholder="Todos")

    lista_anos = sorted(int(a) for a in df["ANO"].unique())
    sel_anos = st.multiselect("Ano", lista_anos, default=lista_anos)

    sel_meses = st.multiselect("Mês", list(MESES), default=list(MESES), format_func=MESES.get)

# df_f: todos os filtros (aba de análise). df_ml: só cidade e tipo (a previsão precisa da série mensal contínua).
df_f = filtrar(df, sel_cidades, sel_naturezas, sel_anos, sel_meses)
df_ml = filtrar(df, sel_cidades, sel_naturezas)

with st.sidebar:
    st.caption(f"{len(df_f):,} de {len(df):,} ocorrências".replace(",", "."))

aba_eda, aba_ml = st.tabs(["📊 Análise Exploratória de Dados", "🤖 Modelo Preditivo"])

# ============================================================================
# ABA 1 — ANÁLISE EXPLORATÓRIA DE DADOS (código original)
# ============================================================================
def render_eda(df):
    if df.empty:
        st.warning("Nenhuma ocorrência com os filtros selecionados.")
        return

    st.subheader("🔢Números Gerais")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total de Ocorrências", f"{len(df):,}".replace(",", "."))
    col2.metric("Natureza Mais Frequente", df["NATUREZA_APURADA"].value_counts().idxmax())
    col3.metric("Munícipio com mais Registros", df["NOME_MUNICIPIO_CIRCUNCRIÇÃO"].value_counts().idxmax())

    st.subheader("📈 Evolução de Ocorrências por Mês")

    ocorrencias_p_data = (
        df.groupby("ANO_MES")["NUM_BO"]
        .nunique()
        .reset_index(name="total_ocs")
    )

    fig1 = px.line(
        ocorrencias_p_data,
        x="ANO_MES",
        y="total_ocs",
        markers=True,
        labels={"ANO_MES": "Ano/Mês", "total_ocs": "Ocorrências de Crime Registradas"},
        title="Ocorrências por Ano/Mês",
    )

    st.plotly_chart(fig1)

    col4, col5 = st.columns(2)

    df_homic = df[df["NATUREZA_APURADA"].str.contains("HOMIC", na=False)]

    top_10_homic = (
        df_homic.groupby("NOME_MUNICIPIO_CIRCUNCRIÇÃO", observed=True)
        .size()
        .reset_index(name="Quantidade de Homicídios")
        .sort_values("Quantidade de Homicídios", ascending=False)
        .head(10)
    )

    fig2 = px.funnel(
        top_10_homic,
        x="Quantidade de Homicídios",
        y="NOME_MUNICIPIO_CIRCUNCRIÇÃO",
        labels={"NOME_MUNICIPIO_CIRCUNCRIÇÃO": "Municípios"},
        title="Top 10 Municípios com mais Homicídios",
    )
    col4.plotly_chart(fig2)

    periodo_limpo = df["DESC_PERIODO"].astype(str).str.strip().str.upper()

    periodo = (
        df.groupby(periodo_limpo)["NUM_BO"]
        .nunique()
        .reset_index(name="total_ocs")
    )

    fig3 = px.bar(
        periodo,
        x="DESC_PERIODO",
        y="total_ocs",
        labels={"DESC_PERIODO": "Períodos do dia", "total_ocs": "Total de Ocorrências"},
        title="Ocorrências Por Período",
    )
    col5.plotly_chart(fig3)

    map_df = df.dropna(subset=["LATITUDE", "LONGITUDE"]).copy()

    # remove coordenadas inválidas (0,0)
    map_df = map_df[(map_df["LATITUDE"] != 0) & (map_df["LONGITUDE"] != 0)]

    # garante só pontos plausíveis no Brasil
    map_df = map_df[
        map_df["LATITUDE"].between(-34, 5) &
        map_df["LONGITUDE"].between(-74, -34)
    ]

    map_df = map_df.sample(min(len(map_df), 5000), random_state=42)

    map1 = px.density_map(
        map_df,
        lat="LATITUDE",
        lon="LONGITUDE",
        radius=8,
        zoom=8,
        center={"lat": map_df["LATITUDE"].mean(), "lon": map_df["LONGITUDE"].mean()},
        height=650,
        color_continuous_scale="YlOrRd",
        title="Densidade Geográfica das Ocorrências",
    )

    map1.update_layout(
        map_style="open-street-map",
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
    )

    st.plotly_chart(map1, use_container_width=True)

with aba_eda:
    render_eda(df_f)

# ============================================================================
# ABA 2 — MODELO PREDITIVO (Machine Learning)
# ============================================================================
with aba_ml:
    st.subheader("🤖 Previsão de Ocorrências Futuras")
    st.caption(
        "Modelo de regressão linear simples: identifica se as ocorrências estão "
        "subindo ou caindo nos meses mais recentes, e projeta essa mesma tendência "
        "para os próximos meses."
    )

    @st.cache_data
    def preparar_serie_mensal(df: pd.DataFrame) -> pd.DataFrame:
        """Agrega o total de ocorrências por mês, em ordem cronológica."""
        serie = (
            df.groupby("ANO_MES")["NUM_BO"]
            .nunique()
            .reset_index(name="total_ocs")
            .sort_values("ANO_MES")
            .reset_index(drop=True)
        )
        # número sequencial do mês (0, 1, 2, 3...) — é só "a ordem no tempo"
        serie["indice_tempo"] = np.arange(len(serie))
        return serie

    serie = preparar_serie_mensal(df_ml)
    st.caption(
        "Os filtros de **cidade** e **tipo de ocorrência** valem aqui. Ano e mês são ignorados: "
        "a previsão precisa da série mensal completa e contínua."
    )

    if len(serie) < 6:
        st.warning("Poucos meses de dados disponíveis para estimar uma tendência confiável.")
    else:
        col_opcao1, col_opcao2 = st.columns(2)

        with col_opcao1:
            ignorar_ultimo_mes = st.checkbox(
                "Ignorar o último mês da base ao calcular a tendência",
                value=True,
                help=(
                    "Dados de ocorrências costumam chegar incompletos no mês mais "
                    "recente, porque os B.O.s ainda estão sendo processados/consolidados "
                    "no sistema no momento da extração. Isso pode fazer o último mês "
                    "aparecer artificialmente baixo, mesmo sem ter havido queda real."
                ),
            )

        serie_para_treino = serie.iloc[:-1] if ignorar_ultimo_mes else serie

        with col_opcao2:
            max_janela = len(serie_para_treino)
            janela_padrao = min(12, max_janela)
            meses_para_tendencia = st.slider(
                "Usar quantos meses recentes para calcular a tendência?",
                min_value=min(3, max_janela), max_value=max_janela, value=janela_padrao,
                help=(
                    "Usar só os meses mais recentes (em vez de todo o histórico) faz o "
                    "modelo reagir ao comportamento atual, e não a uma média dos últimos "
                    "anos inteiros."
                ),
            )

        serie_treino = serie_para_treino.tail(meses_para_tendencia).reset_index(drop=True)
        # Reindexa o tempo para começar em 0 dentro da janela usada no treino —
        # o que importa para a reta é a posição relativa dentro da janela.
        serie_treino["indice_janela"] = np.arange(len(serie_treino))

        # Regressão linear simples: encontra a "linha reta" que melhor
        # acompanha a evolução mensal das ocorrências na janela escolhida.
        # É o modelo mais simples de machine learning para tendência — fácil
        # de explicar: ele olha se o número de ocorrências está subindo ou
        # descendo mês a mês (na janela recente) e projeta essa mesma
        # inclinação para frente.
        modelo = LinearRegression()
        modelo.fit(serie_treino[["indice_janela"]], serie_treino["total_ocs"])

        n_meses_futuro = st.slider("Prever quantos meses à frente?", min_value=1, max_value=12, value=6)

        ultima_data_treino = serie_treino["ANO_MES"].max()
        datas_futuras = pd.date_range(
            ultima_data_treino + pd.offsets.MonthBegin(1), periods=n_meses_futuro, freq="MS"
        )
        indices_futuros = np.arange(len(serie_treino), len(serie_treino) + n_meses_futuro)

        futuro = pd.DataFrame({
            "ANO_MES": datas_futuras,
            "previsao": modelo.predict(indices_futuros.reshape(-1, 1)),
        })

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=serie["ANO_MES"], y=serie["total_ocs"],
            mode="lines+markers", name="Histórico (real)",
        ))
        fig.add_trace(go.Scatter(
            x=serie_treino["ANO_MES"], y=serie_treino["total_ocs"],
            mode="markers", name="Meses usados na tendência",
            marker=dict(size=10, color="rgba(0,0,0,0)", line=dict(width=2, color="green")),
        ))
        fig.add_trace(go.Scatter(
            x=futuro["ANO_MES"], y=futuro["previsao"],
            mode="lines+markers", name="Previsão (tendência)",
            line=dict(dash="dash", color="orange"),
        ))
        fig.update_layout(
            title="Ocorrências ao Longo do Tempo — Histórico e Previsão",
            xaxis_title="Ano/Mês", yaxis_title="Ocorrências",
        )
        st.plotly_chart(fig, use_container_width=True)

        variacao_mensal = modelo.coef_[0]
        tendencia = "de crescimento 📈" if variacao_mensal > 0 else "de queda 📉"
        st.info(
            f"Com base nos últimos **{meses_para_tendencia} meses** considerados, o modelo "
            f"identifica uma tendência **{tendencia}**: em média, o número de ocorrências "
            f"muda cerca de **{abs(variacao_mensal):,.0f} por mês**."
            .replace(",", ".")
        )

        st.markdown("##### Ocorrências previstas por mês")
        st.dataframe(
            futuro.rename(columns={"ANO_MES": "Mês", "previsao": "Ocorrências Previstas"})
            .assign(**{"Ocorrências Previstas": lambda d: d["Ocorrências Previstas"].round(0).astype(int)}),
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "⚠️ Este é um modelo simples de tendência: ele assume que o padrão "
            "observado na janela recente escolhida continua igual nos próximos "
            "meses. Ele não considera fatores externos (novas políticas de segurança, "
            "eventos atípicos, etc.)."
        )