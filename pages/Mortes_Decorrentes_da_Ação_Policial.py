import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
 
st.set_page_config(page_title="Letalidade Policial - Pós COP", page_icon="🚔", layout="wide")
 
# ----------------------------------------------------------------------------
# Configurações
# ----------------------------------------------------------------------------
CAMINHO = "Data/MDIP/MDIP_2026_MDIP_2013_A_2026.parquet"
INICIO_COP = pd.Timestamp("2020-07-01")
DIAS = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
 
AZUL = "#2a78d6"      # série principal / "antes das COPs"
LARANJA = "#eb6834"   # "após as COPs"
VERDE = "#008300"     # marcador das COPs
CORES_REGIME = {"Antes das COPs": AZUL, "Ano de transição (2020)": "#9aa0a6", "Após as COPs": LARANJA}
 
 
# ----------------------------------------------------------------------------
# Dados
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Carregando a base...")
def carregar_dados(caminho: str) -> pd.DataFrame:
    df = pd.read_parquet(caminho)
    df = df[df["COORPORAÇÃO"] == "PM"].copy()
 
    # competência (mês/ano estatístico)
    ano = pd.to_numeric(df["ANO ESTATISTICA"], errors="coerce")
    mes = pd.to_numeric(df["MÊS ESTATISTICA"], errors="coerce")
    df["MES_ANO_EST"] = pd.to_datetime({"year": ano, "month": mes, "day": 1}, errors="coerce")
    df = df.dropna(subset=["MES_ANO_EST"])
    df["ANO"] = df["MES_ANO_EST"].dt.year
 
    # dia da semana e hora do fato
    data_fato = pd.to_datetime(df["DATA_FATO"], errors="coerce")
    df["DIA_SEMANA"] = data_fato.dt.dayofweek.map(dict(enumerate(DIAS)))
    hora = df["HORA_FATO"].astype(str).str.extract(r"^(\d{1,2})")[0]
    df["HORA"] = pd.to_numeric(hora, errors="coerce")
    df.loc[~df["HORA"].between(0, 23), "HORA"] = np.nan
 
    # faixa etária
    idade = pd.to_numeric(df["IDADE_PESSOA"], errors="coerce")
    idade = idade.where(idade.between(0, 110))
    df["FAIXA_ETARIA"] = pd.cut(
        idade,
        bins=[-1, 11, 17, 24, 34, 44, 59, 200],
        labels=["0-11", "12-17", "18-24", "25-34", "35-44", "45-59", "60+"],
    ).astype("object")
 
    # coordenadas (aceita vírgula decimal e descarta o que está fora do estado de SP)
    for col in ("LATITUDE", "LONGITUDE"):
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    fora = ~df["LATITUDE"].between(-25.4, -19.7) | ~df["LONGITUDE"].between(-53.2, -44.1)
    df.loc[fora, ["LATITUDE", "LONGITUDE"]] = np.nan
    return df
 
 
def serie_mensal(d: pd.DataFrame) -> pd.Series:
    """Contagem de registros por mês, incluindo meses sem nenhum registro (zero)."""
    s = d.groupby("MES_ANO_EST").size()
    meses = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    return s.reindex(meses, fill_value=0).astype(float)
 
 
def regime(ano: int) -> str:
    if ano < 2020:
        return "Antes das COPs"
    if ano == 2020:
        return "Ano de transição (2020)"
    return "Após as COPs"
 
 
def barra_contagem(d: pd.DataFrame, col: str, titulo: str, ordem=None, horizontal=False, top=None):
    c = d[col].fillna("Não informado").astype(str).value_counts()
    if ordem:
        c = c.reindex([o for o in ordem if o in c.index] + [i for i in c.index if i not in ordem], fill_value=0)
    if top:
        c = c.head(top)
    c = c.rename_axis(col).reset_index(name="Registros")
    if horizontal:
        fig = px.bar(c.iloc[::-1], x="Registros", y=col, orientation="h", title=titulo)
    else:
        fig = px.bar(c, x=col, y="Registros", title=titulo)
    fig.update_traces(marker_color=AZUL)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), xaxis_title=None, yaxis_title=None)
    return fig
 
 
# ----------------------------------------------------------------------------
# Cabeçalho + carga
# ----------------------------------------------------------------------------
st.title("🚔 Letalidade Policial - Pós COP")
st.caption("Dados coletados a partir de jan/2013 até jun/2026")
 
base = carregar_dados(CAMINHO)
 
# ----------------------------------------------------------------------------
# Filtros
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("🔎 Filtros")
 
    ano_min, ano_max = int(base["ANO"].min()), int(base["ANO"].max())
    anos = st.slider("Período (ano estatístico)", ano_min, ano_max, (ano_min, ano_max))
 
    tipos = sorted(base["TIPO_PESSOA"].dropna().unique())
    sel_tipo = st.multiselect("Tipo de pessoa", tipos, default=tipos)
 
    municipios = sorted(base["MUNICIPIO_CIRCUNSCRICAO"].dropna().unique())
    sel_mun = st.multiselect("Município (vazio = todos)", municipios)
 
df = base[base["ANO"].between(*anos)]
if sel_tipo:
    df = df[df["TIPO_PESSOA"].isin(sel_tipo)]
if sel_mun:
    df = df[df["MUNICIPIO_CIRCUNSCRICAO"].isin(sel_mun)]
 
if df.empty:
    st.warning("Nenhum registro com os filtros selecionados.")
    st.stop()
 
serie = serie_mensal(df)
 
aba1, aba2 = st.tabs(["📊 Análise Exploratória de Dados", "🗺️ Focos geográficos (DBSCAN)"])
 
# ----------------------------------------------------------------------------
# ABA 1 - Análise exploratória
# ----------------------------------------------------------------------------
with aba1:
    pre = serie[serie.index < INICIO_COP]
    pos = serie[serie.index >= INICIO_COP]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros no filtro", f"{int(serie.sum()):,}".replace(",", "."))
    c2.metric("Média mensal antes das COPs", f"{pre.mean():.1f}" if len(pre) else "-")
    c3.metric("Média mensal após as COPs", f"{pos.mean():.1f}" if len(pos) else "-")
    if len(pre) and len(pos) and pre.mean() > 0:
        c4.metric("Variação da média", f"{(pos.mean() / pre.mean() - 1) * 100:+.1f}%")
    else:
        c4.metric("Variação da média", "-")
    st.caption("Comparação descritiva (antes x depois de 01/07/2020). Não prova que as COPs causaram a variação: "
               "outros fatores mudaram no período.")
 
    # 1) série mensal + média móvel
    st.subheader("Ocorrências por Mês e Ano")
    dfm = serie.rename("Ocorrências").rename_axis("MES_ANO_EST").reset_index()
    dfm["Média móvel 12m"] = dfm["Ocorrências"].rolling(12, min_periods=12).mean()
    fig1 = px.line(dfm, x="MES_ANO_EST", y="Ocorrências", markers=True, labels={"MES_ANO_EST": "Mês/Ano"})
    fig1.update_traces(line_color=AZUL, name="Mensal", showlegend=True)
    fig1.add_scatter(x=dfm["MES_ANO_EST"], y=dfm["Média móvel 12m"], mode="lines", name="Média móvel 12m",
                     line=dict(color=LARANJA, width=2.5))
    fig1.add_vline(x=INICIO_COP, line_width=1.92, line_dash="dash", line_color=VERDE)
    fig1.add_annotation(x=INICIO_COP, y=0.98, yref="paper", text="INÍCIO OFICIAL DAS COPs",
                        showarrow=False, yanchor="bottom", font=dict(color=VERDE))
    fig1.update_layout(margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig1, width="stretch")
 
    # 2) média mensal por ano
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Média mensal por ano")
        por_ano = serie.groupby(serie.index.year).agg(["mean", "count"]).reset_index()
        por_ano.columns = ["Ano", "Média mensal", "Meses na base"]
        por_ano["Regime"] = por_ano["Ano"].map(regime)
        fig2 = px.bar(por_ano, x="Ano", y="Média mensal", color="Regime",
                      color_discrete_map=CORES_REGIME, hover_data=["Meses na base"])
        fig2.update_xaxes(type="category")
        fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25, title=None))
        st.plotly_chart(fig2, width="stretch")
        st.caption("Usa média mensal porque 2026 só tem 6 meses na base.")
 
    # 3) distribuição por hora: antes x depois
    with col_b:
        st.subheader("Hora do fato: antes x após as COPs")
        h = df.dropna(subset=["HORA"]).copy()
        h["Regime"] = np.where(h["MES_ANO_EST"] < INICIO_COP, "Antes das COPs", "Após as COPs")
        dist = (h.groupby(["Regime", "HORA"]).size().groupby(level=0).transform(lambda x: x / x.sum() * 100)
                .rename("% dos registros").reset_index())
        fig3 = px.line(dist, x="HORA", y="% dos registros", color="Regime", markers=True,
                       color_discrete_map=CORES_REGIME, labels={"HORA": "Hora do dia"})
        fig3.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25, title=None))
        st.plotly_chart(fig3, width="stretch")
        st.caption("Cada linha soma 100%, então dá para comparar o perfil mesmo com volumes diferentes.")
 
    # 4) municípios
    st.subheader("Municípios com mais registros")
    st.plotly_chart(barra_contagem(df, "MUNICIPIO_CIRCUNSCRICAO", "", horizontal=True, top=15), width="stretch")
 
    # 5) heatmap dia x hora
    st.subheader("Dia da semana x hora")
    piv = (df.dropna(subset=["HORA", "DIA_SEMANA"]).groupby(["DIA_SEMANA", "HORA"]).size()
           .unstack(fill_value=0).reindex(index=DIAS, columns=range(24), fill_value=0))
    fig5 = px.imshow(piv, aspect="auto", color_continuous_scale="Blues",
                     labels=dict(x="Hora do fato", y="Dia da semana", color="Registros"))
    fig5.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig5, width="stretch")
 
    # 6) perfil das vítimas/pessoas
    st.subheader("Perfil das pessoas nos registros")
    p1, p2, p3 = st.columns(3)
    p1.plotly_chart(barra_contagem(df, "SEXO_PESSOA", "Sexo"), width="stretch")
    p2.plotly_chart(barra_contagem(df, "FAIXA_ETARIA", "Faixa etária",
                                   ordem=["0-11", "12-17", "18-24", "25-34", "35-44", "45-59", "60+"]),
                    width="stretch")
    p3.plotly_chart(barra_contagem(df, "COR_PELE", "Cor da pele (autodeclarada/registrada)"), width="stretch")
 
    # 7) local e batalhão
    l1, l2 = st.columns(2)
    l1.plotly_chart(barra_contagem(df, "DESC_TIPOLOCAL", "Tipo de local (top 10)", horizontal=True, top=10),
                    width="stretch")
    l2.plotly_chart(barra_contagem(df, "LABEL_BTL", "Batalhões (top 15)", horizontal=True, top=15),
                    width="stretch")
 
    # 8) mapa
    st.subheader("Mapa de densidade")
    geo = df.dropna(subset=["LATITUDE", "LONGITUDE"])
    if geo.empty:
        st.info("Nenhum registro com coordenadas válidas no filtro atual.")
    else:
        if len(geo) > 30000:
            geo = geo.sample(30000, random_state=42)
            st.caption("Amostra de 30.000 pontos para manter o mapa leve.")
        fig8 = px.density_map(geo, lat="LATITUDE", lon="LONGITUDE", radius=8, zoom=6,
                              center=dict(lat=-22.9, lon=-47.6), map_style="open-street-map")
        fig8.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=520)
        st.plotly_chart(fig8, width="stretch")
 
    with st.expander("Ver dados (primeiras 5.000 linhas)"):
        st.dataframe(df.head(5000), width="stretch")
 
# ----------------------------------------------------------------------------
# ABA 2 - Focos geográficos (DBSCAN)
# ----------------------------------------------------------------------------
EARTH_KM = 6371.0088
CORES_FOCO = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
COR_DEMAIS = "#6250d6"
COR_ISOLADOS = "#c7c7c7"
TOP_FOCOS_COR = 6
 
 
@st.cache_data(show_spinner="Agrupando os pontos (DBSCAN)...")
def rodar_dbscan(lat: np.ndarray, lon: np.ndarray, eps_km: float, min_samples: int) -> np.ndarray:
    coords = np.radians(np.column_stack([lat, lon]))
    modelo = DBSCAN(eps=eps_km / EARTH_KM, min_samples=min_samples, metric="haversine", algorithm="ball_tree")
    return modelo.fit_predict(coords)
 
 
@st.cache_data(show_spinner=False)
def distancias_k(lat: np.ndarray, lon: np.ndarray, k: int) -> np.ndarray:
    """Distância (km) de cada ponto ao seu k-ésimo vizinho, ordenada. O ponto conta como o 1º vizinho, como no DBSCAN."""
    coords = np.radians(np.column_stack([lat, lon]))
    nn = NearestNeighbors(n_neighbors=k, metric="haversine", algorithm="ball_tree").fit(coords)
    dist, _ = nn.kneighbors(coords)
    return np.sort(dist[:, -1] * EARTH_KM)
 
 
def render_dbscan(df: pd.DataFrame, serie: pd.Series) -> None:
    st.subheader("Focos geográficos dos registros (DBSCAN)")
    st.markdown(
        "O DBSCAN agrupa registros que estão **próximos no mapa** e separa os demais como isolados. "
        "Ele não exige definir o número de grupos nem supõe grupos redondos: "
        "um foco pode ter o formato de uma avenida ou de um bairro. Os parâmetros são o **raio de vizinhança** e o "
        "**mínimo de registros** para formar um foco. Os focos são calculados com o período todo e depois "
        "comparados antes e depois das COPs."
    )
 
    geo = df.dropna(subset=["LATITUDE", "LONGITUDE"]).reset_index(drop=True)
    if len(geo) < 50:
        st.warning("Poucos registros com coordenada válida no filtro atual (menos de 50).")
        return
    st.caption(f"{len(geo):,} registros com coordenada válida ({len(geo) / len(df) * 100:.0f}% do filtro).".replace(",", "."))
 
    c1, c2 = st.columns(2)
    eps_km = c1.slider("Raio de vizinhança (km)", 0.1, 5.0, 1.0, step=0.1)
    min_samples = c2.slider("Mínimo de registros para formar um foco", 3, 100, 15)
 
    with st.expander("Como escolher o raio? (distância ao vizinho mais próximo)"):
        amostra = geo if len(geo) <= 20000 else geo.sample(20000, random_state=42)
        if len(amostra) > min_samples:
            d = distancias_k(amostra["LATITUDE"].to_numpy(), amostra["LONGITUDE"].to_numpy(), min_samples)
            fig_k = px.line(x=np.arange(len(d)), y=d,
                            labels={"x": "Registros, do mais ao menos denso",
                                    "y": f"Distância ao {min_samples}º vizinho (km)"})
            fig_k.update_traces(line_color=AZUL)
            fig_k.add_hline(y=eps_km, line_dash="dash", line_color=LARANJA)
            fig_k.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_k, width="stretch")
            st.caption("A linha laranja é o raio escolhido. O \"cotovelo\" da curva (onde ela passa a subir rápido) "
                       "sugere um bom raio: antes dele os registros têm vizinhos próximos, depois ficam isolados.")
 
    rep = int(geo.groupby(["LATITUDE", "LONGITUDE"]).size().max())
    if rep >= max(min_samples, 0.02 * len(geo)):
        st.info(f"{rep} registros têm exatamente a mesma coordenada. Isso costuma vir de geocodificação aproximada "
                "(por exemplo, o centro de um bairro) e pode criar focos artificiais.")
 
    # ---- agrupamento
    lab = rodar_dbscan(geo["LATITUDE"].to_numpy(), geo["LONGITUDE"].to_numpy(), eps_km, min_samples)
    geo["_lab"] = lab
    tamanhos = geo.loc[geo["_lab"] >= 0, "_lab"].value_counts()
    n_focos = len(tamanhos)
    if n_focos == 0:
        st.warning("Nenhum foco encontrado com esses parâmetros. Aumente o raio ou diminua o mínimo de registros.")
        return
 
    geo["Foco"] = geo["_lab"].map({antigo: i + 1 for i, antigo in enumerate(tamanhos.index)})  # 1 = maior foco
    geo["em_foco"] = geo["Foco"].notna()
    geo["pos_cop"] = geo["MES_ANO_EST"] >= INICIO_COP
 
    antes, depois = geo[~geo["pos_cop"]], geo[geo["pos_cop"]]
    top5 = int((geo["Foco"] <= 5).sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Focos encontrados", n_focos)
    m2.metric("Em focos, antes das COPs", f"{antes['em_foco'].mean() * 100:.0f}%" if len(antes) else "-")
    m3.metric("Em focos, após as COPs", f"{depois['em_foco'].mean() * 100:.0f}%" if len(depois) else "-")
    m4.metric("Os 5 maiores focos reúnem", f"{top5 / len(geo) * 100:.0f}%")
    st.caption("Percentuais sobre os registros com coordenada. O restante são registros isolados.")
 
    # ---- mapa
    def rotulo(n):
        if pd.isna(n):
            return "Isolados"
        return f"Foco {int(n)}" if n <= TOP_FOCOS_COR else "Demais focos"
 
    geo["Categoria"] = geo["Foco"].map(rotulo)
    ordem = ["Isolados"] + [f"Foco {i}" for i in range(1, min(TOP_FOCOS_COR, n_focos) + 1)]
    cores = {"Isolados": COR_ISOLADOS, "Demais focos": COR_DEMAIS,
             **{f"Foco {i + 1}": c for i, c in enumerate(CORES_FOCO)}}
    if n_focos > TOP_FOCOS_COR:
        ordem.insert(1, "Demais focos")
 
    plot = geo if len(geo) <= 40000 else geo.sample(40000, random_state=42)
    fig_m = px.scatter_map(plot, lat="LATITUDE", lon="LONGITUDE", color="Categoria", color_discrete_map=cores,
                           category_orders={"Categoria": ordem}, opacity=0.75, zoom=6,
                           center=dict(lat=float(geo["LATITUDE"].mean()), lon=float(geo["LONGITUDE"].mean())),
                           map_style="open-street-map",
                           hover_data={"MUNICIPIO_CIRCUNSCRICAO": True, "Foco": True,
                                       "LATITUDE": False, "LONGITUDE": False, "Categoria": False})
    fig_m.update_traces(marker=dict(size=7))
    fig_m.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=560, legend=dict(title=None))
    st.plotly_chart(fig_m, width="stretch")
    if len(geo) > 40000:
        st.caption("Mapa com amostra de 40.000 pontos; o agrupamento usa todos.")
 
    # ---- tabela dos focos, antes x depois
    meses_antes = max(int((serie.index < INICIO_COP).sum()), 1)
    meses_depois = max(int((serie.index >= INICIO_COP).sum()), 1)
    em = geo[geo["em_foco"]].copy()
    em["Foco"] = em["Foco"].astype(int)
    g = em.groupby("Foco")
 
    def principal(s: pd.Series) -> str:
        s = s.dropna()
        return str(s.mode().iat[0]) if len(s) else "-"
 
    focos = pd.DataFrame({
        "Registros": g.size(),
        "Antes das COPs": g["pos_cop"].apply(lambda s: int((~s).sum())),
        "Após as COPs": g["pos_cop"].apply(lambda s: int(s.sum())),
        "Município principal": g["MUNICIPIO_CIRCUNSCRICAO"].agg(principal),
        "Latitude": g["LATITUDE"].mean().round(4),
        "Longitude": g["LONGITUDE"].mean().round(4),
    })
    focos["Média/mês antes"] = (focos["Antes das COPs"] / meses_antes).round(2)
    focos["Média/mês após"] = (focos["Após as COPs"] / meses_depois).round(2)
    var = (focos["Média/mês após"] / focos["Média/mês antes"].where(focos["Média/mês antes"] > 0) - 1) * 100
    focos["Variação da média (%)"] = var.where(focos["Antes das COPs"] >= 5).round(1)
 
    st.subheader("Maiores focos: antes x após as COPs")
    top10 = focos.head(10).reset_index()
    nomes = "Foco " + top10["Foco"].astype(str)
    barras = pd.DataFrame({
        "Foco": list(nomes) * 2,
        "Período": ["Antes das COPs"] * len(top10) + ["Após as COPs"] * len(top10),
        "Registros por mês": list(top10["Média/mês antes"]) + list(top10["Média/mês após"]),
    })
    fig_b = px.bar(barras, x="Foco", y="Registros por mês", color="Período", barmode="group",
                   color_discrete_map=CORES_REGIME, category_orders={"Foco": list(nomes)})
    fig_b.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.2, title=None))
    st.plotly_chart(fig_b, width="stretch")
    st.caption("Usa média mensal porque os dois períodos têm durações diferentes. A variação só é calculada com pelo "
               "menos 5 registros antes das COPs; com números pequenos ela oscila muito. É uma comparação descritiva, "
               "não prova efeito das COPs.")
 
    with st.expander("Ver tabela dos focos"):
        st.dataframe(focos.reset_index(), hide_index=True, width="stretch")
    st.caption("Os focos mostram onde os registros se concentraram no passado. Não indicam onde ocorrerão novos casos "
               "nem avaliam a atuação de nenhuma unidade. Mudar o raio ou o mínimo de registros muda os focos.")
 
 
with aba2:
    render_dbscan(df, serie)
 
