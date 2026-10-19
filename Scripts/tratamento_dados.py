import pandas as pd
import glob
import os

pasta = r"Data/Criminais"
arquivos = glob.glob(os.path.join(pasta, "*.parquet"))

# 1. Mapa de renomeação simples (mesmo campo, nome diferente)
renomear = {
    "COD IBGE": "CD_IBGE",
    "NOME_DELEGACIA_CIRCUNSCRICAO": "NOME_DELEGACIA_CIRCUNCRIÇÃO",
    "NOME_DEPARTAMENTO_CIRCUNSCRICAO": "NOME_DEPARTAMENTO_CIRCUNCRIÇÃO",
    "NOME_MUNICIPIO_CIRCUNSCRICAO": "NOME_MUNICIPIO_CIRCUNCRIÇÃO",
    "NOME_SECCIONAL_CIRCUNSCRICAO": "NOME_SECCIONAL_CIRCUNCRIÇÃO",
}

# 2. Mapa de "migração": coluna antiga (só 2022) -> coluna nova (2023+)
migrar = {
    "CIDADE": "NOME_MUNICIPIO",
    "DATA_COMUNICACAO_BO": "DATA_REGISTRO",
    "DESCR_PERIODO": "DESC_PERIODO",
}

# 3. Colunas que precisam forçar tipo
colunas_data = ["DATA_OCORRENCIA_BO", "DATA_REGISTRO"]
colunas_hora = ["HORA_OCORRENCIA_BO"]
colunas_numericas = ["LATITUDE", "LONGITUDE", "NUMERO_LOGRADOURO"]

dfs = []
for arquivo in arquivos:
    df = pd.read_parquet(arquivo)
    df = df.rename(columns=renomear)

    # Migração: joga o valor da coluna antiga pra dentro da nova
    for col_antiga, col_nova in migrar.items():
        if col_antiga in df.columns:
            if col_nova in df.columns:
                # não deveria acontecer (antiga e nova juntas no mesmo arquivo), mas por segurança:
                df[col_nova] = df[col_nova].fillna(df[col_antiga])
            else:
                df[col_nova] = df[col_antiga]
            df = df.drop(columns=[col_antiga])

    df["ARQUIVO_ORIGEM"] = os.path.basename(arquivo)

    for col in colunas_data:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in colunas_hora:
        if col in df.columns:
            df[col] = df[col].astype(str)

    for col in colunas_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    dfs.append(df)

df_consolidado = pd.concat(dfs, ignore_index=True)

print(f"Total de linhas: {len(df_consolidado)}")
print(f"Total de colunas: {len(df_consolidado.columns)}")
print(df_consolidado.dtypes)

df_consolidado.to_parquet(os.path.join(pasta, "SPDadosCriminais_CONSOLIDADO.parquet"), index=False)