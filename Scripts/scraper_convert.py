import sys
import re
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests
import pandas as pd

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def obter_url() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1].strip()
    return input("Cole o link do arquivo (.xlsx ou .csv): ").strip()


def detectar_extensao(url: str, content_type: str) -> str:
    """Tenta descobrir a extensão pelo caminho da URL; se falhar, usa o Content-Type."""
    caminho = urlparse(url).path.lower()
    if caminho.endswith(".xlsx") or caminho.endswith(".xls"):
        return "xlsx"
    if caminho.endswith(".csv"):
        return "csv"

    content_type = (content_type or "").lower()
    if "spreadsheet" in content_type or "excel" in content_type:
        return "xlsx"
    if "csv" in content_type or "text/plain" in content_type:
        return "csv"

    raise ValueError(
        "Não foi possível identificar se o arquivo é .xlsx ou .csv. "
        "Verifique o link ou informe a extensão manualmente."
    )


def baixar_arquivo(url: str) -> tuple[Path, str]:
    print(f"\nBaixando: {url}")
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()

    extensao = detectar_extensao(url, resp.headers.get("Content-Type", ""))

    # Nome do arquivo baseado na URL, caindo para um nome genérico se necessário
    nome_original = Path(urlparse(url).path).stem or "dados_baixados"
    nome_original = re.sub(r"[^\w\-]", "_", nome_original)

    caminho_temp = Path(tempfile.gettempdir()) / f"{nome_original}.{extensao}"
    caminho_temp.write_bytes(resp.content)

    print(f"Arquivo baixado ({extensao.upper()}): {caminho_temp} ({len(resp.content)} bytes)")
    return caminho_temp, extensao


def ler_dados(caminho: Path, extensao: str) -> dict[str, pd.DataFrame]:
    """Retorna um dicionário {nome_aba: DataFrame}. Para CSV, a 'aba' é o próprio nome do arquivo."""
    if extensao == "csv":
        # tenta detectar separador automaticamente (vírgula, ponto-e-vírgula, tab)
        df = pd.read_csv(caminho, sep=None, engine="python")
        return {caminho.stem: df}

    # xlsx: lê todas as abas
    return pd.read_excel(caminho, sheet_name=None, engine="openpyxl")


def escolher_pasta_destino() -> Path:
    while True:
        entrada = input(
            "\nEm qual pasta você quer salvar o(s) .parquet? "
            "(caminho relativo ou absoluto): "
        ).strip()
        if not entrada:
            print("Digite um caminho válido.")
            continue

        pasta = Path(entrada).expanduser()
        if not pasta.exists():
            criar = input(f"A pasta '{pasta}' não existe. Criar agora? (s/n): ").strip().lower()
            if criar == "s":
                pasta.mkdir(parents=True, exist_ok=True)
            else:
                continue
        return pasta


def sanitizar_colunas_mistas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrige colunas com tipos misturados (ex.: maioria números, mas com
    algum texto solto no meio, tipo "REGISTRADO NA PF"). O pandas guarda
    isso como dtype 'object', mas o PyArrow tenta inferir um tipo único
    (int64, float64...) e quebra ao encontrar o texto. Aqui, convertemos
    essas colunas para string, preservando os valores nulos (NaN/None).
    """
    for coluna in df.columns:
        if df[coluna].dtype == "object":
            valores_nao_nulos = df[coluna].dropna()
            tipos_encontrados = valores_nao_nulos.map(type).unique()
            if len(tipos_encontrados) > 1:
                df[coluna] = df[coluna].apply(
                    lambda x: x if pd.isna(x) else str(x)
                )
    return df


def converter_e_salvar(planilhas: dict[str, pd.DataFrame], nome_base: str, pasta_destino: Path):
    if len(planilhas) == 1:
        (nome_aba, df), = planilhas.items()
        df = sanitizar_colunas_mistas(df)
        caminho_saida = pasta_destino / f"{nome_base}.parquet"
        df.to_parquet(caminho_saida, index=False)
        print(f"Convertido: {caminho_saida} ({len(df)} linhas)")
    else:
        for nome_aba, df in planilhas.items():
            df = sanitizar_colunas_mistas(df)
            nome_aba_limpo = re.sub(r"[^\w\-]", "_", nome_aba)
            caminho_saida = pasta_destino / f"{nome_base}_{nome_aba_limpo}.parquet"
            df.to_parquet(caminho_saida, index=False)
            print(f"Aba '{nome_aba}' convertida: {caminho_saida} ({len(df)} linhas)")


def main():
    url = obter_url()
    caminho_baixado, extensao = baixar_arquivo(url)

    print("Lendo os dados...")
    planilhas = ler_dados(caminho_baixado, extensao)

    pasta_destino = escolher_pasta_destino()

    nome_base = caminho_baixado.stem
    converter_e_salvar(planilhas, nome_base, pasta_destino)

    print("\nConcluído!")


if __name__ == "__main__":
    main()