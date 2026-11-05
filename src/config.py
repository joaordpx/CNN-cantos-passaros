"""Configurações compartilhadas por todas as etapas do pipeline."""

from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

PASTA_DADOS = RAIZ / "dados"
PASTA_AUDIOS = PASTA_DADOS / "audios"
PASTA_ESPECTROGRAMAS = PASTA_DADOS / "espectrogramas"
ARQUIVO_DIVISAO = PASTA_DADOS / "divisao.csv"
ARQUIVO_CREDITOS = PASTA_DADOS / "creditos_xeno_canto.csv"

PASTA_MODELOS = RAIZ / "modelos"
PASTA_RESULTADOS = RAIZ / "resultados"

# nome da pasta -> (gênero, espécie) no xeno-canto
ESPECIES = {
    "bem-te-vi": ("Pitangus", "sulphuratus"),
    "canario-da-horta": ("Sicalis", "flaveola"),
    "sabia-do-campo": ("Mimus", "saturninus"),
    "trinca-ferro": ("Saltator", "similis"),
}

# áudio
TAXA_AMOSTRAGEM = 22050
DURACAO_TRECHO_S = 5.0
DURACAO_MINIMA_S = 1.0  # trechos finais mais curtos que isso são descartados
LIMIAR_SILENCIO_DBFS = -40.0  # trecho com volume médio abaixo disso é descartado
N_MELS = 128

# divisão e treino
PROPORCOES = {"treino": 0.70, "validacao": 0.15, "teste": 0.15}
TAMANHO_IMAGEM = (128, 128)
SEMENTE = 42
