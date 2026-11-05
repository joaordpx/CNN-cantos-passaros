"""Divide os espectrogramas em treino, validação e teste.

Modo "gravacao" (o correto): todos os trechos de uma mesma gravação ficam no mesmo
conjunto. Assim o teste só tem gravações que o modelo nunca ouviu.

Modo "trecho" (o da versão original do trabalho): cada trecho é sorteado sozinho, então
trechos vizinhos da mesma gravação caem no treino e no teste. O modelo reconhece a
gravação (ruído de fundo, microfone, o próprio pássaro) e a acurácia sai inflada.
Fica aqui só para medir esse vazamento.

A divisão é feita por espécie, então as proporções valem para cada classe.

Uso:
    python src/dividir_dataset.py --modo gravacao
"""

import argparse

import numpy as np
import pandas as pd

from config import ARQUIVO_DIVISAO, ESPECIES, PASTA_ESPECTROGRAMAS, PROPORCOES, SEMENTE

CONJUNTOS = list(PROPORCOES)


def listar_espectrogramas() -> pd.DataFrame:
    linhas = []
    for especie in ESPECIES:
        for caminho in sorted((PASTA_ESPECTROGRAMAS / especie).glob("*.png")):
            gravacao = caminho.stem.rsplit("_seg", 1)[0]
            linhas.append({"caminho": str(caminho.relative_to(PASTA_ESPECTROGRAMAS)),
                           "especie": especie, "gravacao": gravacao})
    if not linhas:
        raise SystemExit(f"Nenhum espectrograma em {PASTA_ESPECTROGRAMAS}. "
                         "Rode src/gerar_espectrogramas.py antes.")
    return pd.DataFrame(linhas)


def _repartir(itens: np.ndarray) -> dict[str, np.ndarray]:
    n_treino = int(round(len(itens) * PROPORCOES["treino"]))
    n_validacao = int(round(len(itens) * PROPORCOES["validacao"]))
    return {
        "treino": itens[:n_treino],
        "validacao": itens[n_treino:n_treino + n_validacao],
        "teste": itens[n_treino + n_validacao:],
    }


def dividir(modo: str = "gravacao", semente: int = SEMENTE) -> pd.DataFrame:
    dados = listar_espectrogramas()
    rng = np.random.default_rng(semente)
    dados["conjunto"] = ""

    for especie, grupo in dados.groupby("especie"):
        chave = "gravacao" if modo == "gravacao" else "caminho"
        unidades = np.array(sorted(grupo[chave].unique()), dtype=object)
        rng.shuffle(unidades)
        for conjunto, escolhidas in _repartir(unidades).items():
            filtro = (dados["especie"] == especie) & dados[chave].isin(escolhidas)
            dados.loc[filtro, "conjunto"] = conjunto
    return dados


def resumir(dados: pd.DataFrame) -> pd.DataFrame:
    return dados.groupby(["especie", "conjunto"]).agg(
        trechos=("caminho", "size"), gravacoes=("gravacao", "nunique")
    ).unstack("conjunto").reindex(columns=CONJUNTOS, level=1)


def gravacoes_em_comum(dados: pd.DataFrame, a: str, b: str) -> int:
    ga = set(dados.loc[dados["conjunto"] == a, "gravacao"])
    gb = set(dados.loc[dados["conjunto"] == b, "gravacao"])
    return len(ga & gb)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--modo", choices=["gravacao", "trecho"], default="gravacao")
    parser.add_argument("--semente", type=int, default=SEMENTE)
    args = parser.parse_args()

    dados = dividir(args.modo, args.semente)
    dados.to_csv(ARQUIVO_DIVISAO, index=False)
    print(resumir(dados))
    print(f"\nGravações do teste que também aparecem no treino: "
          f"{gravacoes_em_comum(dados, 'teste', 'treino')}")
    print(f"Divisão salva em {ARQUIVO_DIVISAO}")


if __name__ == "__main__":
    main()
