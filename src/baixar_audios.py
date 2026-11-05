"""Baixa gravações de canto das espécies do projeto pela API v3 do xeno-canto.

A chave da API vem da variável de ambiente XENO_CANTO_API_KEY (nunca do código).
Além dos MP3, grava dados/creditos_xeno_canto.csv com autor, licença e link de cada
gravação, para dar o crédito exigido pelas licenças Creative Commons.

Uso:
    set XENO_CANTO_API_KEY=suachave      (Windows)
    python src/baixar_audios.py --limite 100
"""

import argparse
import csv
import os
import sys
import time

import requests

from config import ARQUIVO_CREDITOS, ESPECIES, PASTA_AUDIOS

URL_API = "https://xeno-canto.org/api/3/recordings"
QUALIDADES = ["A", "B"]
DURACAO_MINIMA_S = 5


def duracao_em_segundos(texto: str) -> int:
    partes = [int(p) for p in texto.split(":")]
    total = 0
    for parte in partes:
        total = total * 60 + parte
    return total


def ler_creditos() -> dict[str, dict]:
    if not ARQUIVO_CREDITOS.exists():
        return {}
    with open(ARQUIVO_CREDITOS, newline="", encoding="utf-8") as f:
        return {linha["id"]: linha for linha in csv.DictReader(f)}


def gravar_creditos(creditos: dict[str, dict]) -> None:
    campos = ["id", "especie", "autor", "licenca", "url"]
    ARQUIVO_CREDITOS.parent.mkdir(parents=True, exist_ok=True)
    with open(ARQUIVO_CREDITOS, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        for linha in sorted(creditos.values(), key=lambda l: (l["especie"], int(l["id"]))):
            escritor.writerow(linha)


def baixar_especie(nome: str, genero: str, especie: str, limite: int, chave: str,
                   creditos: dict[str, dict]) -> None:
    destino = PASTA_AUDIOS / nome
    destino.mkdir(parents=True, exist_ok=True)
    existentes = {arquivo.name.split("_")[0] for arquivo in destino.glob("*.mp3")}

    for qualidade in QUALIDADES:
        pagina = 1
        while len(existentes) < limite:
            consulta = f"gen:{genero} sp:{especie} cnt:Brazil type:song q:{qualidade}"
            resposta = requests.get(
                URL_API,
                params={"query": consulta, "page": pagina, "key": chave},
                timeout=30,
            )
            resposta.raise_for_status()
            dados = resposta.json()
            gravacoes = dados.get("recordings", [])
            if not gravacoes:
                break

            for rec in gravacoes:
                if len(existentes) >= limite:
                    break
                if duracao_em_segundos(rec.get("length", "0:00")) < DURACAO_MINIMA_S:
                    continue

                rec_id = str(rec["id"])
                creditos[rec_id] = {
                    "id": rec_id,
                    "especie": nome,
                    "autor": rec.get("rec", ""),
                    "licenca": rec.get("lic", ""),
                    "url": f"https://xeno-canto.org/{rec_id}",
                }
                if rec_id in existentes:
                    continue

                caminho = destino / f"{rec_id}_{rec['gen']}_{rec['sp']}.mp3"
                try:
                    audio = requests.get(rec["file"], timeout=60)
                    audio.raise_for_status()
                    # o servidor às vezes devolve uma página HTML de erro com status 200
                    if not audio.headers.get("Content-Type", "").startswith("audio"):
                        print(f"  ignorado {rec_id}: resposta não é áudio")
                        continue
                    caminho.write_bytes(audio.content)
                    existentes.add(rec_id)
                    print(f"  baixado {caminho.name}")
                except requests.RequestException as erro:
                    print(f"  falhou {rec_id}: {erro}")
                time.sleep(0.5)  # gentileza com o servidor

            if pagina >= int(dados.get("numPages", 1)):
                break
            pagina += 1

    print(f"{nome}: {len(existentes)} gravações")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limite", type=int, default=100, help="gravações por espécie")
    args = parser.parse_args()

    chave = os.environ.get("XENO_CANTO_API_KEY")
    if not chave:
        sys.exit("Defina a variável de ambiente XENO_CANTO_API_KEY (crie a chave em xeno-canto.org).")

    creditos = ler_creditos()
    for nome, (genero, especie) in ESPECIES.items():
        baixar_especie(nome, genero, especie, args.limite, chave, creditos)
    gravar_creditos(creditos)


if __name__ == "__main__":
    main()
