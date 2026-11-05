"""Corta cada gravação em trechos de 5 s e salva o espectrograma mel de cada trecho.

Trechos silenciosos (volume médio abaixo do limiar) são descartados. O último trecho de
cada gravação, se tiver ao menos 1 s, é completado com silêncio até 5 s, para que todas
as imagens tenham a mesma escala de tempo.

O nome do arquivo guarda a gravação de origem (ex.: 210230_Pitangus_sulphuratus_seg1.png),
e é isso que permite dividir o dataset por gravação depois.

Uso:
    python src/gerar_espectrogramas.py
"""

import librosa
import matplotlib.pyplot as plt
import numpy as np

from config import (
    DURACAO_MINIMA_S,
    DURACAO_TRECHO_S,
    ESPECIES,
    LIMIAR_SILENCIO_DBFS,
    N_MELS,
    PASTA_AUDIOS,
    PASTA_ESPECTROGRAMAS,
    TAXA_AMOSTRAGEM,
)


def volume_dbfs(sinal: np.ndarray) -> float:
    rms = np.sqrt(np.mean(sinal**2))
    return -np.inf if rms == 0 else 20 * np.log10(rms)


def salvar_espectrograma(trecho: np.ndarray, destino) -> None:
    mel = librosa.feature.melspectrogram(y=trecho, sr=TAXA_AMOSTRAGEM, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # frequências baixas embaixo, como no specshow
    plt.imsave(destino, mel_db, cmap="magma", origin="lower")


def processar_gravacao(caminho, pasta_saida) -> int:
    sinal, _ = librosa.load(caminho, sr=TAXA_AMOSTRAGEM, mono=True)
    amostras_trecho = int(DURACAO_TRECHO_S * TAXA_AMOSTRAGEM)
    amostras_minimas = int(DURACAO_MINIMA_S * TAXA_AMOSTRAGEM)
    salvos = 0

    for i, inicio in enumerate(range(0, len(sinal), amostras_trecho), start=1):
        trecho = sinal[inicio:inicio + amostras_trecho]
        if len(trecho) < amostras_minimas:
            continue
        if volume_dbfs(trecho) < LIMIAR_SILENCIO_DBFS:
            continue
        if len(trecho) < amostras_trecho:
            trecho = np.pad(trecho, (0, amostras_trecho - len(trecho)))

        salvar_espectrograma(trecho, pasta_saida / f"{caminho.stem}_seg{i}.png")
        salvos += 1
    return salvos


def main() -> None:
    for especie in ESPECIES:
        pasta_entrada = PASTA_AUDIOS / especie
        pasta_saida = PASTA_ESPECTROGRAMAS / especie
        pasta_saida.mkdir(parents=True, exist_ok=True)

        gravacoes = sorted(pasta_entrada.glob("*.mp3"))
        total = 0
        for caminho in gravacoes:
            try:
                total += processar_gravacao(caminho, pasta_saida)
            except Exception as erro:  # arquivo corrompido não deve parar o lote
                print(f"  erro em {caminho.name}: {erro}")
        print(f"{especie}: {len(gravacoes)} gravações -> {total} espectrogramas")


if __name__ == "__main__":
    main()
