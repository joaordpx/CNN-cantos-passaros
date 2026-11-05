"""Diz qual das quatro espécies está cantando numa gravação (MP3 ou WAV).

A gravação é cortada em trechos de 5 s do mesmo jeito que no treino; o modelo dá uma
probabilidade para cada trecho e a resposta final soma todos eles.

Uso:
    python src/prever.py caminho/da/gravacao.mp3
"""

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import librosa
import numpy as np

from config import (
    DURACAO_MINIMA_S,
    DURACAO_TRECHO_S,
    ESPECIES,
    LIMIAR_SILENCIO_DBFS,
    PASTA_MODELOS,
    TAMANHO_IMAGEM,
    TAXA_AMOSTRAGEM,
)
from gerar_espectrogramas import salvar_espectrograma, volume_dbfs

CLASSES = list(ESPECIES)


def espectrogramas_da_gravacao(caminho: Path) -> np.ndarray:
    sinal, _ = librosa.load(caminho, sr=TAXA_AMOSTRAGEM, mono=True)
    amostras_trecho = int(DURACAO_TRECHO_S * TAXA_AMOSTRAGEM)
    imagens = []
    with tempfile.TemporaryDirectory() as pasta:
        for i, inicio in enumerate(range(0, len(sinal), amostras_trecho)):
            trecho = sinal[inicio:inicio + amostras_trecho]
            if len(trecho) < DURACAO_MINIMA_S * TAXA_AMOSTRAGEM or volume_dbfs(trecho) < LIMIAR_SILENCIO_DBFS:
                continue
            trecho = np.pad(trecho, (0, amostras_trecho - len(trecho)))
            arquivo = Path(pasta) / f"{i}.png"
            salvar_espectrograma(trecho, arquivo)
            imagens.append(keras.utils.img_to_array(keras.utils.load_img(arquivo, target_size=TAMANHO_IMAGEM)))
    return np.stack(imagens) if imagens else np.empty((0, *TAMANHO_IMAGEM, 3))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("gravacao", type=Path)
    parser.add_argument("--modelo", type=Path, default=PASTA_MODELOS / "cnn_cantos.keras")
    args = parser.parse_args()

    imagens = espectrogramas_da_gravacao(args.gravacao)
    if len(imagens) == 0:
        raise SystemExit("A gravação não tem nenhum trecho com som suficiente.")

    modelo = keras.models.load_model(args.modelo)
    probs = modelo.predict(imagens, verbose=0)
    media = probs.mean(axis=0)

    print(f"{len(imagens)} trechos analisados")
    for indice in np.argsort(media)[::-1]:
        print(f"  {CLASSES[indice]:<18} {media[indice]:6.1%}")
    print(f"\nEspécie prevista: {CLASSES[int(media.argmax())]}")


if __name__ == "__main__":
    main()
