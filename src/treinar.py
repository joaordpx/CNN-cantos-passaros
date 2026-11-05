"""Treina e avalia a CNN que classifica a espécie pelo espectrograma do canto.

Para cada semente: divide o dataset, treina com parada antecipada pela validação e
avalia no teste de dois jeitos:
  - por trecho: cada espectrograma de 5 s conta como um exemplo;
  - por gravação: as probabilidades dos trechos de uma gravação são somadas e a
    gravação inteira recebe uma única resposta (é como o modelo seria usado de verdade).

Uso:
    python src/treinar.py --modo gravacao --sementes 3
    python src/treinar.py --modo trecho --sementes 3     (só para medir o vazamento)
"""

import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import keras
from keras import layers
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from config import ESPECIES, PASTA_ESPECTROGRAMAS, PASTA_MODELOS, PASTA_RESULTADOS, SEMENTE, TAMANHO_IMAGEM
from dividir_dataset import dividir, gravacoes_em_comum

CLASSES = list(ESPECIES)


def carregar_imagens(caminhos: pd.Series) -> np.ndarray:
    return np.stack([
        keras.utils.img_to_array(
            keras.utils.load_img(PASTA_ESPECTROGRAMAS / c, target_size=TAMANHO_IMAGEM)
        ).astype("uint8")
        for c in caminhos
    ])


def criar_modelo() -> keras.Model:
    """Mesma arquitetura do trabalho original, para que a comparação meça só a divisão.

    O que mudou foi o aumento de dados: girar e espelhar um espectrograma inverte o tempo
    ou mistura frequências, o que não existe num canto real. Aqui só há deslocamento no
    tempo (eixo horizontal), que equivale a começar a gravação um pouco antes ou depois.
    """
    entrada = keras.Input(shape=(*TAMANHO_IMAGEM, 3))
    x = layers.Rescaling(1.0 / 255)(entrada)
    x = layers.RandomTranslation(height_factor=0.0, width_factor=0.1, fill_mode="wrap")(x)
    for filtros in (32, 64, 128):
        x = layers.Conv2D(filtros, 3, activation="relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    saida = layers.Dense(len(CLASSES), activation="softmax")(x)

    modelo = keras.Model(entrada, saida, name="cnn_cantos")
    modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return modelo


def pesos_das_classes(y: np.ndarray) -> dict[int, float]:
    contagem = np.bincount(y, minlength=len(CLASSES))
    return {i: len(y) / (len(CLASSES) * n) for i, n in enumerate(contagem)}


def salvar_curvas(historico: dict, destino) -> None:
    fig, eixos = plt.subplots(1, 2, figsize=(11, 4))
    for eixo, metrica, titulo in ((eixos[0], "accuracy", "Acurácia"), (eixos[1], "loss", "Perda")):
        eixo.plot(historico[metrica], label="Treino")
        eixo.plot(historico[f"val_{metrica}"], label="Validação")
        eixo.set_title(f"{titulo} por época")
        eixo.set_xlabel("Época")
        eixo.legend()
    fig.tight_layout()
    fig.savefig(destino, dpi=120)
    plt.close(fig)


def salvar_matriz(y_true, y_pred, titulo: str, destino) -> None:
    matriz = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    fig, eixo = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES,
                yticklabels=CLASSES, ax=eixo)
    eixo.set_xlabel("Previsão")
    eixo.set_ylabel("Verdadeiro")
    eixo.set_title(titulo)
    fig.tight_layout()
    fig.savefig(destino, dpi=120)
    plt.close(fig)


def rodar(modo: str, semente: int, epocas: int, pasta: "os.PathLike", salvar_modelo: bool) -> dict:
    keras.utils.set_random_seed(semente)
    pasta.mkdir(parents=True, exist_ok=True)

    dados = dividir(modo, semente)
    dados["rotulo"] = dados["especie"].map(CLASSES.index)
    dados.to_csv(pasta / "divisao.csv", index=False)
    conj = {nome: dados[dados["conjunto"] == nome] for nome in ("treino", "validacao", "teste")}
    x = {nome: carregar_imagens(df["caminho"]) for nome, df in conj.items()}
    y = {nome: df["rotulo"].to_numpy() for nome, df in conj.items()}

    modelo = criar_modelo()
    parada = keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    historico = modelo.fit(
        x["treino"], y["treino"],
        validation_data=(x["validacao"], y["validacao"]),
        epochs=epocas, batch_size=32, class_weight=pesos_das_classes(y["treino"]),
        callbacks=[parada], verbose=2,
    )

    probs = modelo.predict(x["teste"], verbose=0)
    pred_trecho = probs.argmax(axis=1)

    teste = conj["teste"].assign(**{f"p_{i}": probs[:, i] for i in range(len(CLASSES))})
    por_gravacao = teste.groupby("gravacao").agg(
        rotulo=("rotulo", "first"), **{f"p_{i}": (f"p_{i}", "sum") for i in range(len(CLASSES))}
    )
    pred_gravacao = por_gravacao[[f"p_{i}" for i in range(len(CLASSES))]].to_numpy().argmax(axis=1)

    metricas = {
        "modo": modo,
        "semente": semente,
        "epocas_treinadas": len(historico.history["loss"]),
        "melhor_epoca": int(np.argmin(historico.history["val_loss"])) + 1,
        "trechos_teste": int(len(teste)),
        "gravacoes_teste": int(len(por_gravacao)),
        "gravacoes_teste_tambem_no_treino": gravacoes_em_comum(dados, "teste", "treino"),
        "acuracia_trecho": float(accuracy_score(y["teste"], pred_trecho)),
        "f1_macro_trecho": float(f1_score(y["teste"], pred_trecho, average="macro")),
        "acuracia_gravacao": float(accuracy_score(por_gravacao["rotulo"], pred_gravacao)),
        "f1_macro_gravacao": float(f1_score(por_gravacao["rotulo"], pred_gravacao, average="macro")),
    }

    (pasta / "metricas.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")
    (pasta / "relatorio_por_trecho.txt").write_text(
        classification_report(y["teste"], pred_trecho, target_names=CLASSES, digits=3), encoding="utf-8")
    (pasta / "relatorio_por_gravacao.txt").write_text(
        classification_report(por_gravacao["rotulo"], pred_gravacao, target_names=CLASSES, digits=3),
        encoding="utf-8")
    salvar_curvas(historico.history, pasta / "curvas.png")
    salvar_matriz(y["teste"], pred_trecho, "Matriz de confusão (por trecho)", pasta / "matriz_por_trecho.png")
    salvar_matriz(por_gravacao["rotulo"], pred_gravacao, "Matriz de confusão (por gravação)",
                  pasta / "matriz_por_gravacao.png")

    if salvar_modelo:
        PASTA_MODELOS.mkdir(parents=True, exist_ok=True)
        modelo.save(PASTA_MODELOS / "cnn_cantos.keras")

    print(json.dumps(metricas, indent=2, ensure_ascii=False))
    return metricas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--modo", choices=["gravacao", "trecho"], default="gravacao")
    parser.add_argument("--sementes", type=int, default=1, help="quantas repetições (sementes 42, 43, ...)")
    parser.add_argument("--epocas", type=int, default=40, help="máximo de épocas")
    args = parser.parse_args()

    resultados = []
    for i in range(args.sementes):
        semente = SEMENTE + i
        pasta = PASTA_RESULTADOS / args.modo / f"semente_{semente}"
        salvar = args.modo == "gravacao" and i == 0
        resultados.append(rodar(args.modo, semente, args.epocas, pasta, salvar))

    tabela = pd.DataFrame(resultados)
    colunas = ["acuracia_trecho", "f1_macro_trecho", "acuracia_gravacao", "f1_macro_gravacao"]
    resumo = {
        "modo": args.modo,
        "sementes": tabela["semente"].tolist(),
        "media": tabela[colunas].mean().round(4).to_dict(),
        "desvio_padrao": tabela[colunas].std(ddof=0).round(4).to_dict(),
        "por_semente": resultados,
    }
    (PASTA_RESULTADOS / f"resumo_{args.modo}.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    print(tabela[["semente", *colunas]].to_string(index=False))


if __name__ == "__main__":
    main()
