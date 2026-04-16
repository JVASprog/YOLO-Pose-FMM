"""
ETAPA 1 — Extração de Frames e Split do Dataset
================================================
Estrutura esperada dos vídeos:
  Angulo/
    superior/
      pegar_console/          → qualquer número de vídeos
      ler_etiqueta/           → qualquer número de vídeos
      girar_berco/            → qualquer número de vídeos
      montagem_parafusamento/ → qualquer número de vídeos
      inspecao_visual/        → qualquer número de vídeos
    paralelo/    (mesma estrutura)
    perspectiva/ (mesma estrutura)

Regras de split dinâmico:
  - 1 vídeo  → tudo vai para treino (sem val/test)
  - 2 vídeos → 1 treino, 1 teste (sem val)
  - 3 vídeos → 1 treino, 1 val, 1 teste
  - 4+       → proporcional: ~70% treino, ~15% val, ~15% teste
               garantindo no mínimo 1 vídeo em val e test

Resultado gerado:
  dataset_frames/
    train/superior/pegar_console/video1/frame_00001.jpg ...
    val/
    test/
"""

import cv2
import math
import random
import torch
from pathlib import Path

RAIZ_VIDEOS = "Angulo"
RAIZ_FRAMES = "dataset_frames"
FPS_ALVO    = 5
SEED        = 42
EXTENSOES   = {'.mp4', '.avi', '.mov', '.mkv'}

# Proporções alvo para o split (usadas quando há vídeos suficientes)
PROP_TRAIN = 0.70
PROP_VAL   = 0.15
PROP_TEST  = 0.15

ANGULOS = ["superior", "perspectiva"]
ATIVIDADES = [
    "pegar_console",
    "ler_etiqueta",
    "girar_berco",
    "montagem_parafusamento",
    "inspecao_visual",
]

def verificar_gpu():
    print("\n🖥️  VERIFICAÇÃO DE HARDWARE")
    print("-" * 40)
    if torch.cuda.is_available():
        nome = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"  ✅ GPU  : {nome}")
        print(f"  ✅ VRAM : {vram:.1f} GB")
        print(f"  ✅ CUDA : {torch.version.cuda}")
    else:
        print("  ⚠️  GPU NVIDIA não detectada — usando CPU.")
        print("     Execute: python 0_diagnostico.py para instruções.")
    print("-" * 40 + "\n")

def calcular_split(n: int) -> tuple[int, int, int]:
    """
    Recebe o total de vídeos e retorna (n_train, n_val, n_test).

    Casos especiais:
      n == 1 → (1, 0, 0)
      n == 2 → (1, 0, 1)
      n == 3 → (1, 1, 1)
      n >= 4 → proporcional com mínimo 1 em val e test
    """
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 0, 1
    if n == 3:
        return 1, 1, 1

    # n >= 4: proporcional, garantindo mínimo 1 em val e test
    n_test  = max(1, math.floor(n * PROP_TEST))
    n_val   = max(1, math.floor(n * PROP_VAL))
    n_train = n - n_val - n_test

    # Segurança: se arredondamento deixar train < 1, ajusta
    if n_train < 1:
        n_train = 1
        # Reduz val antes de test se necessário
        if n_val > 1:
            n_val -= 1
        else:
            n_test -= 1

    return n_train, n_val, n_test


def montar_split(raiz_videos, seed=42):
    random.seed(seed)
    split_map = {}
    avisos    = []

    for angulo in ANGULOS:
        for atividade in ATIVIDADES:
            pasta = Path(raiz_videos) / angulo / atividade
            if not pasta.exists():
                continue

            videos = sorted(f for f in pasta.iterdir()
                            if f.suffix.lower() in EXTENSOES)

            if not videos:
                avisos.append(f"⚠️  Sem vídeos em: {pasta}")
                continue

            n = len(videos)
            random.shuffle(videos)

            n_train, n_val, n_test = calcular_split(n)

            # Fatia os vídeos na ordem shuffled
            train_vids = videos[:n_train]
            val_vids   = videos[n_train:n_train + n_val]
            test_vids  = videos[n_train + n_val:]

            split_map[(angulo, atividade, "train")] = train_vids
            split_map[(angulo, atividade, "val")]   = val_vids
            split_map[(angulo, atividade, "test")]  = test_vids

            resumo = f"  {angulo}/{atividade}: {n} vídeo(s) → {n_train} treino | {n_val} val | {n_test} teste"
            avisos.append(resumo)

    # Exibe resumo do split antes de começar a extração
    print("\n📋 DISTRIBUIÇÃO DO SPLIT")
    print("-" * 60)
    for msg in avisos:
        print(msg)
    print("-" * 60 + "\n")

    return split_map

def extrair_frames(video_path, saida_dir, fps_alvo=5):
    saida_dir = Path(saida_dir)
    saida_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ❌ Não foi possível abrir: {video_path}")
        return 0

    fps_orig  = cap.get(cv2.CAP_PROP_FPS) or 30
    intervalo = max(1, int(fps_orig / fps_alvo))
    idx = salvo = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % intervalo == 0:
            cv2.imwrite(str(saida_dir / f"frame_{salvo:05d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            salvo += 1
        idx += 1

    cap.release()
    return salvo


def pipeline():
    print("=" * 60)
    print("  EXTRAÇÃO DE FRAMES E SPLIT DO DATASET")
    print("=" * 60)
    verificar_gpu()

    split_map = montar_split(RAIZ_VIDEOS, SEED)
    contagem  = {"train": 0, "val": 0, "test": 0}
    frames    = {"train": 0, "val": 0, "test": 0}

    for (angulo, atividade, split), videos in split_map.items():
        for v in videos:
            destino = Path(RAIZ_FRAMES) / split / angulo / atividade / v.stem
            print(f"🎬 [{split:5s}] {angulo:12s} | {atividade:25s} | {v.name}")
            n = extrair_frames(v, destino, FPS_ALVO)
            print(f"       → {n} frames → {destino}")
            contagem[split] += 1
            frames[split]   += n

    print("\n" + "=" * 60)
    print("  RESUMO")
    print("=" * 60)
    for s in ["train", "val", "test"]:
        print(f"  {s.upper():5s}: {contagem[s]:3d} vídeos | {frames[s]:6d} frames")
    print(f"  TOTAL: {sum(contagem.values())} vídeos | {sum(frames.values())} frames")
    print(f"\n  📂 {Path(RAIZ_FRAMES).resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    pipeline()
