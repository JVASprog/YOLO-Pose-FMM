"""
ETAPA 3 — Treino do YOLO-Pose (GPU)
=====================================
Treina YOLOv8-Pose no dataset anotado.
Ajusta automaticamente o batch size conforme
a VRAM disponível na sua GPU NVIDIA.
"""

import torch
from ultralytics import YOLO
from pathlib import Path

DATASET_YAML = "dataset_yolo/data.yaml"
MODELO_BASE  = "yolov8m-pose.pt"
EPOCHS       = 100
IMGSZ        = 640
PROJETO      = "treino_pose"
NOME         = "v1"

def configurar_hardware():
    """
    Detecta a GPU e sugere batch size baseado na VRAM.
    Retorna (device, batch_size, usar_fp16).
    """
    print("\n🖥️  CONFIGURAÇÃO DE HARDWARE")
    print("-" * 40)

    if not torch.cuda.is_available():
        print("  ⚠️  GPU não detectada — treinando na CPU.")
        print("     Isso será MUITO mais lento.")
        print("     Execute python 0_diagnostico.py para instruções.")
        return "cpu", 4, False

    nome = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    cuda_version = torch.version.cuda

    print(f"  ✅ GPU         : {nome}")
    print(f"  ✅ VRAM        : {vram:.1f} GB")
    print(f"  ✅ CUDA        : {cuda_version}")
    print(f"  ✅ PyTorch     : {torch.__version__}")

    # Batch size recomendado por faixa de VRAM
    if vram >= 16:
        batch = 32
    elif vram >= 8:
        batch = 16
    elif vram >= 6:
        batch = 8
    else:
        batch = 4

    # FP16 (half precision) — mais rápido e usa menos VRAM
    # Compatível com GPUs Ampere (RTX 30xx) e mais recentes
    usar_fp16 = vram >= 4

    print(f"\n  📐 Batch size   : {batch}  (baseado em {vram:.0f} GB VRAM)")
    print(f"  📐 Precisão FP16: {'Ativada' if usar_fp16 else 'Desativada'}")
    print("-" * 40 + "\n")

    return 0, batch, usar_fp16   # device=0 → primeira GPU

def treinar():
    print("=" * 60)
    print("  TREINO YOLO-POSE")
    print("=" * 60)

    device, batch, fp16 = configurar_hardware()

    print(f"  Modelo base  : {MODELO_BASE}")
    print(f"  Dataset      : {DATASET_YAML}")
    print(f"  Épocas       : {EPOCHS}")
    print(f"  Imagem (px)  : {IMGSZ}")
    print("=" * 60 + "\n")

    model = YOLO(MODELO_BASE)

    results = model.train(
        data       = DATASET_YAML,
        epochs     = EPOCHS,
        imgsz      = IMGSZ,
        batch      = batch,
        device     = device,
        half       = fp16,        
        patience   = 20,
        hsv_h      = 0.015,
        hsv_s      = 0.7,
        hsv_v      = 0.4,
        degrees    = 10.0,
        translate  = 0.1,
        scale      = 0.3,
        fliplr     = 0.5,
        flipud     = 0.0,
        mosaic     = 0.5,
        optimizer  = "AdamW",
        lr0        = 0.001,
        weight_decay = 0.0005,
        save         = True,
        save_period  = 10,
        project      = PROJETO,
        name         = NOME,
        plots        = True,
        verbose      = True,
        cache        = True,
        workers      = 4,
    )

    melhor = Path(PROJETO) / NOME / "weights" / "best.pt"
    print(f"\n✅ Treino concluído!")
    print(f"   Melhor modelo : {melhor.resolve()}")

    # Liberar memória GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("   Cache GPU liberado.")

    return str(melhor)


if __name__ == "__main__":
    treinar()
