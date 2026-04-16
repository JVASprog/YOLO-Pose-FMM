"""
ETAPA 2 — Anotação Automática com YOLO-Pose (GPU)
==================================================
Lê os frames de dataset_frames/, roda YOLOv8-pose
na GPU para detectar keypoints e salva anotações
no formato YOLO-Pose (.txt) + data.yaml.

Formato por linha:
  <class_id> <xc> <yc> <w> <h>
  <kp1_x> <kp1_y> <kp1_vis> ... <kp17_x> <kp17_y> <kp17_vis>
  (tudo normalizado 0-1)
"""

import cv2
import shutil
import torch
import yaml
from pathlib import Path
from ultralytics import YOLO

RAIZ_FRAMES = "dataset_frames"
RAIZ_YOLO   = "dataset_yolo"
MODELO_BASE = "yolov8m-pose.pt"
CONF_MINIMA = 0.5

ATIVIDADES = [
    "pegar_console",           # 0
    "ler_etiqueta",            # 1
    "girar_berco",             # 2
    "montagem_parafusamento",  # 3
    "inspecao_visual",         # 4
]

ANGULOS = ["superior", "perspectiva"]

def selecionar_device():
    if torch.cuda.is_available():
        nome = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"  ✅ GPU : {nome}  ({vram:.1f} GB VRAM)")
        return 0          # índice da GPU para o YOLO
    else:
        print("  ⚠️  GPU não detectada — usando CPU (mais lento).")
        return "cpu"

def bbox_yolo(x1, y1, x2, y2, W, H):
    xc = ((x1 + x2) / 2) / W
    yc = ((y1 + y2) / 2) / H
    w  = (x2 - x1) / W
    h  = (y2 - y1) / H
    return xc, yc, w, h


def kp_yolo(kp_array, W, H):
    partes = []
    for kp in kp_array:
        x   = float(kp[0]) / W
        y   = float(kp[1]) / H
        vis = 2 if float(kp[2]) > 0.5 else (1 if float(kp[2]) > 0.1 else 0)
        partes += [f"{x:.6f}", f"{y:.6f}", str(vis)]
    return " ".join(partes)


def anotar_frame(model, img_path, class_id):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    H, W = img.shape[:2]

    # half=True usa FP16 na GPU → ~2× mais rápido
    results = model(img, verbose=False, conf=CONF_MINIMA, half=torch.cuda.is_available())

    linhas = []
    for r in results:
        if r.keypoints is None or r.boxes is None:
            continue
        for box, kp in zip(r.boxes.xyxy.cpu().numpy(),
                           r.keypoints.data.cpu().numpy()):
            xc, yc, w, h = bbox_yolo(*box, W, H)
            linha = f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {kp_yolo(kp, W, H)}"
            linhas.append(linha)

    return linhas or None

def anotar_dataset():
    print("=" * 60)
    print("  ANOTAÇÃO AUTOMÁTICA COM YOLO-POSE (GPU)")
    print("=" * 60)

    device = selecionar_device()
    model  = YOLO(MODELO_BASE)
    model.to(device if device == "cpu" else f"cuda:{device}")

    contagem = {"total": 0, "ok": 0, "vazio": 0}

    for split in ["train", "val", "test"]:
        for angulo in ANGULOS:
            for atividade in ATIVIDADES:
                class_id = ATIVIDADES.index(atividade)
                pasta = Path(RAIZ_FRAMES) / split / angulo / atividade

                if not pasta.exists():
                    continue  # ângulo ou atividade ausente — pula sem erro

                for video_dir in sorted(pasta.iterdir()):
                    if not video_dir.is_dir():
                        continue

                    for img_path in sorted(video_dir.glob("*.jpg")):
                        contagem["total"] += 1

                        # Caminho espelho em dataset_yolo/
                        rel       = img_path.relative_to(Path(RAIZ_FRAMES) / split)
                        nome_flat = "__".join(rel.parts)  # achata subpastas no nome

                        img_dest   = Path(RAIZ_YOLO) / split / "images" / nome_flat
                        label_dest = (Path(RAIZ_YOLO) / split / "labels"
                                      / nome_flat.replace(".jpg", ".txt"))

                        img_dest.parent.mkdir(parents=True, exist_ok=True)
                        label_dest.parent.mkdir(parents=True, exist_ok=True)

                        linhas = anotar_frame(model, img_path, class_id)

                        if linhas:
                            shutil.copy2(img_path, img_dest)
                            label_dest.write_text("\n".join(linhas))
                            contagem["ok"] += 1
                        else:
                            contagem["vazio"] += 1

                        if contagem["total"] % 200 == 0:
                            pct = contagem["ok"] / contagem["total"] * 100
                            print(f"  Processados {contagem['total']:5d} | "
                                  f"Anotados {contagem['ok']:5d} ({pct:.0f}%) | "
                                  f"Vazios {contagem['vazio']:4d}")

    gerar_yaml(RAIZ_YOLO)

    print("\n" + "=" * 60)
    print("  RESUMO")
    print("=" * 60)
    print(f"  Total frames    : {contagem['total']}")
    print(f"  Anotados        : {contagem['ok']}")
    print(f"  Sem detecção    : {contagem['vazio']}")
    cobertura = contagem["ok"] / contagem["total"] * 100 if contagem["total"] else 0
    print(f"  Cobertura       : {cobertura:.1f}%")
    print(f"\n  📂 {Path(RAIZ_YOLO).resolve()}")
    print("=" * 60)


def gerar_yaml(raiz):
    cfg = {
        "path":       str(Path(raiz).resolve()),
        "train":      "train/images",
        "val":        "val/images",
        "test":       "test/images",
        "kpt_shape":  [17, 3],
        "names":      {i: a for i, a in enumerate(ATIVIDADES)},
        "skeleton": [
            [16,14],[14,12],[17,15],[15,13],[12,13],
            [6,12],[7,13],[6,7],[6,8],[7,9],
            [8,10],[9,11],[2,3],[1,2],[1,3],
            [2,4],[3,5],[4,6],[5,7],
        ],
    }
    p = Path(raiz) / "data.yaml"
    with open(p, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    print(f"\n  ✅ data.yaml → {p}")


if __name__ == "__main__":
    anotar_dataset()
