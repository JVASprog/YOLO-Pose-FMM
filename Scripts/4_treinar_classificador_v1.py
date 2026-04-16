"""
ETAPA 4 — Treino do Classificador LSTM com Packed Sequences (GPU)
=================================================================
Estratégia Definitiva: Extração Invariante (Apenas Ângulos e Proporções)
  - Abandono das coordenadas XY e Bounding Boxes.
  - Features reduzidas para 13 (10 ângulos corporais + 3 proporções de mãos).
  - Previne totalmente o overfitting por "memorização de posição na imagem".
"""

import cv2
import json
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from ultralytics import YOLO

MODELO_POSE  = "runs/pose/treino_pose/v1/weights/best.pt"
RAIZ_FRAMES  = "dataset_frames"
SAIDA        = "modelos_classificador"
CONF_MINIMA  = 0.5

ATIVIDADES = [
    "pegar_console",
    "ler_etiqueta",
    "girar_berco",
    "montagem_parafusamento",
    "inspecao_visual",
]
ANGULOS = ["superior", "perspectiva"]

JANELA_POR_CLASSE = {
    "pegar_console":          12,
    "ler_etiqueta":            9,
    "girar_berco":            12,
    "montagem_parafusamento": 40,
    "inspecao_visual":        12,
}

PASSO_POR_CLASSE = {
    "pegar_console":           4,
    "ler_etiqueta":            1,
    "girar_berco":             4,
    "montagem_parafusamento": 20,
    "inspecao_visual":         4,
}

JANELA_MAX = max(JANELA_POR_CLASSE.values())

INPUT_SIZE    = 13       
HIDDEN_SIZE   = 32       
NUM_LAYERS    = 1
DROPOUT       = 0.5
NUM_CLASSES   = len(ATIVIDADES)

EPOCHS        = 80
BATCH_SIZE    = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY  = 1e-3
PATIENCE      = 25

def selecionar_device():
    print("\n🖥️  HARDWARE")
    print("-" * 45)
    if torch.cuda.is_available():
        nome = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"  ✅ GPU  : {nome}  ({vram:.1f} GB VRAM)")
        print(f"  ✅ CUDA : {torch.version.cuda}")
        device = torch.device("cuda:0")
    else:
        print("  ⚠️  GPU não detetada — a usar CPU.")
        device = torch.device("cpu")
    print("-" * 45 + "\n")
    return device

def _angulo(a, b, c):
    A, B, C = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
    ba, bc  = A - B, C - B
    norm    = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm == 0:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / norm, -1, 1))))

def _dist(a, b):
    return float(np.linalg.norm(np.array(a[:2]) - np.array(b[:2])))

def extrair_features(kp, bbox):
    feats = [
        _angulo(kp[5], kp[7],  kp[9]),   # 1. Cotovelo Esquerdo
        _angulo(kp[6], kp[8],  kp[10]),  # 2. Cotovelo Direito
        _angulo(kp[11], kp[5], kp[7]),   # 3. Ombro Esquerdo (Axila)
        _angulo(kp[12], kp[6], kp[8]),   # 4. Ombro Direito (Axila)
        _angulo(kp[0], kp[5],  kp[7]),   # 5. Pescoço ao Ombro Esq
        _angulo(kp[0], kp[6],  kp[8]),   # 6. Pescoço ao Ombro Dir
        _angulo(kp[5], kp[11], kp[13]),  # 7. Anca Esquerda
        _angulo(kp[6], kp[12], kp[14]),  # 8. Anca Direita
        _angulo(kp[11], kp[13], kp[15]), # 9. Joelho Esquerdo
        _angulo(kp[12], kp[14], kp[16]), # 10. Joelho Direito
    ]

    ombros_dist = _dist(kp[5], kp[6])
    if ombros_dist < 1e-5:
        ombros_dist = 1.0 # Previne divisão por zero se a deteção falhar

    feats += [
        _dist(kp[9], kp[10]) / ombros_dist, # 11. Distância entre as mãos
        _dist(kp[9], kp[0])  / ombros_dist, # 12. Distância mão esq ao rosto
        _dist(kp[10], kp[0]) / ombros_dist, # 13. Distância mão dir ao rosto
    ]

    return np.array(feats, dtype=np.float32)

def coletar_por_video(model_pose, split, device):
    videos_data = []
    fp16 = device.type == "cuda"

    for angulo in ANGULOS:
        for atividade in ATIVIDADES:
            pasta = Path(RAIZ_FRAMES) / split / angulo / atividade
            if not pasta.exists():
                continue
            janela_classe = JANELA_POR_CLASSE[atividade]

            for video_dir in sorted(pasta.iterdir()):
                if not video_dir.is_dir():
                    continue
                frames = sorted(video_dir.glob("*.jpg"))
                if not frames:
                    continue

                feats_video = []
                for img_path in frames:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    results = model_pose(img, verbose=False,
                                         conf=CONF_MINIMA, half=fp16)
                    for r in results:
                        if r.keypoints is None or r.boxes is None:
                            continue
                        kps = r.keypoints.data.cpu().numpy()
                        bbs = r.boxes.xyxy.cpu().numpy()
                        if len(bbs) == 0:
                            continue
                        idx = int(np.argmax(r.boxes.conf.cpu().numpy()))
                        f   = extrair_features(kps[idx], bbs[idx])
                        if f is not None:
                            feats_video.append(f)

                if len(feats_video) >= janela_classe:
                    videos_data.append({
                        "features":  np.array(feats_video, dtype=np.float32),
                        "atividade": atividade,
                        "angulo":    angulo,
                    })
    return videos_data

def videos_para_sequencias(videos_data, le):
    X_list, y_list, lens, angs = [], [], [], []

    for v in videos_data:
        atividade = v["atividade"]
        feats     = v["features"]
        label     = le.transform([atividade])[0]
        janela    = JANELA_POR_CLASSE[atividade]
        passo     = PASSO_POR_CLASSE[atividade]
        n         = len(feats)

        for start in range(0, n - janela + 1, passo):
            X_list.append(feats[start : start + janela])
            y_list.append(label)
            lens.append(janela)
            angs.append(v["angulo"])

    return (X_list,
            np.array(y_list, dtype=np.int64),
            np.array(lens,   dtype=np.int64),
            np.array(angs))

class SequenceDataset(Dataset):
    def __init__(self, X_list, y, lens, augment=False):
        self.X_list  = X_list
        self.y       = torch.tensor(y,    dtype=torch.long)
        self.lens    = torch.tensor(lens, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        seq = self.X_list[idx].copy() 
        
        if self.augment:
            ruido = np.random.normal(0, 0.15, seq.shape).astype(np.float32)
            seq = seq + ruido

        pad_len = JANELA_MAX - len(seq)
        if pad_len > 0:
            pad = np.zeros((pad_len, INPUT_SIZE), dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=0)  
            
        return (torch.tensor(seq, dtype=torch.float32),
                self.y[idx],
                self.lens[idx])


def collate_fn(batch):
    seqs, labels, lens = zip(*batch)
    seqs   = torch.stack(seqs)
    labels = torch.stack(labels)
    lens   = torch.stack(lens)
    return seqs, labels, lens

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 num_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = 0, 
            bidirectional = False,
        )
        self.norm    = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lens=None):
        if lens is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lens.cpu(), batch_first=True, enforce_sorted=False)
            _, (h_n, _) = self.lstm(packed)
            last = h_n[-1]
        else:
            _, (h_n, _) = self.lstm(x)
            last = h_n[-1]

        return self.fc(self.dropout(self.norm(last)))

def treinar_lstm(X_tr, y_tr, l_tr, X_v, y_v, l_v, device):
    contagem = np.bincount(y_tr, minlength=NUM_CLASSES)
    pesos    = 1.0 / (np.sqrt(contagem) + 1e-6)
    pesos    = pesos / pesos.sum() * NUM_CLASSES
    pesos_t  = torch.tensor(pesos, dtype=torch.float32).to(device)

    ds_tr = SequenceDataset(X_tr, y_tr, l_tr, augment=True)
    ds_v  = SequenceDataset(X_v,  y_v,  l_v,  augment=False)
    
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,
                       collate_fn=collate_fn, num_workers=0)
    dl_v  = DataLoader(ds_v,  batch_size=BATCH_SIZE, shuffle=False,
                       collate_fn=collate_fn, num_workers=0)

    model     = LSTMClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
                                NUM_CLASSES, DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss(weight=pesos_t)
    optimizer = optim.AdamW(model.parameters(),
                            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7)

    melhor_val  = float("inf")
    sem_melhora = 0
    historico   = {"train_loss": [], "val_loss": [], "val_acc": []}

    print(f"\n{'Época':>6} {'Loss Treino':>12} {'Loss Val':>10} {'Acc Val':>8} {'LR':>10}")
    print("-" * 52)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for X_b, y_b, l_b in dl_tr:
            X_b, y_b, l_b = X_b.to(device), y_b.to(device), l_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b, l_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += loss.item()
        avg_tr = total / len(dl_tr)

        model.eval()
        val_loss = corretos = 0
        with torch.no_grad():
            for X_b, y_b, l_b in dl_v:
                X_b, y_b, l_b = X_b.to(device), y_b.to(device), l_b.to(device)
                logits    = model(X_b, l_b)
                val_loss += criterion(logits, y_b).item()
                corretos += (logits.argmax(1) == y_b).sum().item()
        avg_v  = val_loss / len(dl_v)
        acc_v  = corretos / len(ds_v)
        lr_now = optimizer.param_groups[0]["lr"]

        scheduler.step(avg_v)
        historico["train_loss"].append(avg_tr)
        historico["val_loss"].append(avg_v)
        historico["val_acc"].append(acc_v)

        print(f"{epoch:>6} {avg_tr:>12.4f} {avg_v:>10.4f} {acc_v:>8.2%} {lr_now:>10.2e}")

        if avg_v < melhor_val:
            melhor_val  = avg_v
            sem_melhora = 0
            Path(SAIDA).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), f"{SAIDA}/lstm_model.pt")
        else:
            sem_melhora += 1
            if sem_melhora >= PATIENCE:
                print(f"\n  ⏹  Early stopping na época {epoch}")
                break

    _plotar_historico(historico)
    return model


def _plotar_historico(hist):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(hist["train_loss"], label="Treino")
    ax1.plot(hist["val_loss"],   label="Validação")
    ax1.set_title("Loss por Época"); ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(hist["val_acc"])
    ax2.set_title("Acurácia na Validação"); ax2.set_xlabel("Época")
    ax2.set_ylabel("Acurácia")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(SAIDA).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{SAIDA}/historico_treino.png", dpi=120)
    plt.close()

def avaliar_lstm(model, le, X, y_enc, lens, angulos, titulo, pasta_saida, device):
    Path(pasta_saida).mkdir(parents=True, exist_ok=True)
    model.eval()

    ds  = SequenceDataset(X, y_enc, lens, augment=False)
    dl  = DataLoader(ds, batch_size=64, shuffle=False,
                     collate_fn=collate_fn, num_workers=0)
    preds = []
    with torch.no_grad():
        for X_b, _, l_b in dl:
            logits = model(X_b.to(device), l_b.to(device))
            preds.extend(logits.argmax(1).cpu().numpy())

    y_pred  = le.inverse_transform(preds)
    y_true  = le.inverse_transform(y_enc)
    classes = list(le.classes_)

    print(f"\n{'='*60}\n  AVALIAÇÃO LSTM — {titulo}\n{'='*60}")
    print(classification_report(
        y_true, y_pred, labels=classes, target_names=classes, zero_division=0))

    _confusion(y_true, y_pred, classes, f"Geral — {titulo}", f"{pasta_saida}/confusion_geral.png")


def _confusion(y_true, y_pred, classes, titulo, caminho):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title(titulo, fontsize=12)
    plt.xlabel("Predito"); plt.ylabel("Real")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(caminho, dpi=120)
    plt.close()

def pipeline():
    device     = selecionar_device()
    model_pose = YOLO(MODELO_POSE)
    model_pose.to(device)

    print("\n🔍 A coletar TREINO (Extração Geométrica Invariante)...")
    dados_tr = coletar_por_video(model_pose, "train", device)
    print("\n🔍 A coletar VALIDAÇÃO (Extração Geométrica Invariante)...")
    dados_v  = coletar_por_video(model_pose, "val",   device)
    print("\n🔍 A coletar TESTE (Extração Geométrica Invariante)...")
    dados_te = coletar_por_video(model_pose, "test",  device)

    le = LabelEncoder()
    le.fit(ATIVIDADES)

    todas_feats = np.vstack([v["features"] for v in dados_tr])
    scaler = StandardScaler()
    scaler.fit(todas_feats)

    def normalizar(videos):
        for v in videos:
            v["features"] = scaler.transform(v["features"])
        return videos

    dados_tr = normalizar(dados_tr)
    dados_v  = normalizar(dados_v)
    dados_te = normalizar(dados_te)

    X_tr, y_tr, l_tr, a_tr = videos_para_sequencias(dados_tr, le)
    X_v,  y_v,  l_v,  a_v  = videos_para_sequencias(dados_v,  le)
    X_te, y_te, l_te, a_te = videos_para_sequencias(dados_te, le)

    print("\n🏋️  A treinar LSTM (Geometria + Balanceamento)...")
    model = treinar_lstm(X_tr, y_tr, l_tr, X_v, y_v, l_v, device)

    model.load_state_dict(torch.load(f"{SAIDA}/lstm_model.pt", map_location=device))

    joblib.dump(le,     f"{SAIDA}/label_encoder.pkl")
    joblib.dump(scaler, f"{SAIDA}/scaler.pkl")

    # Salva config.json com TODOS os parâmetros necessários para o script 5.
    # Sempre que alterar qualquer hiperparâmetro neste script,
    # o config.json será regenerado automaticamente ao treinar.
    config = {
        # Arquitetura do LSTM (fonte de verdade também está no checkpoint,
        # mas ter aqui facilita debugging e visualização)
        "input_size":    INPUT_SIZE,
        "hidden_size":   HIDDEN_SIZE,
        "num_layers":    NUM_LAYERS,
        "dropout":       DROPOUT,
        "num_classes":   NUM_CLASSES,
        "atividades":        list(le.classes_),
        "janela_por_classe": JANELA_POR_CLASSE,
        "janela_max":        JANELA_MAX,
        "passo_por_classe":  PASSO_POR_CLASSE,
        "feature_mode":   "geometrico",  
        "padding_mode":   "post",        
        "enforce_sorted": False,         
    }
    with open(f"{SAIDA}/config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ config.json salvo em: {Path(SAIDA).resolve()}/config.json")

    avaliar_lstm(model, le, X_v,  y_v,  l_v,  a_v, "VALIDAÇÃO", f"{SAIDA}/avaliacao_val",  device)
    avaliar_lstm(model, le, X_te, y_te, l_te, a_te, "TESTE",     f"{SAIDA}/avaliacao_test", device)


if __name__ == "__main__":
    pipeline()
