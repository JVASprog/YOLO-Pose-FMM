"""
ETAPA 5 — Inferência em Tempo Real com LSTM (GPU)
==================================================
Script totalmente dinâmico: lê TODOS os parâmetros de arquitetura
diretamente do checkpoint (lstm_model.pt), sem depender de nenhum
valor hardcoded ou de config.json.

Como funciona:
  - input_size, hidden_size, num_layers → lidos dos tensores salvos
  - janela_por_classe, num_classes      → lidos do config.json (se existir)
                                          ou inferidos do label_encoder.pkl
  - Se você mudar HIDDEN_SIZE, NUM_LAYERS ou INPUT_SIZE no script 4
    e treinar de novo, este script se adapta automaticamente.

Modos:
  python 5_inferencia.py                          # webcam padrão (índice 0)
  python 5_inferencia.py --camera 2               # webcam USB no índice 2
  python 5_inferencia.py --listar-cameras         # lista todas as câmeras disponíveis
  python 5_inferencia.py --source video.mp4       # arquivo de vídeo (exibe ao vivo)
  python 5_inferencia.py --source video.mp4 --salvar          # salva em 1080p
  python 5_inferencia.py --source video.mp4 --salvar --saida resultado.mp4
  python 5_inferencia.py --sem-classificador      # só pose, sem LSTM

Como descobrir o índice da webcam USB:
  Execute com --listar-cameras e o script testa os índices 0-9,
  mostrando resolução e se a câmera está disponível.
  A câmera integrada do portátil costuma ser o índice 0;
  a USB costuma ser 1 ou 2 dependendo da ordem de ligação.

Notas sobre --salvar:
  - A saída é sempre em 1080p (1920×1080), com upscale proporcional e
    letterbox preto para preencher o espaço restante.
  - O codec padrão é mp4v (.mp4). Se quiser H.264, instale opencv com
    suporte a ffmpeg e troque o fourcc para 'avc1' ou 'H264'.
  - No modo --salvar a janela de visualização NÃO é aberta, o que
    acelera o processamento significativamente.
  - O nome do arquivo de saída é gerado automaticamente se --saida não
    for especificado: inferencia_YYYYMMDD_HHMMSS.mp4
"""

import cv2
import json
import numpy as np
import joblib
import torch
import torch.nn as nn
import argparse
import time
from pathlib import Path
from collections import deque, Counter
from ultralytics import YOLO
from datetime import datetime
from PIL import ImageFont, ImageDraw, Image

try:
    import psycopg2
    POSTGRES_OK = True
except ImportError:
    print("⚠️  psycopg2 não instalado. Execute: pip install psycopg2-binary")
    POSTGRES_OK = False


# ──────────────────────────────────────────────────────────────
# LISTAGEM DE CÂMERAS DISPONÍVEIS
# ──────────────────────────────────────────────────────────────
def listar_cameras(max_idx=9):
    """
    Testa os índices 0..max_idx e imprime as câmeras que respondem.
    Útil para identificar qual índice corresponde à webcam USB.

    A câmera integrada do portátil costuma ser o índice 0.
    A webcam USB costuma aparecer como índice 1 ou 2 dependendo
    da ordem em que foi ligada ao sistema.

    No Windows o backend padrão é DSHOW; no Linux é V4L2.
    Se uma câmera aparecer mas com resolução 0×0, tenta abri-la
    manualmente com --camera N para confirmar.
    """
    print("\n🔍 A procurar câmeras disponíveis (índices 0 a %d)..." % max_idx)
    print("-" * 50)
    encontradas = []
    for i in range(max_idx + 1):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_cam = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            encontradas.append(i)
            print(f"  ✅ Índice {i}  →  {w}×{h}  @ {fps_cam:.0f}fps")
        else:
            cap.release()
    print("-" * 50)
    if not encontradas:
        print("  ⚠️  Nenhuma câmera encontrada.")
    else:
        print(f"  ℹ️  Use --camera N para selecionar o índice desejado.")
        print(f"      Exemplo: python 5_inferencia.py --camera {encontradas[-1]}")
    print()


# ──────────────────────────────────────────────────────────────
# CAMINHOS DOS ARQUIVOS
# ──────────────────────────────────────────────────────────────
MODELO_POSE   = "runs/pose/treino_pose/v1/weights/best.pt"
MODELO_BASE   = "yolov8m-pose.pt"
LSTM_WEIGHTS  = "modelos_classificador/lstm_model.pt"
LABEL_ENCODER = "modelos_classificador/label_encoder.pkl"
SCALER        = "modelos_classificador/scaler.pkl"
CONFIG_JSON   = "modelos_classificador/config.json"   # opcional

# Resolução do vídeo salvo (--salvar)
SAIDA_LARGURA = 1920
SAIDA_ALTURA  = 1080
SAIDA_FPS     = 30          # fps fixo no ficheiro de saída

CONF_POSE           = 0.3   # limiar baixo — detecta mesmo em poses parciais
CONF_CLASSIFICADOR  = 0.6
BUFFER_PRED         = 8
TEMPO_MINIMO_SALVAR = 2.0

DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "user":     "admin",
    "password": "1234",
    "dbname":   "postgres",
    "options":  "-c search_path=yolo_pose",
}

LABELS_PT = {
    "pegar_console":          "Pegando Console",
    "ler_etiqueta":           "Lendo Etiqueta",
    "girar_berco":            "Girando Berço",
    "montagem_parafusamento": "Montagem / Parafusamento",
    "inspecao_visual":        "Inspeção Visual",
    "demonstracao":           "Modo Demonstração",
}
CORES = {
    "pegar_console":          (0,   200,   0),
    "ler_etiqueta":           (255, 165,   0),
    "girar_berco":            (0,   165, 255),
    "montagem_parafusamento": (255,   0, 200),
    "inspecao_visual":        (0,   230, 230),
    "demonstracao":           (180, 180, 180),
}



# ──────────────────────────────────────────────────────────────
# UPSCALE PARA 1080p COM LETTERBOX
# ──────────────────────────────────────────────────────────────
def frame_para_1080p(frame):
    """
    Redimensiona o frame para 1920×1080 preservando a proporção original.
    O espaço restante é preenchido com preto (letterbox).
    Usado exclusivamente no modo --salvar.
    """
    h, w    = frame.shape[:2]
    scale   = min(SAIDA_LARGURA / w, SAIDA_ALTURA / h)
    novo_w  = int(w * scale)
    novo_h  = int(h * scale)
    redim   = cv2.resize(frame, (novo_w, novo_h), interpolation=cv2.INTER_LANCZOS4)

    canvas  = np.zeros((SAIDA_ALTURA, SAIDA_LARGURA, 3), dtype=np.uint8)
    x_off   = (SAIDA_LARGURA - novo_w) // 2
    y_off   = (SAIDA_ALTURA  - novo_h) // 2
    canvas[y_off:y_off+novo_h, x_off:x_off+novo_w] = redim
    return canvas


# ──────────────────────────────────────────────────────────────
# INSPEÇÃO DO CHECKPOINT — fonte de verdade para a arquitetura
# ──────────────────────────────────────────────────────────────
def inspecionar_checkpoint(caminho, device):
    """
    Lê os tensores salvos e infere os hiperparâmetros da arquitetura
    sem depender de nenhum arquivo externo.

    Regras de inferência:
      weight_ih_l0 : shape (4 * hidden_size, input_size)
      weight_ih_lN : existência indica número de camadas
      norm.weight  : shape (hidden_size,) — confirma hidden_size
      fc.weight    : shape (num_classes, hidden_size)
      fc.bias      : shape (num_classes,) — confirma num_classes
    """
    ckpt = torch.load(caminho, map_location=device)

    w0         = ckpt["lstm.weight_ih_l0"]   # (4*hidden, input)
    input_size  = w0.shape[1]
    hidden_size = w0.shape[0] // 4
    num_layers  = sum(1 for k in ckpt if k.startswith("lstm.weight_ih_l"))
    num_classes = ckpt["fc.bias"].shape[0]

    # dropout não é salvo nos pesos — usa 0 para carregar sem erros;
    # dropout só afeta o treino, não a inferência (model.eval() desativa)
    dropout = 0.0

    return ckpt, {
        "input_size":  input_size,
        "hidden_size": hidden_size,
        "num_layers":  num_layers,
        "num_classes": num_classes,
        "dropout":     dropout,
    }


# ──────────────────────────────────────────────────────────────
# ARQUITETURA LSTM — idêntica ao script 4
# ──────────────────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 num_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = 0,            # desativado na inferência
            bidirectional = False,
        )
        self.norm    = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lens=None):
        if lens is not None:
            # enforce_sorted=False — sem necessidade de ordenar o batch
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lens.cpu(), batch_first=True, enforce_sorted=False)
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(x)

        last = h_n[-1]
        return self.fc(self.dropout(self.norm(last)))


# ──────────────────────────────────────────────────────────────
# FEATURES — idêntica ao script 4 (ângulos + proporções invariantes)
# ──────────────────────────────────────────────────────────────
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
    """
    Extrai 13 features invariantes à posição e escala:
      10 ângulos articulares + 3 distâncias relativas à largura dos ombros.
    O argumento bbox é mantido por compatibilidade mas não é usado.
    """
    feats = [
        _angulo(kp[5],  kp[7],  kp[9]),   # 1.  Cotovelo Esquerdo
        _angulo(kp[6],  kp[8],  kp[10]),  # 2.  Cotovelo Direito
        _angulo(kp[11], kp[5],  kp[7]),   # 3.  Ombro Esquerdo (Axila)
        _angulo(kp[12], kp[6],  kp[8]),   # 4.  Ombro Direito (Axila)
        _angulo(kp[0],  kp[5],  kp[7]),   # 5.  Pescoço ao Ombro Esq
        _angulo(kp[0],  kp[6],  kp[8]),   # 6.  Pescoço ao Ombro Dir
        _angulo(kp[5],  kp[11], kp[13]),  # 7.  Anca Esquerda
        _angulo(kp[6],  kp[12], kp[14]),  # 8.  Anca Direita
        _angulo(kp[11], kp[13], kp[15]),  # 9.  Joelho Esquerdo
        _angulo(kp[12], kp[14], kp[16]),  # 10. Joelho Direito
    ]
    ombros_dist = _dist(kp[5], kp[6])
    if ombros_dist < 1e-5:
        ombros_dist = 1.0
    feats += [
        _dist(kp[9],  kp[10]) / ombros_dist,  # 11. Dist entre mãos
        _dist(kp[9],  kp[0])  / ombros_dist,  # 12. Mão esq ao rosto
        _dist(kp[10], kp[0])  / ombros_dist,  # 13. Mão dir ao rosto
    ]
    return np.array(feats, dtype=np.float32)


# ──────────────────────────────────────────────────────────────
# FILAS DESLIZANTES POR CLASSE
# ──────────────────────────────────────────────────────────────
class FilasPorClasse:
    """
    Uma deque independente por atividade, com maxlen = janela da classe.
    O mesmo vetor de features é adicionado a todas as filas a cada frame.
    """
    def __init__(self, janela_por_classe):
        self.janela_por_classe = janela_por_classe
        self.filas = {
            a: deque(maxlen=j) for a, j in janela_por_classe.items()
        }

    def adicionar(self, feats_norm):
        for fila in self.filas.values():
            fila.append(feats_norm)

    def pronta(self, atividade):
        fila = self.filas[atividade]
        return len(fila) == self.janela_por_classe[atividade]

    def alguma_pronta(self):
        return any(self.pronta(a) for a in self.janela_por_classe)

    def sequencia_padded(self, atividade, janela_max, input_size):
        """Retorna sequência com post-padding para janela_max (igual ao script 4)."""
        seq     = np.array(self.filas[atividade], dtype=np.float32)
        pad_len = janela_max - len(seq)
        if pad_len > 0:
            pad = np.zeros((pad_len, input_size), dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=0)
        return seq

    def progresso(self):
        """Frames acumulados na fila de maior janela (para barra de progresso)."""
        a_ref = max(self.janela_por_classe, key=self.janela_por_classe.get)
        return len(self.filas[a_ref]), self.janela_por_classe[a_ref]


# ──────────────────────────────────────────────────────────────
# CLASSIFICAÇÃO EM BATCH ÚNICO
# ──────────────────────────────────────────────────────────────
def classificar(lstm, le, filas, janela_max, input_size, device, conf_min):
    """
    Para cada classe com fila cheia, monta uma sequência padded e
    envia tudo em um único batch ao LSTM. Retorna a classe com maior
    probabilidade de ser ela mesma (autoconfiança), acima de conf_min.
    """
    prontas = [a for a in filas.janela_por_classe if filas.pronta(a)]
    if not prontas:
        return None, 0.0

    seqs = np.array([
        filas.sequencia_padded(a, janela_max, input_size) for a in prontas
    ], dtype=np.float32)                                   # (N, janela_max, input_size)

    lens = torch.tensor(
        [filas.janela_por_classe[a] for a in prontas], dtype=torch.long)

    batch  = torch.tensor(seqs).to(device)
    with torch.no_grad():
        logits = lstm(batch, lens)
        probas = torch.softmax(logits, dim=1).cpu().numpy()  # (N, num_classes)

    # Score de cada classe = probabilidade de ser ela mesma
    scores = {
        a: float(probas[i, le.transform([a])[0]])
        for i, a in enumerate(prontas)
    }

    melhor      = max(scores, key=scores.get)
    melhor_conf = scores[melhor]

    return (melhor, melhor_conf) if melhor_conf >= conf_min else (None, 0.0)


# ──────────────────────────────────────────────────────────────
# HARDWARE
# ──────────────────────────────────────────────────────────────
def configurar_gpu():
    print("\n🖥️  HARDWARE")
    print("-" * 45)
    if torch.cuda.is_available():
        nome = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  ✅ GPU  : {nome}")
        print(f"  ✅ VRAM : {vram:.1f} GB  |  FP16: Ativado")
        print("-" * 45 + "\n")
        return torch.device("cuda:0"), True
    print("  ⚠️  GPU não detectada — usando CPU.")
    print("-" * 45 + "\n")
    return torch.device("cpu"), False


# ──────────────────────────────────────────────────────────────
# CARREGAMENTO DOS MODELOS
# ──────────────────────────────────────────────────────────────
def carregar_modelos(device, fp16, usar_classificador):
    """
    Retorna todos os artefatos necessários para a inferência.

    Ordem de prioridade para os parâmetros:
      1. Arquitetura (input_size, hidden_size, num_layers, num_classes)
         → sempre lida do checkpoint (fonte de verdade absoluta)
      2. janela_por_classe
         → config.json se existir, senão janela única de fallback
      3. dropout
         → irrelevante na inferência (model.eval() desativa)
    """
    lstm              = None
    le                = None
    scaler            = None
    janela_por_classe = {}
    janela_max        = 30
    modo_demo         = True

    # ── YOLO-Pose ─────────────────────────────────────────────
    modelo_path = MODELO_BASE
    if usar_classificador and Path(MODELO_POSE).exists():
        modelo_path = MODELO_POSE
        print(f"  ✅ YOLO-Pose  : {MODELO_POSE} (treinado)")
    else:
        print(f"  ✅ YOLO-Pose  : {MODELO_BASE} (base COCO)")

    model_pose = YOLO(modelo_path)
    model_pose.to(device)

    # ── LSTM ──────────────────────────────────────────────────
    if not usar_classificador:
        print(f"  ℹ️  Modo demonstração — apenas estimativa de pose.")
        return model_pose, None, None, None, {}, 30, True

    arquivos_obrigatorios = [LSTM_WEIGHTS, LABEL_ENCODER, SCALER]
    faltando = [p for p in arquivos_obrigatorios if not Path(p).exists()]
    if faltando:
        print(f"  ⚠️  Arquivos ausentes: {faltando}")
        print(f"     Execute o script 4 para treinar o LSTM.")
        print(f"  ℹ️  Continuando em modo demonstração...")
        return model_pose, None, None, None, {}, 30, True

    # 1. Lê arquitetura diretamente do checkpoint
    print(f"  🔍 Lendo arquitetura do checkpoint...")
    ckpt, arq = inspecionar_checkpoint(LSTM_WEIGHTS, device)
    print(f"  ✅ Arquitetura : input={arq['input_size']}, "
          f"hidden={arq['hidden_size']}, "
          f"layers={arq['num_layers']}, "
          f"classes={arq['num_classes']}")

    # 2. Instancia o LSTM com os parâmetros reais do checkpoint
    lstm = LSTMClassifier(**arq).to(device)
    lstm.load_state_dict(ckpt)
    lstm.eval()

    # 3. Label encoder e scaler
    le     = joblib.load(LABEL_ENCODER)
    scaler = joblib.load(SCALER)

    # 4. Janelas por classe — config.json (opcional) ou fallback uniforme
    if Path(CONFIG_JSON).exists():
        with open(CONFIG_JSON) as f:
            cfg = json.load(f)
        if "janela_por_classe" in cfg:
            janela_por_classe = cfg["janela_por_classe"]
            janela_max        = cfg.get("janela_max", max(janela_por_classe.values()))
            print(f"  ✅ Janelas     : config.json ({len(janela_por_classe)} classes)")
        else:
            # config.json existe mas é formato antigo (janela_frames única)
            j = cfg.get("janela_frames", 30)
            janela_por_classe = {a: j for a in le.classes_}
            janela_max        = j
            print(f"  ✅ Janelas     : config.json legado → {j} frames para todas")
    else:
        # Sem config.json — usa janela uniforme de fallback
        janela_max        = 30
        janela_por_classe = {a: janela_max for a in le.classes_}
        print(f"  ⚠️  config.json não encontrado → janela uniforme de {janela_max} frames")
        print(f"     Para janelas por classe, execute o script 4 novamente.")

    print(f"  ✅ Janela máx  : {janela_max} frames")
    for a, j in janela_por_classe.items():
        print(f"     {a:30s} → {j} frames")

    modo_demo = False
    return model_pose, lstm, le, scaler, janela_por_classe, janela_max, modo_demo


# ──────────────────────────────────────────────────────────────
# BANCO DE DADOS
# ──────────────────────────────────────────────────────────────
def conectar_db():
    if not POSTGRES_OK:
        return None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = False
        _criar_tabela(conn)
        print("  ✅ PostgreSQL conectado")
        return conn
    except Exception as e:
        print(f"  ⚠️  Falha ao conectar: {e}")
        return None

def _criar_tabela(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS yolo_pose;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS yolo_pose.sessoes_atividade (
                id              SERIAL PRIMARY KEY,
                atividade       VARCHAR(60)    NOT NULL,
                atividade_label VARCHAR(80),
                inicio          TIMESTAMP      NOT NULL,
                fim             TIMESTAMP      NOT NULL,
                duracao_seg     NUMERIC(10,2)  NOT NULL,
                fonte           VARCHAR(200),
                confianca_media NUMERIC(5,2),
                classificador   VARCHAR(20)    DEFAULT 'lstm',
                criado_em       TIMESTAMP      DEFAULT NOW()
            );
        """)
        cur.execute("""
            CREATE OR REPLACE VIEW yolo_pose.resumo_atividades AS
            SELECT atividade, atividade_label,
                COUNT(*)                            AS total_ocorrencias,
                ROUND(AVG(duracao_seg)::numeric, 2) AS media_duracao_seg,
                ROUND(MIN(duracao_seg)::numeric, 2) AS min_duracao_seg,
                ROUND(MAX(duracao_seg)::numeric, 2) AS max_duracao_seg,
                ROUND(SUM(duracao_seg)::numeric, 2) AS total_duracao_seg
            FROM yolo_pose.sessoes_atividade
            GROUP BY atividade, atividade_label
            ORDER BY total_ocorrencias DESC;
        """)
        conn.commit()

def salvar_sessao(conn, atividade, inicio, fim, fonte, conf_media):
    if conn is None:
        return
    duracao = (fim - inicio).total_seconds()
    if duracao < TEMPO_MINIMO_SALVAR:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO yolo_pose.sessoes_atividade
                    (atividade, atividade_label, inicio, fim,
                     duracao_seg, fonte, confianca_media, classificador)
                VALUES (%s,%s,%s,%s,%s,%s,%s,'lstm') RETURNING id;
            """, (
                atividade, LABELS_PT.get(atividade, atividade),
                inicio, fim, round(duracao, 2),
                str(fonte), round(conf_media * 100, 2),
            ))
            novo_id = cur.fetchone()[0]
            conn.commit()
            print(f"\n  💾 id={novo_id} | {LABELS_PT.get(atividade,atividade)} "
                  f"| {duracao:.1f}s | conf {conf_media:.0%}")
    except Exception as e:
        conn.rollback()
        print(f"  ⚠️  Erro ao salvar: {e}")


# ──────────────────────────────────────────────────────────────
# TEMPORIZADOR
# ──────────────────────────────────────────────────────────────
class Temporizador:
    def __init__(self):
        self.atividade_atual = None
        self.inicio          = None
        self.confianças      = []

    def atualizar(self, nova, confianca, conn, fonte):
        agora = datetime.now()
        if nova != self.atividade_atual:
            self._salvar(agora, conn, fonte)
            self.atividade_atual = nova
            self.inicio          = agora
            self.confianças      = [confianca]
        else:
            self.confianças.append(confianca)
        return 0.0 if self.inicio is None else (agora - self.inicio).total_seconds()

    def _salvar(self, fim, conn, fonte):
        if self.atividade_atual is None or self.inicio is None:
            return
        conf = float(np.mean(self.confianças)) if self.confianças else 0.0
        salvar_sessao(conn, self.atividade_atual, self.inicio, fim, fonte, conf)

    def finalizar(self, conn, fonte):
        self._salvar(datetime.now(), conn, fonte)


# ──────────────────────────────────────────────────────────────
# OVERLAY
# ──────────────────────────────────────────────────────────────

# Carrega uma fonte TTF com suporte a UTF-8 (ç, ã, etc.)
# Tenta fontes comuns do sistema; usa a fonte padrão do PIL como fallback.
def _carregar_fonte(tamanho):
    candidatas = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",           # macOS
        "C:/Windows/Fonts/arial.ttf",                    # Windows
    ]
    for caminho in candidatas:
        if Path(caminho).exists():
            try:
                return ImageFont.truetype(caminho, tamanho)
            except Exception:
                pass
    return ImageFont.load_default()

# Pré-carrega as fontes uma única vez
_FONTE_GRANDE  = _carregar_fonte(26)
_FONTE_MEDIA   = _carregar_fonte(19)
_FONTE_PEQUENA = _carregar_fonte(13)


def _puttext_pil(frame, texto, posicao, fonte, cor_rgb):
    """Renderiza `texto` (UTF-8) no `frame` OpenCV usando PIL."""
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    # cor PIL usa RGB; OpenCV usa BGR — converter
    draw.text(posicao, texto, font=fonte, fill=cor_rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def _bgr2rgb(bgr):
    """Converte tupla BGR (OpenCV) → RGB (PIL)."""
    return (bgr[2], bgr[1], bgr[0])


def desenhar_overlay(frame, atividade, confianca, tempo_seg,
                     fps, modo_demo, gpu_info, prog_atual, prog_max):
    h, w = frame.shape[:2]

    # ── Painel branco semi-transparente na parte inferior ──────
    ALTURA_PAINEL = 130
    y0 = h - ALTURA_PAINEL

    ov = frame.copy()
    cv2.rectangle(ov, (0, y0), (w, h), (255, 255, 255), -1)
    cv2.addWeighted(ov, 0.82, frame, 0.18, 0, frame)

    # Linha separadora sutil no topo do painel
    cv2.line(frame, (0, y0), (w, y0), (200, 200, 200), 1)

    cor_bgr = CORES.get(atividade, (100, 100, 100)) if atividade else (100, 100, 100)
    cor_rgb = _bgr2rgb(cor_bgr)
    cinza_escuro = (60, 60, 60)
    cinza_medio  = (110, 110, 110)

    if atividade:
        label = LABELS_PT.get(atividade, atividade)

        # Linha 1 — Atividade (fonte grande, cor da classe)
        frame = _puttext_pil(frame, f"Atividade : {label}",
                             (12, y0 + 8), _FONTE_GRANDE, cor_rgb)

        # Linha 2 — Confiança + FPS (fonte média, cinza escuro)
        frame = _puttext_pil(frame, f"Confiança : {confianca:.0%}   |   FPS: {fps:.1f}",
                             (12, y0 + 40), _FONTE_MEDIA, cinza_escuro)

        # Linha 3 — Duração (fonte média, cor da classe)
        mins  = int(tempo_seg) // 60
        segs  = int(tempo_seg) % 60
        cents = int((tempo_seg % 1) * 10)
        frame = _puttext_pil(frame, f"Duração   : {mins:02d}:{segs:02d}.{cents}",
                             (12, y0 + 64), _FONTE_MEDIA, cor_rgb)

        # Barra de confiança (canto direito do painel)
        bw, bh = 200, 14
        bx = w - bw - 12
        by = y0 + 20
        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (210, 210, 210), -1)
        cv2.rectangle(frame, (bx, by), (bx+int(bw*confianca), by+bh), cor_bgr, -1)
        frame = _puttext_pil(frame, "confiança",
                             (bx, by - 16), _FONTE_PEQUENA, cinza_medio)

    else:
        # Estado inicial — coletando sequência
        pct = prog_atual / max(prog_max, 1)
        frame = _puttext_pil(frame,
                             f"Coletando sequência: {prog_atual}/{prog_max} frames...",
                             (12, y0 + 20), _FONTE_MEDIA, cinza_escuro)
        cv2.rectangle(frame, (12, y0 + 50), (412, y0 + 64), (210, 210, 210), -1)
        cv2.rectangle(frame, (12, y0 + 50), (12+int(400*pct), y0 + 64), (100, 180, 100), -1)

    # Rodapé — GPU / modo / FPS (linha inferior do painel)
    modo_txt = "LSTM ativo" if not modo_demo else "DEMO"
    frame = _puttext_pil(frame, f"{gpu_info}   |   {modo_txt}   |   FPS: {fps:.1f}",
                         (12, y0 + 108), _FONTE_PEQUENA, cinza_medio)

    return frame


# ──────────────────────────────────────────────────────────────
# LOOP PRINCIPAL
# ──────────────────────────────────────────────────────────────
def inferir(source, usar_classificador, salvar=False, caminho_saida=None,
            camera_idx=0):
    print("=" * 55)
    print("  INFERÊNCIA EM TEMPO REAL — LSTM DINÂMICO")
    print("=" * 55)

    device, fp16 = configurar_gpu()

    print("\n📦 CARREGANDO MODELOS...")
    print("-" * 45)
    (model_pose, lstm, le, scaler,
     janela_por_classe, janela_max, modo_demo) = carregar_modelos(
        device, fp16, usar_classificador)

    print("\n🗄️  BANCO DE DADOS...")
    print("-" * 45)
    conn = conectar_db()

    # ── Resolução da fonte ────────────────────────────────────
    # --source tem prioridade (arquivo de vídeo ou índice explícito)
    # --camera é o atalho específico para webcam USB
    if str(source) != "0":
        # Arquivo de vídeo ou índice passado via --source
        src = int(source) if str(source).isdigit() else source
    else:
        # Webcam: usa o índice definido por --camera (padrão 0)
        src = camera_idx

    # No Windows usa DirectShow para melhor compatibilidade com webcams USB.
    # No Linux usa o backend padrão (V4L2).
    import platform
    if isinstance(src, int):
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(src)
    else:
        cap = cv2.VideoCapture(src)   # arquivo de vídeo — sem backend específico

    if not cap.isOpened():
        print(f"\n❌ Não foi possível abrir a fonte: {src}")
        if isinstance(src, int):
            print(f"   Dica: execute com --listar-cameras para ver os índices disponíveis.")
        return

    # Configura resolução preferida para webcams (ignorado para arquivos)
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)   # ativa autofoco se suportado

    if src == 0:
        fonte_nome = "webcam_0"
    elif isinstance(src, int):
        fonte_nome = f"webcam_{src}"
    else:
        fonte_nome = str(source)
    largura    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\n  📹 Fonte  : {fonte_nome}  ({largura}×{altura})")

    # ── VideoWriter (modo --salvar) ────────────────────────────
    writer = None
    if salvar:
        if caminho_saida is None:
            ts            = datetime.now().strftime("%Y%m%d_%H%M%S")
            caminho_saida = f"inferencia_{ts}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            caminho_saida, fourcc, SAIDA_FPS,
            (SAIDA_LARGURA, SAIDA_ALTURA)
        )
        if not writer.isOpened():
            print(f"  ⚠️  Não foi possível criar o ficheiro de saída: {caminho_saida}")
            writer = None
        else:
            print(f"  💾 Gravando em  : {caminho_saida}  ({SAIDA_LARGURA}×{SAIDA_ALTURA} @ {SAIDA_FPS}fps)")
            print(f"  ℹ️  Janela de visualização desativada no modo --salvar")
    else:
        print("\n  Pressione  Q  para encerrar\n")

    # input_size inferido do checkpoint — weight_ih_l0 tem shape (4*hidden, input_size)
    input_size = next(lstm.lstm.parameters()).shape[1] if lstm else 13

    filas          = FilasPorClasse(janela_por_classe) if not modo_demo else None
    buf_pred       = deque(maxlen=BUFFER_PRED)
    buf_conf       = deque(maxlen=BUFFER_PRED)
    temporizador   = Temporizador()
    atividade_suav = None   # última predição válida — persiste entre frames
    confianca_suav = 0.0

    t0       = time.time()
    n_frames = 0
    gpu_info = (torch.cuda.get_device_name(0).replace("NVIDIA GeForce ", "")
                if torch.cuda.is_available() else "CPU")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\n  ⏹  Fim do vídeo ou câmera desconectada.")
            break

        n_frames += 1
        fps = n_frames / max(time.time() - t0, 1e-6)

        results   = model_pose(frame, verbose=False, conf=CONF_POSE, half=fp16)
        annotated = results[0].plot(
            kpt_radius  = 6,      # keypoints maiores e mais visíveis
            line_width  = 3,      # linhas do esqueleto mais espessas
            boxes       = False,  # remove o bounding box — menos poluição visual
            kpt_line    = True,   # liga os keypoints com linhas (esqueleto)
        )

        atividade_frame = None
        confianca_frame = 0.0

        for r in results:
            if r.keypoints is None or r.boxes is None:
                continue
            kps       = r.keypoints.data.cpu().numpy()
            bbs       = r.boxes.xyxy.cpu().numpy()
            confs_det = r.boxes.conf.cpu().numpy()
            if len(bbs) == 0:
                continue

            idx   = int(np.argmax(confs_det))
            feats = extrair_features(kps[idx], bbs[idx])
            if feats is None:
                continue

            if modo_demo:
                atividade_frame = "demonstracao"
                confianca_frame = float(confs_det[idx])
            else:
                feats_norm = scaler.transform([feats])[0]
                filas.adicionar(feats_norm)

                if filas.alguma_pronta():
                    atividade_frame, confianca_frame = classificar(
                        lstm, le, filas, janela_max,
                        input_size, device, CONF_CLASSIFICADOR)

        if atividade_frame:
            buf_pred.append(atividade_frame)
            buf_conf.append(confianca_frame)

        # Mantém a última predição válida mesmo em frames sem detecção.
        # O overlay só some se o buffer inteiro estiver vazio (início da sessão).
        if buf_pred:
            votos = Counter(buf_pred)
            melhor, contagem = votos.most_common(1)[0]
            if contagem / len(buf_pred) >= 0.70:
                atividade_suav = melhor
                confianca_suav = float(np.mean(buf_conf))
        # se não atingir 70%, mantém a predição anterior

        tempo_seg = 0.0
        if atividade_suav:
            tempo_seg = temporizador.atualizar(
                atividade_suav, confianca_suav, conn, fonte_nome)

        prog_atual, prog_max = (filas.progresso() if filas else (0, 1))

        annotated = desenhar_overlay(
            annotated, atividade_suav, confianca_suav,
            tempo_seg, fps, modo_demo, gpu_info, prog_atual, prog_max)

        # ── Saída: gravar ou exibir ────────────────────────────
        if salvar:
            # Upscale para 1080p com letterbox e grava no ficheiro
            writer.write(frame_para_1080p(annotated))
        else:
            # Redimensiona para caber no ecrã ao usar ficheiro de vídeo.
            # Webcams (qualquer índice) usam a resolução nativa sem redimensionamento.
            if not isinstance(src, int):
                h_ann, w_ann = annotated.shape[:2]
                max_w, max_h = 1280, 720
                scale = min(max_w / w_ann, max_h / h_ann, 1.0)
                if scale < 1.0:
                    novo_w = int(w_ann * scale)
                    novo_h = int(h_ann * scale)
                    annotated = cv2.resize(annotated, (novo_w, novo_h),
                                           interpolation=cv2.INTER_AREA)
            cv2.imshow("YOLO-Pose + LSTM", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n  ⏹  Encerrado pelo usuário.")
                break

    temporizador.finalizar(conn, fonte_nome)
    cap.release()
    if writer:
        writer.release()
        print(f"\n  ✅ Vídeo salvo   : {caminho_saida}")
    if not salvar:
        cv2.destroyAllWindows()
    if conn:
        conn.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n  ✅ Frames processados : {n_frames}")
    print(f"  ✅ FPS médio          : {n_frames / max(time.time()-t0, 1e-6):.1f}")


# ──────────────────────────────────────────────────────────────
# EXECUÇÃO
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="YOLO-Pose + LSTM — Inferência em Tempo Real"
    )
    parser.add_argument(
        "--source", default=None,
        help="Caminho de um arquivo de vídeo. Para webcam use --camera."
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Índice da webcam a usar (default: 0). "
             "Use --listar-cameras para descobrir o índice da USB."
    )
    parser.add_argument(
        "--listar-cameras", action="store_true",
        help="Lista todas as câmeras disponíveis e encerra."
    )
    parser.add_argument(
        "--salvar", action="store_true",
        help="Grava o resultado em vídeo 1080p em vez de exibir ao vivo"
    )
    parser.add_argument(
        "--saida", default=None,
        help="Caminho do ficheiro de saída (default: inferencia_YYYYMMDD_HHMMSS.mp4)"
    )
    parser.add_argument(
        "--sem-classificador", action="store_true",
        help="Modo demonstração — só pose, sem LSTM"
    )
    args = parser.parse_args()

    if args.listar_cameras:
        listar_cameras()
    else:
        # Se --source foi passado, usa-o (arquivo ou índice explícito).
        # Caso contrário, usa o índice de --camera (webcam).
        if args.source is not None:
            source = int(args.source) if args.source.isdigit() else args.source
        else:
            source = 0   # sinaliza "usar camera_idx"

        inferir(
            source             = source,
            usar_classificador = not args.sem_classificador,
            salvar             = args.salvar,
            caminho_saida      = args.saida,
            camera_idx         = args.camera,
        )
