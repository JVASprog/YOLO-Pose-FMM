# Pipeline YOLO-Pose — Classificação de Atividades (GPU NVIDIA / Windows)

## 1. Instalação

### Passo 1 — Instale o driver NVIDIA
Baixe o driver mais recente para sua placa:
https://www.nvidia.com/Download/index.aspx

### Passo 2 — Instale PyTorch com CUDA
Abra o Prompt de Comando e execute:

```bash
# Desinstale versão anterior (se houver)
pip uninstall torch torchvision torchaudio -y

# Instale com CUDA 12.1 (recomendado para RTX 20xx, 30xx, 40xx)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Para python 3.12+ CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Se sua GPU for antiga (GTX 10xx, 16xx) use CUDA 11.8:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Passo 3 — Instale as demais dependências
```bash
pip install ultralytics opencv-python scikit-learn joblib matplotlib seaborn pyyaml psycopg2-binary
```

---

## 2. Verifique o ambiente antes de começar
```bash
python Scripts/0_diagnostico.py
```
Este script vai confirmar se sua GPU está sendo detectada corretamente.

---

## 3. Estrutura de pastas esperada
```
Angulo/
  superior/
    pegar_console/          (6 vídeos)
    ler_etiqueta/           (6 vídeos)
    girar_berco/            (6 vídeos)
    montagem_parafusamento/ (6 vídeos)
    inspecao_visual/        (6 vídeos)
  perspectiva/ (mesma estrutura)
```
> ℹ️ Pastas ausentes são ignoradas automaticamente. Quando o ângulo
> paralelo estiver disponível, basta adicioná-lo à lista `ANGULOS`
> nos scripts 1, 2 e 4 e rodar novamente.

---

## 4. Ordem de execução
```bash
python Scripts/0_diagnostico.py           # Verifica GPU e dependências
python Scripts/1_extrair_e_split.py       # Extrai frames (4 treino / 1 val / 1 teste)
python Scripts/2_anotar_keypoints.py      # Anotação automática de keypoints na GPU
python Scripts/3_treinar_yolo_pose.py     # Treina YOLO-Pose (batch ajustado pela VRAM)
python Scripts/4_treinar_classificador_v1.py # Treina LSTM + avaliação por ângulo

# Antes de treinar — testar só estimativa de pose na webcam:
python Scripts/5_inferencia_withsaveAndCam_v2.py --sem-classificador

# Usar um vídeo em vez da webcam:
python Scripts/5_inferencia_withsaveAndCam_v2.py --source caminho/do/video.mp4

# Vídeo sem classificador (só pose):
python Scripts/5_inferencia_withsaveAndCam_v2.py --source caminho/do/video.mp4 --sem-classificador

# Com POSE e CLASSIFICADOR treinados (apontar os scripts "best.pt" e "classificador.pkl"):
python Scripts/5_inferencia_withsaveAndCam_v2.py                     # webcam com classificação
python Scripts/5_inferencia_withsaveAndCam_v2.py --source caminho/do/video.mp4  # vídeo com classificação
python Scripts/5_inferencia_withsaveAndCam_v2.py --source caminho/do/video.mp4 --salvar # nome gerado automaticamente
python Scripts/5_inferencia_withsaveAndCam_v2.py --source caminho/do/video.mp4 --salvar --saida resultado.mp4 # salva com o nome escolhido

# Para utilização de Câmera USB:
# 1. Primeiro descobre qual índice é a USB
python Scripts/5_inferencia_withsaveAndCam_v2.py --listar-cameras

# Exemplo de output:
#   ✅ Índice 0  →  1280×720  @ 30fps   ← câmera integrada
#   ✅ Índice 2  →  1920×1080 @ 30fps   ← webcam USB

# 2. Usa o índice correto
python Scripts/5_inferencia_withsaveAndCam_v2.py --camera índice

# Com gravação
python Scripts/5_inferencia_withsaveAndCam_v2.py --camera índice --salvar --saida resultado.mp4
```

---

## 5. Banco de dados PostgreSQL
O script 5 salva automaticamente cada atividade na tabela:
```
yolo_pose.sessoes_atividade
  id              — identificador único
  atividade       — nome interno        (ex: girar_berco)
  atividade_label — nome legível        (ex: Girando Berço)
  inicio          — timestamp de início
  fim             — timestamp de fim
  duracao_seg     — duração em segundos
  fonte           — webcam ou caminho do vídeo
  confianca_media — confiança média durante a atividade (%)
  criado_em       — quando foi inserido

-- View de resumo:
SELECT * FROM yolo_pose.resumo_atividades;
```
Configuração usada: host=localhost, port=5432, user=admin, db=postgres, schema=yolo_pose

---

## 6. O que foi otimizado para GPU
| Script | Otimização |
|--------|-----------|
| 2, 3, 4, 5 | Detecção automática da GPU NVIDIA |
| 2, 4, 5 | FP16 (half precision) ativado → ~2× mais rápido |
| 3 | Batch size ajustado automaticamente pela VRAM disponível |
| 3 | Cache de imagens na RAM para treino mais rápido |
| 5 | Exibe uso de VRAM, FPS e cronômetro em tempo real na tela |
| Todos | `torch.cuda.empty_cache()` ao encerrar |
