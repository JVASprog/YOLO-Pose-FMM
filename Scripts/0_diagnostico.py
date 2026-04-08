"""
DIAGNÓSTICO — Verifica GPU, CUDA e dependências
================================================
Execute este script ANTES de qualquer outro.
Ele vai dizer exatamente o que está instalado
e o que precisa ser instalado.

Uso:
  python 0_diagnostico.py
"""

import subprocess
import sys


def checar_gpu_nvidia():
    print("\n📌 VERIFICANDO GPU NVIDIA...")
    try:
        resultado = subprocess.run(
            ["nvidia-smi"],
            capture_output=True, text=True
        )
        if resultado.returncode == 0:
            # Pega as linhas relevantes do nvidia-smi
            linhas = resultado.stdout.splitlines()
            for linha in linhas:
                if any(k in linha for k in ["Driver", "CUDA Version", "|", "+"]):
                    print(" ", linha)
            print("\n  ✅ Driver NVIDIA detectado!")
            return True
        else:
            print("  ❌ nvidia-smi não encontrado.")
            print("     → Instale o driver NVIDIA: https://www.nvidia.com/drivers")
            return False
    except FileNotFoundError:
        print("  ❌ nvidia-smi não encontrado no PATH.")
        print("     → Instale o driver NVIDIA: https://www.nvidia.com/drivers")
        return False


def checar_cuda_pytorch():
    print("\n📌 VERIFICANDO PyTorch + CUDA...")
    try:
        import torch
        print(f"  PyTorch versão : {torch.__version__}")
        print(f"  CUDA disponível: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"  CUDA versão    : {torch.version.cuda}")
            print(f"  GPU detectada  : {torch.cuda.get_device_name(0)}")
            mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  VRAM total     : {mem:.1f} GB")
            print("\n  ✅ PyTorch com CUDA funcionando!")
            return True
        else:
            print("\n  ⚠️  PyTorch instalado, mas SEM suporte a CUDA.")
            print("     → Você tem PyTorch CPU-only instalado.")
            print("     → Siga as instruções abaixo para reinstalar com CUDA.")
            return False

    except ImportError:
        print("  ❌ PyTorch não instalado.")
        return False


def checar_ultralytics():
    print("\n📌 VERIFICANDO ULTRALYTICS (YOLO)...")
    try:
        import ultralytics
        print(f"  ✅ Ultralytics versão: {ultralytics.__version__}")
        return True
    except ImportError:
        print("  ❌ Ultralytics não instalado.")
        return False


def checar_outras():
    print("\n📌 VERIFICANDO OUTRAS DEPENDÊNCIAS...")
    pacotes = {
        "cv2":        "opencv-python",
        "sklearn":    "scikit-learn",
        "joblib":     "joblib",
        "matplotlib": "matplotlib",
        "seaborn":    "seaborn",
        "yaml":       "pyyaml",
        "numpy":      "numpy",
    }
    faltando = []
    for modulo, pacote in pacotes.items():
        try:
            __import__(modulo)
            print(f"  ✅ {pacote}")
        except ImportError:
            print(f"  ❌ {pacote}  ← FALTANDO")
            faltando.append(pacote)
    return faltando


def recomendar_instalacao(tem_driver, tem_cuda):
    print("\n" + "=" * 60)
    print("  INSTRUÇÕES DE INSTALAÇÃO")
    print("=" * 60)

    if not tem_driver:
        print("""
  1. INSTALE O DRIVER NVIDIA:
     → https://www.nvidia.com/Download/index.aspx
     Escolha sua placa e baixe o driver mais recente.
""")

    if not tem_cuda:
        print("""
  2. INSTALE PyTorch COM CUDA:

     Primeiro DESINSTALE o PyTorch atual (se houver):
       pip uninstall torch torchvision torchaudio -y

     Depois instale a versão com CUDA 12.1:
       pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

     (Se sua GPU for antiga e só suportar CUDA 11.8):
       pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

  3. INSTALE AS DEMAIS DEPENDÊNCIAS:
       pip install ultralytics opencv-python scikit-learn joblib matplotlib seaborn pyyaml
""")
    else:
        print("""
  ✅ Ambiente pronto! Você pode rodar os scripts na ordem:
     python 1_extrair_e_split.py
     python 2_anotar_keypoints.py
     python 3_treinar_yolo_pose.py
     python 4_treinar_classificador.py
     python 5_inferencia.py
""")


def main():
    print("=" * 60)
    print("  DIAGNÓSTICO DO AMBIENTE — YOLO-POSE GPU")
    print("=" * 60)

    tem_driver = checar_gpu_nvidia()
    tem_cuda   = checar_cuda_pytorch()
    checar_ultralytics()
    faltando   = checar_outras()

    print("\n" + "=" * 60)
    print("  RESUMO")
    print("=" * 60)
    print(f"  Driver NVIDIA  : {'✅' if tem_driver else '❌'}")
    print(f"  PyTorch + CUDA : {'✅' if tem_cuda   else '❌'}")
    print(f"  Pacotes faltando: {faltando if faltando else 'nenhum ✅'}")

    recomendar_instalacao(tem_driver, tem_cuda)


if __name__ == "__main__":
    main()
