# 🚀 [Pose Estimation - FMM]

Este projeto utiliza o modelo de visão computacional YOLOv8m-pose e o modelo de classificação LSTM, para identificar as atividades que estão sendo realizadas pelo operador numa linha de produção e temporalizá-las.

---

## 🛠️ Configuração e Execução (Setup)

Após baixar e extrair o arquivo `.zip` do GitHub, você precisará preparar o ambiente. Agrupamos todos os comandos necessários para que você possa executá-los em sequência no seu terminal.

### Pré-requisitos
* **Python 3.8+** instalado.

### Comandos de Instalação e Execução

Abra o seu terminal, substitua o caminho da pasta e rode o bloco de comandos correspondente ao seu sistema operacional:

**➡️ Para Windows (Prompt de Comando):**
```bash
:: 1. Navegue até a pasta extraída
cd caminho\para\a\pasta\do\projeto

:: 2. Crie o ambiente virtual
python -m venv venv

:: 3. Ative o ambiente virtual
venv\Scripts\activate.bat

:: 4. Instale as dependências
pip install -r requirements.txt

:: 5. Execute o script principal
python scripts\main.py
```

**➡️ Para Linux/MacOS (Prompt de Comando):**

```bash
# 1. Navegue até a pasta extraída
cd caminho/para/a/pasta/do/projeto

# 2. Crie o ambiente virtual
python3 -m venv venv

# 3. Ative o ambiente virtual
source venv/bin/activate

# 4. Instale as dependências
pip install -r requirements.txt

# 5. Execute o script principal
python scripts/main.py
```

## Após a instalação, utilize o arquivo README.md que está na pasta dos Scripts para prosseguir com a utilização.