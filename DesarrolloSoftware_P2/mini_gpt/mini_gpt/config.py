from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent
CARPETA_CORPUS = BASE_DIR / "corpus"
CARPETA_DATOS = BASE_DIR / "datos"
CARPETA_MODELOS = BASE_DIR / "modelos"
CARPETA_RESULTADOS = BASE_DIR / "resultados"

ARCHIVO_CORPUS_LIMPIO = CARPETA_DATOS / "corpus_limpio.txt"
ARCHIVO_VOCABULARIO = CARPETA_DATOS / "vocabulario.json"
ARCHIVO_TOKENS = CARPETA_DATOS / "tokens.pt"
ARCHIVO_MODELO = CARPETA_MODELOS / "minigpt.pt"
ARCHIVO_HISTORIAL = CARPETA_RESULTADOS / "historial_entrenamiento.json"

TOKEN_DESCONOCIDO = "<UNK>"
TOKEN_PADDING = "<PAD>"
TOKEN_INICIO = "<BOS>"
TOKEN_FIN = "<EOS>"
FRECUENCIA_MINIMA = 1

# --- ARQUITECTURA Y MEMORIA ---
LONGITUD_CONTEXTO = 256      # Ampliado para no olvidar el prompt y la respuesta inicial
TAMANIO_BATCH = 16           
PORCENTAJE_ENTRENAMIENTO = 0.90

DIMENSION_EMBEDDING = 128
NUM_CABEZAS = 4
NUM_CAPAS = 3
DIMENSION_FEED_FORWARD = 512
DROPOUT = 0.10

# --- ENTRENAMIENTO ---
EPOCAS = 40                 
TASA_APRENDIZAJE = 3e-4
MOSTRAR_CADA = 20
SEMILLA = 42

# --- INFERENCIA / GENERACIÓN ---
MAX_TOKENS_GENERADOS = 120
TEMPERATURA = 0.4            # Ligeramente ajustado para naturalidad
TOP_K = 15                   

if torch.cuda.is_available():
    DISPOSITIVO = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DISPOSITIVO = "mps"
else:
    DISPOSITIVO = "cpu"