import torch
from config import ARCHIVO_TOKENS, LONGITUD_CONTEXTO, TAMANIO_BATCH, PORCENTAJE_ENTRENAMIENTO, DISPOSITIVO

class GestorDataset:
    def __init__(self):
        if not ARCHIVO_TOKENS.exists():
            raise RuntimeError("No existe datos/tokens.pt. Ejecuta tokenizador.py")
        self.tokens = torch.load(ARCHIVO_TOKENS, map_location="cpu")
        corte = int(len(self.tokens) * PORCENTAJE_ENTRENAMIENTO)
        self.train = self.tokens[:corte]
        self.val = self.tokens[corte:]
        if len(self.train) <= LONGITUD_CONTEXTO + 1:
            raise RuntimeError("Corpus demasiado pequeño para LONGITUD_CONTEXTO")

    def obtener_batch(self, division="train"):
        datos = self.train if division == "train" else self.val
        if len(datos) <= LONGITUD_CONTEXTO + 1:
            datos = self.train
        max_inicio = len(datos) - LONGITUD_CONTEXTO - 1
        indices = torch.randint(0, max_inicio, (TAMANIO_BATCH,))
        x = torch.stack([datos[i:i+LONGITUD_CONTEXTO] for i in indices])
        y = torch.stack([datos[i+1:i+LONGITUD_CONTEXTO+1] for i in indices])
        return x.to(DISPOSITIVO), y.to(DISPOSITIVO)
