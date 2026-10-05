import re
from collections import Counter
import torch
from config import ARCHIVO_CORPUS_LIMPIO, ARCHIVO_VOCABULARIO, ARCHIVO_TOKENS, TOKEN_DESCONOCIDO, TOKEN_PADDING, TOKEN_INICIO, TOKEN_FIN, FRECUENCIA_MINIMA
from utilidades import guardar_json, cargar_json

PATRON_TOKEN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
TOKENS_ESPECIALES = [TOKEN_PADDING, TOKEN_DESCONOCIDO, TOKEN_INICIO, TOKEN_FIN]

class TokenizadorEducativo:
    def __init__(self):
        self.token_a_id = {}
        self.id_a_token = {}

    def separar(self, texto):
        return PATRON_TOKEN.findall(texto.lower())

    def entrenar(self, texto):
        tokens = self.separar(texto)
        frecuencias = Counter(tokens)
        vocabulario = list(TOKENS_ESPECIALES)
        for token, frecuencia in sorted(frecuencias.items(), key=lambda x: (-x[1], x[0])):
            if frecuencia >= FRECUENCIA_MINIMA and token not in vocabulario:
                vocabulario.append(token)
        self.token_a_id = {token: i for i, token in enumerate(vocabulario)}
        self.id_a_token = {i: token for token, i in self.token_a_id.items()}
        return frecuencias

    def codificar(self, texto, agregar_inicio_fin=True):
        if not self.token_a_id:
            raise RuntimeError("El tokenizador aún no tiene vocabulario")
        tokens = self.separar(texto)
        unk = self.token_a_id[TOKEN_DESCONOCIDO]
        ids = [self.token_a_id.get(token, unk) for token in tokens]
        if agregar_inicio_fin:
            ids = [self.token_a_id[TOKEN_INICIO]] + ids + [self.token_a_id[TOKEN_FIN]]
        return ids

    def decodificar(self, ids, omitir_especiales=True):
        especiales = set(TOKENS_ESPECIALES)
        tokens = []
        for idx in ids:
            token = self.id_a_token.get(int(idx), TOKEN_DESCONOCIDO)
            if omitir_especiales and token in especiales:
                continue
            tokens.append(token)
        texto = " ".join(tokens)
        texto = re.sub(r"\s+([,.;:!?])", r"\1", texto)
        return texto

    def guardar(self, ruta=ARCHIVO_VOCABULARIO):
        guardar_json(self.token_a_id, ruta)

    def cargar(self, ruta=ARCHIVO_VOCABULARIO):
        datos = cargar_json(ruta)
        self.token_a_id = {k: int(v) for k, v in datos.items()}
        self.id_a_token = {v: k for k, v in self.token_a_id.items()}

    @property
    def tamanio_vocabulario(self):
        return len(self.token_a_id)

def entrenar_tokenizador_y_tokenizar():
    if not ARCHIVO_CORPUS_LIMPIO.exists():
        raise RuntimeError("Primero ejecuta preparar_corpus.py")
    texto = ARCHIVO_CORPUS_LIMPIO.read_text(encoding="utf-8")
    tokenizador = TokenizadorEducativo()
    frecuencias = tokenizador.entrenar(texto)
    tokenizador.guardar()
    ids = tokenizador.codificar(texto, agregar_inicio_fin=False)
    torch.save(torch.tensor(ids, dtype=torch.long), ARCHIVO_TOKENS)
    print(f"Vocabulario: {tokenizador.tamanio_vocabulario:,} tokens")
    print(f"Corpus tokenizado: {len(ids):,} IDs")
    print("10 tokens más frecuentes:")
    for token, frecuencia in frecuencias.most_common(10):
        print(f"  {token!r}: {frecuencia}")
    return tokenizador

if __name__ == "__main__":
    entrenar_tokenizador_y_tokenizar()
