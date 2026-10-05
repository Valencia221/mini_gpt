import torch
import torch.nn.functional as F

from config import (
    ARCHIVO_MODELO,
    LONGITUD_CONTEXTO,
    MAX_TOKENS_GENERADOS,
    TEMPERATURA,
    TOP_K,
    DISPOSITIVO
)

from modelo import MiniGPT
from tokenizador import TokenizadorEducativo


def cargar_modelo():
    tokenizador = TokenizadorEducativo()
    tokenizador.cargar()

    checkpoint = torch.load(
        ARCHIVO_MODELO,
        map_location=DISPOSITIVO
    )

    modelo = MiniGPT(
        checkpoint["tamanio_vocabulario"]
    ).to(DISPOSITIVO)

    modelo.load_state_dict(
        checkpoint["estado_modelo"]
    )

    modelo.eval()

    return modelo, tokenizador


@torch.no_grad()
def generar_texto(
    modelo,
    tokenizador,
    prompt,
    max_tokens=MAX_TOKENS_GENERADOS,
    temperatura=TEMPERATURA,
    top_k=TOP_K
):

    ids = tokenizador.codificar(
        prompt,
        agregar_inicio_fin=False
    )

    if not ids:
        ids = [tokenizador.token_a_id.get("<BOS>", 1)]

    secuencia = torch.tensor(
        [ids],
        dtype=torch.long,
        device=DISPOSITIVO
    )

    id_eos = tokenizador.token_a_id.get("<EOS>", -1)

    for _ in range(max_tokens):
        contexto = secuencia[:, -LONGITUD_CONTEXTO:]
        logits, _ = modelo(contexto)
        logits = logits[:, -1, :] / max(float(temperatura), 1e-5)

        if top_k and top_k > 0:
            k = min(top_k, logits.size(-1))
            valores, indices = torch.topk(logits, k)
            filtrados = torch.full_like(logits, float("-inf"))
            filtrados.scatter_(1, indices, valores)
            logits = filtrados

        probs = F.softmax(logits, dim=-1)
        siguiente = torch.multinomial(probs, num_samples=1)

        secuencia = torch.cat([secuencia, siguiente], dim=1)

        token_generado_id = int(siguiente.item())
        if token_generado_id == id_eos:
            break

        # Protección anti-bucle: si genera 3 tokens seguidos exactamente iguales, detener
        tokens_actuales = secuencia[0].tolist()
        if len(tokens_actuales) > 3 and tokens_actuales[-1] == tokens_actuales[-2] == tokens_actuales[-3]:
            break

        texto_actual = tokenizador.decodificar(tokens_actuales[len(ids):])
        if "[INSTRUCCION]" in texto_actual or "\n\n\n" in texto_actual:
            break

    ids_generados = secuencia[0].tolist()[len(ids):]
    texto_generado = tokenizador.decodificar(ids_generados).strip()

    for corte in ["[INSTRUCCION]", "<EOS>", "<PAD>", "[RESPUESTA]"]:
        if corte in texto_generado:
            texto_generado = texto_generado.split(corte)[0].strip()

    return texto_generado


if __name__ == "__main__":
    modelo, tokenizador = cargar_modelo()
    prompt = input("Escribe un inicio de texto: ").strip()
    print("\nGENERACIÓN:\n")
    print(
        generar_texto(
            modelo,
            tokenizador,
            prompt,
            temperatura=0.4,
            top_k=15
        )
    )