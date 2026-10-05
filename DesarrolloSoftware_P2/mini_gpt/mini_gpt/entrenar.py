import time
import torch
from config import ARCHIVO_MODELO, ARCHIVO_HISTORIAL, EPOCAS, TASA_APRENDIZAJE, MOSTRAR_CADA, DISPOSITIVO, LONGITUD_CONTEXTO, TAMANIO_BATCH, DIMENSION_EMBEDDING, NUM_CABEZAS, NUM_CAPAS
from dataset import GestorDataset
from modelo import MiniGPT
from tokenizador import TokenizadorEducativo
from utilidades import fijar_semilla, preparar_carpetas, contar_parametros, guardar_json

def evaluar(modelo, dataset, pasos=10):
    modelo.eval(); perdidas=[]
    with torch.no_grad():
        for _ in range(pasos):
            x,y = dataset.obtener_batch("val")
            _, perdida = modelo(x,y)
            perdidas.append(perdida.item())
    modelo.train()
    return sum(perdidas)/len(perdidas)

def entrenar():
    fijar_semilla(); preparar_carpetas()
    tokenizador = TokenizadorEducativo(); tokenizador.cargar()
    dataset = GestorDataset()
    modelo = MiniGPT(tokenizador.tamanio_vocabulario).to(DISPOSITIVO)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=TASA_APRENDIZAJE)
    pasos_por_epoca = max(1, len(dataset.train)//(TAMANIO_BATCH*LONGITUD_CONTEXTO))
    print(f"Dispositivo: {DISPOSITIVO}")
    print(f"Vocabulario: {tokenizador.tamanio_vocabulario:,}")
    print(f"Parámetros entrenables: {contar_parametros(modelo):,}")
    print(f"Pasos aproximados por época: {pasos_por_epoca}")
    historial=[]; paso_global=0; inicio=time.time()
    for epoca in range(1, EPOCAS+1):
        modelo.train()
        for _ in range(pasos_por_epoca):
            x,y = dataset.obtener_batch("train")
            optimizador.zero_grad(set_to_none=True)
            _, perdida = modelo(x,y)
            perdida.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            optimizador.step()
            paso_global += 1
            if paso_global % MOSTRAR_CADA == 0:
                val = evaluar(modelo,dataset)
                historial.append({"epoca":epoca,"paso":paso_global,"perdida_train":float(perdida.item()),"perdida_val":float(val)})
                print(f"Época {epoca:03d} | Paso {paso_global:05d} | Train {perdida.item():.4f} | Val {val:.4f}")
    checkpoint={
        "estado_modelo":modelo.state_dict(),
        "tamanio_vocabulario":tokenizador.tamanio_vocabulario,
        "configuracion":{"longitud_contexto":LONGITUD_CONTEXTO,"dimension_embedding":DIMENSION_EMBEDDING,"num_cabezas":NUM_CABEZAS,"num_capas":NUM_CAPAS}
    }
    torch.save(checkpoint, ARCHIVO_MODELO)
    guardar_json(historial, ARCHIVO_HISTORIAL)
    print(f"Entrenamiento finalizado en {time.time()-inicio:.1f}s")
    print(f"Modelo guardado en {ARCHIVO_MODELO}")
    return modelo

if __name__ == "__main__":
    entrenar()
