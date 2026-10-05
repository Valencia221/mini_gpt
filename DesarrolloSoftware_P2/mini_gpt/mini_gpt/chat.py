import re
from generar import cargar_modelo, generar_texto


def limpiar_respuesta_generada(texto_generado, prompt):
    """Limpia el texto generado eliminando rastros del prompt y etiquetas de control."""
    if texto_generado.startswith(prompt):
        respuesta = texto_generado[len(prompt):]
    else:
        respuesta = texto_generado

    patrones_corte = [
        r"\[INSTRUCCION\]",
        r"\[RESPUESTA\]",
        r"Pregunta:",
        r"Usuario:",
        r"\n\n\n"
    ]
    
    for patron in patrones_corte:
        respuesta = re.split(patron, respuesta, flags=re.IGNORECASE)[0]

    return respuesta.strip()


def main():
    modelo, tokenizador = cargar_modelo()

    print("\n==================================================")
    print("  MINI GPT EDUCATIVO - SISMICIDAD Y GEOLOGÍA  ")
    print("  Escribe 'salir' para terminar")
    print("==================================================\n")

    while True:
        pregunta = input("Usuario: ").strip()

        if not pregunta:
            continue

        if pregunta.lower() in {"salir", "exit", "quit"}:
            print("\n¡Hasta luego!")
            break

        # Formato de instrucción alineado con tu nuevo corpus
        prompt = f"[INSTRUCCION] {pregunta}\n[RESPUESTA]"

        salida_raw = generar_texto(
            modelo,
            tokenizador,
            prompt
        )

        respuesta_limpia = limpiar_respuesta_generada(salida_raw, prompt)

        if not respuesta_limpia:
            respuesta_limpia = "No he podido procesar una respuesta coherente con el contexto actual. Prueba reformulando la pregunta."

        print(f"\nModelo: {respuesta_limpia}\n")


if __name__ == "__main__":
    main()
from generar import cargar_modelo, generar_texto


def limpiar_respuesta_generada(texto_generado, prompt):
    """Limpia el texto generado eliminando rastros del prompt y etiquetas de control."""
    if texto_generado.startswith(prompt):
        respuesta = texto_generado[len(prompt):]
    else:
        respuesta = texto_generado

    patrones_corte = [
        r"\[INSTRUCCION\]",
        r"\[RESPUESTA\]",
        r"Pregunta:",
        r"Usuario:",
        r"\n\n\n"
    ]
    
    for patron in patrones_corte:
        respuesta = re.split(patron, respuesta, flags=re.IGNORECASE)[0]

    return respuesta.strip()


def main():
    modelo, tokenizador = cargar_modelo()

    print("\n==================================================")
    print("  MINI GPT EDUCATIVO - SISMICIDAD Y GEOLOGÍA  ")
    print("  Escribe 'salir' para terminar")
    print("==================================================\n")

    while True:
        pregunta = input("Usuario: ").strip()

        if not pregunta:
            continue

        if pregunta.lower() in {"salir", "exit", "quit"}:
            print("\n¡Hasta luego!")
            break

        # Formato de instrucción alineado con tu nuevo corpus
        prompt = f"[INSTRUCCION] {pregunta}\n[RESPUESTA]"

        salida_raw = generar_texto(
            modelo,
            tokenizador,
            prompt
        )

        respuesta_limpia = limpiar_respuesta_generada(salida_raw, prompt)

        if not respuesta_limpia:
            respuesta_limpia = "No he podido procesar una respuesta coherente con el contexto actual. Prueba reformulando la pregunta."

        print(f"\nModelo: {respuesta_limpia}\n")


if __name__ == "__main__":
    main()
from generar import cargar_modelo, generar_texto


def limpiar_respuesta_generada(texto_generado, prompt):
    """Limpia el texto generado eliminando rastros del prompt y etiquetas de control."""
    if texto_generado.startswith(prompt):
        respuesta = texto_generado[len(prompt):]
    else:
        respuesta = texto_generado

    patrones_corte = [
        r"\[INSTRUCCION\]",
        r"\[RESPUESTA\]",
        r"Pregunta:",
        r"Usuario:",
        r"\n\n\n"
    ]
    
    for patron in patrones_corte:
        respuesta = re.split(patron, respuesta, flags=re.IGNORECASE)[0]

    return respuesta.strip()


def main():
    modelo, tokenizador = cargar_modelo()

    print("\n==================================================")
    print("  MINI GPT EDUCATIVO - SISMICIDAD Y GEOLOGÍA  ")
    print("  Escribe 'salir' para terminar")
    print("==================================================\n")

    while True:
        pregunta = input("Usuario: ").strip()

        if not pregunta:
            continue

        if pregunta.lower() in {"salir", "exit", "quit"}:
            print("\n¡Hasta luego!")
            break

        # Formato de instrucción alineado con tu nuevo corpus
        prompt = f"[INSTRUCCION] {pregunta}\n[RESPUESTA]"

        salida_raw = generar_texto(
            modelo,
            tokenizador,
            prompt
        )

        respuesta_limpia = limpiar_respuesta_generada(salida_raw, prompt)

        if not respuesta_limpia:
            respuesta_limpia = "No he podido procesar una respuesta coherente con el contexto actual. Prueba reformulando la pregunta."

        print(f"\nModelo: {respuesta_limpia}\n")


if __name__ == "__main__":
    main()