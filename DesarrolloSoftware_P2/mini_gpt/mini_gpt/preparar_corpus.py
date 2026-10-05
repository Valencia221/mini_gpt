import json
import re
from config import CARPETA_CORPUS, ARCHIVO_CORPUS_LIMPIO
from utilidades import preparar_carpetas

EXTENSIONES_VALIDAS = {".txt", ".md", ".json", ".pdf"}

def leer_txt_md(ruta):
    return ruta.read_text(encoding="utf-8", errors="ignore")

def extraer_textos_json(objeto):
    textos = []
    if isinstance(objeto, str):
        textos.append(objeto)
    elif isinstance(objeto, dict):
        for valor in objeto.values():
            textos.extend(extraer_textos_json(valor))
    elif isinstance(objeto, list):
        for item in objeto:
            textos.extend(extraer_textos_json(item))
    return textos

def leer_json(ruta):
    with open(ruta, "r", encoding="utf-8") as f:
        datos = json.load(f)
    return "\n".join(extraer_textos_json(datos))

def leer_pdf(ruta):
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("Para leer PDF instala pypdf: pip install pypdf")
    lector = PdfReader(str(ruta))
    return "\n".join((pagina.extract_text() or "") for pagina in lector.pages)

def limpiar_texto_pdf(texto):
    """Limpieza profunda diseñada para texto extraído de PDFs académicos/técnicos."""
    texto = texto.replace("\x00", " ")
    texto = re.sub(r"\r\n?", "\n", texto)

    # Unir palabras cortadas por guion al final de línea
    texto = re.sub(r'(\w+)-\n(\w+)', r'\1\2', texto)

    # Eliminar URLs, DOIs e ISSNs
    texto = re.sub(r"https?://\S+|www\.\S+", " ", texto)
    texto = re.sub(r"doi:\s*\S+", " ", texto, flags=re.IGNORECASE)
    texto = re.sub(r"ISSN[\s:0-9\-]+", " ", texto, flags=re.IGNORECASE)

    # Eliminar marcas de agua/encabezados
    texto = re.sub(r"microsoft edge pdf document", " ", texto, flags=re.IGNORECASE)
    texto = re.sub(r"microsoft edge pdf", " ", texto, flags=re.IGNORECASE)

    # Eliminar pies de página y números sueltos
    texto = re.sub(r"p[aá]gina\s+\d+(\s+de\s+\d+)?", " ", texto, flags=re.IGNORECASE)
    texto = re.sub(r"^\s*\d+\s*$", "", texto, flags=re.MULTILINE)

    # Limpieza de citas y referencias a figuras
    texto = re.sub(r"\b(figura|tabla|gr[aá]fico|ilustraci[oó]n|anexo)\s+\d+(\.\d+)*\b", "", texto, flags=re.IGNORECASE)
    texto = re.sub(r"\b\d+\.\d+\.\d+(\.\d+)*\b", " ", texto)
    texto = re.sub(r"\[\d+\]", "", texto)
    texto = re.sub(r"\([A-Z][a-z]+(?:\s+et\s+al\.)?,\s*\d{4}\)", "", texto)

    # Unir párrafos divididos
    texto = re.sub(r"(?<!\n)\n(?!\n)", " ", texto)

    # Espacios finales
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)

    return texto.strip()

def limpiar_texto_simple(texto):
    """Limpieza suave para archivos .txt/.md/.json para PRESERVAR formatos de instrucción."""
    texto = texto.replace("\x00", " ")
    texto = re.sub(r"\r\n?", "\n", texto)
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()

def preparar_corpus():
    preparar_carpetas()
    CARPETA_CORPUS.mkdir(parents=True, exist_ok=True)
    archivos = [p for p in CARPETA_CORPUS.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONES_VALIDAS]
    
    if not archivos:
        raise RuntimeError(f"No se encontraron documentos en {CARPETA_CORPUS}")
    
    bloques = []
    for ruta in sorted(archivos):
        print(f"Leyendo: {ruta.name}")
        ext = ruta.suffix.lower()
        
        if ext in {".txt", ".md"}:
            texto = leer_txt_md(ruta)
            texto = limpiar_texto_simple(texto)
        elif ext == ".json":
            texto = leer_json(ruta)
            texto = limpiar_texto_simple(texto)
        elif ext == ".pdf":
            texto = leer_pdf(ruta)
            texto = limpiar_texto_pdf(texto)
        else:
            continue
            
        if texto:
            bloques.append(f"\n<DOCUMENTO> {ruta.name}\n{texto}\n</DOCUMENTO>\n")
            
    corpus = "\n".join(bloques)
    ARCHIVO_CORPUS_LIMPIO.write_text(corpus, encoding="utf-8")
    
    print(f"\n¡Corpus preparado exitosamente en: {ARCHIVO_CORPUS_LIMPIO}!")
    print(f"Documentos procesados: {len(bloques)} | Caracteres totales: {len(corpus):,}")
    return corpus

if __name__ == "__main__":
    preparar_corpus()