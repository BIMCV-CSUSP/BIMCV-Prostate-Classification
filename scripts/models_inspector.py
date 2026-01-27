# ==============================================================================
# VOLCADO DE ESTRUCTURA COMPLETA EFFICIENTNET-B7
# Guarda en un .txt todas las capas y sus resoluciones para elegir Target Layer.
# ==============================================================================
import torch
import sys
import os

# --- 1. CONFIGURACIÓN ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUTPUT_FILE = "estructura_modelo_completa.txt"

# Rutas (Mismas que tu configuración)
MODEL_PATHS = [
    "../logs/classification/Prostate_BIMCV/02-Sep-2025-16:24:58/models/best-model-weights.pth",
]

try:
    from bimcv_prostate.models.efficientnet import EfficientNet_pretrained
except ImportError:
    print("[ERROR] Ejecuta esto desde la raíz del proyecto para importar bimcv_prostate.")
    sys.exit(1)

# --- 2. CARGA DEL MODELO ---
def load_model():
    path = MODEL_PATHS[0]
    if not os.path.exists(path):
        print(f"[ERROR] No encuentro el modelo: {path}")
        return None
    
    print(f"Cargando modelo desde: {path}")
    state = torch.load(path, map_location="cpu")["state_dict"]
    
    if list(state.keys())[0].startswith("model."):
        model = EfficientNet_pretrained("efficientnet-b7", 2, 3, None)
        model.load_state_dict(state)
    else:
        model = EfficientNet_pretrained("efficientnet-b7", 2, 3, path)
    
    model.to(device).eval()
    return model

# --- 3. GENERADOR DE REPORTE ---
def dump_structure_to_file(model, filename):
    print(f"Iniciando análisis de estructura... (Esto puede tardar unos segundos)")
    
    layer_info = {}
    hooks = []

    def hook_fn(name):
        def forward_hook(module, input, output):
            # Guardamos la forma de salida
            if isinstance(output, torch.Tensor):
                layer_info[name] = tuple(output.shape)
            elif isinstance(output, (list, tuple)) and isinstance(output[0], torch.Tensor):
                # A veces devuelven tuplas
                layer_info[name] = tuple(output[0].shape)
        return forward_hook

    # Registramos hooks en TODAS las capas con nombre
    # No filtramos nada esta vez para que tengas la visión total
    for name, module in model.named_modules():
        # Evitamos registrar el contenedor padre 'model' para no duplicar
        if name == "" or name == "model": 
            continue
        # Nos interesan principalmente las capas que hacen operaciones (Conv, BN, Act)
        # o bloques enteros. Registramos todo para estar seguros.
        hooks.append(module.register_forward_hook(hook_fn(name)))

    # Input dummy (Tamaño estándar de tus imágenes)
    dummy = torch.zeros((1, 3, 320, 320, 24)).to(device)
    
    try:
        with torch.no_grad():
            model(dummy)
    except Exception as e:
        print(f"Error en forward pass: {e}")
        return
    finally:
        for h in hooks: h.remove()

    # --- ESCRITURA EN ARCHIVO ---
    print(f"Guardando resultados en: {filename}")
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("="*100 + "\n")
        f.write(f"ESTRUCTURA COMPLETA DEL MODELO (EfficientNet-B7 3D)\n")
        f.write(f"Input simulado: {tuple(dummy.shape)}\n")
        f.write("="*100 + "\n")
        f.write(f"{'Nombre de la Capa':<60} | {'Shape Salida (B, C, D, H, W)':<30} | {'Notas'}\n")
        f.write("-" * 100 + "\n")

        # Recorremos en orden de ejecución (aproximado por nombre)
        # Al ser named_modules() ordenado por definición, debería salir en orden topológico aprox.
        prev_h = 0
        
        for name, module in model.named_modules():
            if name in layer_info:
                shape = layer_info[name]
                shape_str = str(shape)
                
                # Análisis de resolución
                note = ""
                if len(shape) == 5: # (B, C, D, H, W)
                    d, h, w = shape[2], shape[3], shape[4]
                    
                    # Detectar cambio de resolución (Pooling/Stride)
                    if prev_h != 0 and h < prev_h:
                        f.write(f"{'--- BAJADA DE RESOLUCIÓN ---':^100}\n")
                    
                    prev_h = h
                    
                    # Sugerencias
                    if h >= 20 and h <= 40 and "project_conv" in name:
                        note = "<--- ¡BUENA OPCIÓN!"
                    elif h == 10:
                        note = "(Muy baja res)"
                
                f.write(f"{name:<60} | {shape_str:<30} | {note}\n")

    print("¡Listo! Revisa el archivo generado.")

if __name__ == "__main__":
    model = load_model()
    if model:
        dump_structure_to_file(model, OUTPUT_FILE)