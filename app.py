 

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, classification_report

# --- Lazy imports for heavy ML libs (allow deploy without TF/JAX/PennyLane) ---
HAS_TF = True
HAS_PENNYLANE = True
# Placeholders for symbols that normally come from TF/Keras
tf = None
load_img = None
img_to_array = None
image_dataset_from_directory = None
vgg16_preprocess = None
try:
    import tensorflow as tf
    # Import Keras helpers from tensorflow.keras inside the try so missing TF
    # doesn't crash the app at import time on Streamlit Cloud when requirements
    # omit heavy ML packages.
    try:
        from tensorflow.keras.utils import load_img, img_to_array, image_dataset_from_directory
        from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
    except Exception:
        # If keras helpers aren't available, keep placeholders as None
        load_img = None
        img_to_array = None
        image_dataset_from_directory = None
        vgg16_preprocess = None
    HAS_TF = True
except Exception:
    tf = None
    load_img = None
    img_to_array = None
    image_dataset_from_directory = None
    vgg16_preprocess = None
    HAS_TF = False

try:
    import pennylane as _qml
    HAS_PENNYLANE = True
except Exception:
    HAS_PENNYLANE = False

# NOTE: touch for deploy/no-tf - ensure Streamlit Cloud picks up latest commit
# deploy-touch: 2025-11-02 21:10:22 -0500

# ============================
# Configuración inicial
# ============================
st.set_page_config(page_title="Clasificación de Aves", page_icon="🦜", layout="wide")
st.title("🦜 Clasificador de Aves – Demo")
st.caption("Sube una imagen para identificar la especie o revisa el desempeño del modelo en el conjunto de validación.")

# Directorio raíz de modelos
MODELS_DIR = "modelos"
BATCH_SIZE = 32 # Tamaño de lote para evaluación
IMG_SIZE = (224, 224)  # Forma esperada por VGG16
# Opcional: forzar una lista blanca de modelos (usar nombres de carpeta o nombres de archivo sin extensión)
# Si quieres usar sólo los dos modelos clásicos, define aquí la lista. Poner None para usar cualquier modelo en `modelos/`.
MODEL_WHITELIST = [
    "0vgg16_01_l_128_acc_32_42_data04",
    "mobilenetv2_aves_01_l_128_acc_32_42_data04",
]
# ============================
# Cargar modelo entrenado
# ============================
# Cargar modelo entrenado (SavedModel) usando Keras 3


@st.cache_resource(show_spinner=False)
def load_model_generic(path: str):
    """Cargar un modelo: puede ser SavedModel (dir con saved_model.pb) o .keras/.h5 Keras file.
    Devuelve un objeto invocable (TFSMLayer o tf.keras.Model).
    """
    # SavedModel directory -> TFSMLayer (inferencia vía signature)
    if os.path.isdir(path) and os.path.isfile(os.path.join(path, "saved_model.pb")):
        if not HAS_TF:
            raise RuntimeError("TensorFlow no está disponible en este despliegue. Cargue el modelo localmente en `.venv310` para usar SavedModel.")
        # Importar TFSMLayer localmente para evitar error en entornos sin TF
        try:
            from keras.layers import TFSMLayer
        except Exception:
            try:
                # Alternativa: tensorflow.keras.layers (si aplica)
                from tensorflow.keras.layers import TFSMLayer
            except Exception:
                raise RuntimeError("TFSMLayer no está disponible en este entorno. No se puede cargar SavedModel.")
        return TFSMLayer(path, call_endpoint="serving_default")

    # Keras file (.keras, .h5) -> tf.keras.models.load_model
    if os.path.isfile(path) and path.lower().endswith((".keras", ".h5", ".hdf5")):
        # importar dentro de la función para evitar error si pennylane no está instalado
        from tensorflow import keras
        # Intentar cargar el .keras manejando la capa KerasLayer de PennyLane
        # Si pennylane está instalado, pasamos la clase en `custom_objects` para
        # que Keras pueda deserializar la capa correctamente.
        # Intentar cargar el .keras manejando la capa KerasLayer de PennyLane.
        # Si pennylane está instalado, intentaremos primero una carga normal; si falla
        # por faltar el argumento `qnode`, construiremos un loader que recree un
        # qnode razonable y volvamos a intentar con `custom_objects`.
        if not HAS_TF:
            raise RuntimeError("TensorFlow no está disponible en este despliegue. Cargue el modelo localmente en `.venv310` para usar archivos .keras.")
        try:
            try:
                import pennylane as qml
                from pennylane.qnn.keras import KerasLayer as PLKerasLayer
            except Exception:
                qml = None
                PLKerasLayer = None

            # Intento de carga simple primero
            try:
                model = keras.models.load_model(path, compile=False)
                return model
            except Exception as e_load:
                # Si falla y parece relacionado con KerasLayer/qnode, intentar fallback
                msg = str(e_load)
                if qml is None or "KerasLayer" not in msg and "qnode" not in msg:
                    # No parece un error solucionable aquí: propagar con mensaje informativo
                    raise RuntimeError(
                        "Error cargando el archivo .keras. Si el modelo contiene una capa de PennyLane (KerasLayer), "
                        "asegúrate de tener instalado `pennylane` y versiones compatibles de TensorFlow/Keras. "
                        f"Error original: {e_load}"
                    ) from e_load

                # Construir una clase fallback con `from_config` y pasarla en custom_objects.
                # Esto evita registrar una clase serializable (que requiere get_config()).
                def build_keraslayer_fallback():
                    try:
                        import pennylane as _qml
                        from pennylane.qnn.keras import KerasLayer as _PLKerasLayer

                        class KerasLayerFallback:
                            @classmethod
                            def from_config(cls, config, custom_objects=None):
                                # Intentar inferir información útil desde la config
                                weight_shapes = config.get("weight_shapes", None)
                                output_dim = config.get("output_dim", None)
                                ws = None
                                try:
                                    # Algunos serializados ponen estructuras varias en weight_shapes
                                    if isinstance(weight_shapes, dict):
                                        ws = weight_shapes.get("weights") or weight_shapes
                                    else:
                                        ws = weight_shapes
                                except Exception:
                                    ws = weight_shapes

                                # heurística para n_qubits: preferir inferir de output_dim si es potencia de 2
                                import math

                                def is_power_of_two(n):
                                    return n > 0 and (n & (n - 1)) == 0

                                n_qubits = None
                                if output_dim is not None:
                                    try:
                                        od = int(output_dim)
                                        if is_power_of_two(od):
                                            n_qubits = int(math.log2(od))
                                    except Exception:
                                        n_qubits = None

                                # Si no deducido, intentar heurística desde weight_shapes
                                if n_qubits is None:
                                    n_qubits = 4
                                    if isinstance(ws, (list, tuple)) and len(ws) >= 2:
                                        try:
                                            # heurística antigua: ws[1] era n_qubits en algunos modelos
                                            candidate = int(ws[1])
                                            if 1 <= candidate <= 16:
                                                n_qubits = candidate
                                        except Exception:
                                            pass

                                dev = _qml.device("default.qubit", wires=n_qubits)

                                # Construir circuito en función de output_dim:
                                # - Si output_dim es potencia de 2 y coincide con 2**n_qubits -> devolver probs
                                # - Si output_dim es pequeño y no potencia de 2 -> devolver expectativas (PauliZ)
                                def circuit_probs(inputs, weights):
                                    # Evitar AngleEmbedding (validaciones estrictas de features).
                                    # Hacemos un embedding simple: aplicar RY con entradas a cada qubit
                                    L = int(getattr(inputs, "shape", (len(inputs),))[-1]) if hasattr(inputs, "shape") else len(inputs)
                                    for j in range(n_qubits):
                                        idx = j % L
                                        _qml.RY(inputs[idx], wires=j)
                                    _qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                                    return _qml.probs(wires=list(range(n_qubits)))

                                def circuit_expvals(inputs, weights, n_out):
                                    # Embedding manual: map elementos de inputs a qubits por rotaciones RY
                                    L = int(getattr(inputs, "shape", (len(inputs),))[-1]) if hasattr(inputs, "shape") else len(inputs)
                                    for j in range(n_qubits):
                                        idx = j % L
                                        _qml.RY(inputs[idx], wires=j)
                                    _qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                                    res = []
                                    for i in range(n_out):
                                        wire = i % n_qubits
                                        res.append(_qml.expval(_qml.PauliZ(wire)))
                                    return tuple(res) if len(res) > 1 else res[0]

                                # Elegir tipo de medición
                                use_probs = False
                                if output_dim is not None:
                                    try:
                                        od = int(output_dim)
                                        if is_power_of_two(od) and 2 ** n_qubits == od:
                                            use_probs = True
                                        else:
                                            use_probs = False
                                    except Exception:
                                        use_probs = False
                                else:
                                    # sin output_dim asumimos probs (común en ejemplos)
                                    use_probs = True

                                if use_probs:
                                    qnode = _qml.QNode(circuit_probs, dev, interface="tf")
                                else:
                                    # cuando usamos expvals debemos envolver para devolver vector
                                    def qfunc(inputs, weights):
                                        return circuit_expvals(inputs, weights, output_dim or n_qubits)

                                    qnode = _qml.QNode(qfunc, dev, interface="tf")

                                # Crear KerasLayer pasando weight_shapes y output_dim si es posible
                                try:
                                    if output_dim is not None:
                                        return _PLKerasLayer(qnode, weight_shapes=weight_shapes, output_dim=output_dim)
                                    else:
                                        return _PLKerasLayer(qnode, weight_shapes=weight_shapes)
                                except Exception:
                                    # último recurso: crear sin weight_shapes
                                    if output_dim is not None:
                                        return _PLKerasLayer(qnode, output_dim=output_dim)
                                    return _PLKerasLayer(qnode)

                        return KerasLayerFallback
                    except Exception:
                        return None

                fallback_cls = build_keraslayer_fallback()
                custom = {}
                if fallback_cls is not None:
                    custom["KerasLayer"] = fallback_cls
                    custom["pennylane.qnn.keras.KerasLayer"] = fallback_cls
                elif PLKerasLayer is not None:
                    custom["KerasLayer"] = PLKerasLayer
                    custom["pennylane.qnn.keras.KerasLayer"] = PLKerasLayer

                model = keras.models.load_model(path, compile=False, custom_objects=custom if custom else None)
                # Si el modelo contiene capas de PennyLane (KerasLayer) que usan embeddings
                # originales (AngleEmbedding), reemplazamos su qnode por una versión
                # que usa embedding manual (RY) para evitar errores de longitud de features.
                try:
                    import pennylane as _qml2
                    from pennylane.qnn.keras import KerasLayer as _PLKerasLayer2

                    def rebuild_qnode_for_layer(layer):
                        try:
                            cfg = layer.get_config() or {}
                        except Exception:
                            cfg = {}
                        weight_shapes = cfg.get("weight_shapes", None)
                        output_dim = cfg.get("output_dim", None)

                        # heurística n_qubits
                        import math

                        def is_power_of_two(n):
                            return n > 0 and (n & (n - 1)) == 0

                        n_qubits = None
                        if output_dim is not None:
                            try:
                                od = int(output_dim)
                                if is_power_of_two(od):
                                    n_qubits = int(math.log2(od))
                            except Exception:
                                n_qubits = None

                        if n_qubits is None:
                            n_qubits = 4
                            try:
                                ws = weight_shapes
                                if isinstance(ws, dict):
                                    ws = ws.get("weights") or ws
                                if isinstance(ws, (list, tuple)) and len(ws) >= 2:
                                    cand = int(ws[1])
                                    if 1 <= cand <= 32:
                                        n_qubits = cand
                            except Exception:
                                pass

                        dev = _qml2.device("default.qubit", wires=n_qubits)

                        def q_embed_ry(inputs, weights):
                            L = int(getattr(inputs, "shape", (len(inputs),))[-1]) if hasattr(inputs, "shape") else len(inputs)
                            for j in range(n_qubits):
                                idx = j % L
                                _qml2.RY(inputs[idx], wires=j)
                            _qml2.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                            # devolver probs si coincide con potencia de dos
                            if output_dim is not None:
                                try:
                                    od = int(output_dim)
                                    if is_power_of_two(od) and 2 ** n_qubits == od:
                                        return _qml2.probs(wires=list(range(n_qubits)))
                                except Exception:
                                    pass
                            # por defecto devolver expectativas PauliZ hasta output_dim
                            n_out = int(output_dim) if output_dim is not None else n_qubits
                            res = []
                            for i in range(n_out):
                                res.append(_qml2.expval(_qml2.PauliZ(i % n_qubits)))
                            return tuple(res) if len(res) > 1 else res[0]

                        qnode = _qml2.QNode(q_embed_ry, dev, interface="tf")
                        try:
                            # intentar reemplazar el qnode del layer
                            layer.qnode = qnode
                            try:
                                layer._qnode = qnode
                            except Exception:
                                pass
                            return True
                        except Exception:
                            return False

                    # Aplicar sobre todas las capas si el modelo es un tf.keras.Model
                    if hasattr(model, "layers"):
                        for lyr in model.layers:
                            try:
                                if isinstance(lyr, _PLKerasLayer2):
                                    rebuild_qnode_for_layer(lyr)
                            except Exception:
                                # ignorar capas que no se puedan procesar
                                pass
                except Exception:
                    # Si pennylane no está disponible o ocurre error, continuar sin parche
                    pass

                return model
        except Exception as e:
            # Mejor mensaje de error para el usuario: indicar que puede necesitar pennylane
            raise RuntimeError(
                "Error cargando el archivo .keras. Si el modelo contiene una capa de PennyLane (KerasLayer), "
                "asegúrate de tener instalado `pennylane` y una versión de TensorFlow/Keras compatible. "
                f"Error original: {e}"
            )

    raise ValueError(f"Ruta de modelo no válida o formato no soportado: {path}")

# ============================
# Datasets y utilidades
# ============================

def get_dataset(split: str = "valid", img_size=IMG_SIZE):
    directory = os.path.join("datos", split)
    if image_dataset_from_directory is None:
        raise RuntimeError("Función image_dataset_from_directory no disponible: TensorFlow no está instalado en este despliegue.")
    ds = image_dataset_from_directory(
        directory,
        image_size=img_size,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    return ds


def get_model_diagnostics(model):
    """Recolectar información útil sobre el modelo y sus capas para depuración.
    Devuelve una lista de dicts con: name, class, input_shape, output_shape, extra (config/qnode info).
    """
    diagnostics = []
    try:
        from tensorflow.keras.models import Model
    except Exception:
        Model = None

    def _shape_to_list(s):
        try:
            if hasattr(s, "as_list"):
                return s.as_list()
            return tuple(int(x) if x is not None else None for x in s)
        except Exception:
            return str(s)

    if hasattr(model, "layers"):
        for lyr in getattr(model, "layers", []):
            info = {
                "name": getattr(lyr, "name", str(type(lyr))),
                "class": type(lyr).__name__,
                "input_shape": None,
                "output_shape": None,
                "extra": {},
            }
            try:
                if hasattr(lyr, "input_shape"):
                    info["input_shape"] = _shape_to_list(lyr.input_shape)
                elif hasattr(lyr, "input_spec") and getattr(lyr, "input_spec") is not None:
                    spec = lyr.input_spec
                    if isinstance(spec, (list, tuple)):
                        spec = spec[0]
                    if hasattr(spec, "shape"):
                        info["input_shape"] = _shape_to_list(spec.shape)
            except Exception:
                pass
            try:
                if hasattr(lyr, "output_shape"):
                    info["output_shape"] = _shape_to_list(lyr.output_shape)
            except Exception:
                pass

            # Si es una KerasLayer de PennyLane, añadir config y qnode_weights
            try:
                modname = type(lyr).__module__
                clsname = type(lyr).__name__
                if "pennylane" in modname or clsname.lower().find("keraslayer") >= 0:
                    try:
                        cfg = lyr.get_config()
                        info["extra"]["config"] = cfg
                    except Exception:
                        info["extra"]["config"] = "<no get_config()>"
                    try:
                        qw = getattr(lyr, "qnode_weights", None)
                        if isinstance(qw, dict):
                            info["extra"]["qnode_weights"] = {k: getattr(v, "shape", str(type(v))) for k, v in qw.items()}
                        else:
                            info["extra"]["qnode_weights"] = str(type(qw))
                    except Exception:
                        info["extra"]["qnode_weights"] = "<error>"
            except Exception:
                pass

            diagnostics.append(info)
    else:
        diagnostics.append({"model": str(type(model)), "note": "No layers attribute"})

    return diagnostics

# Función para cargar nombres de clases
def list_available_models():
    models = {}
    if os.path.isdir(MODELS_DIR):
        for name in sorted(os.listdir(MODELS_DIR)):
            path = os.path.join(MODELS_DIR, name)
            # SavedModel directory
            if os.path.isdir(path) and os.path.isfile(os.path.join(path, "saved_model.pb")):
                models[name] = path
            # Keras single-file models (.keras, .h5)
            if os.path.isfile(path) and path.lower().endswith((".keras", ".h5", ".hdf5")):
                # use filename without extension as model name
                key = os.path.splitext(name)[0]
                models[key] = path
    # Si existe una lista blanca, filtrar los modelos a los listados ahí
    if MODEL_WHITELIST:
        filtered = {}
        for keep in MODEL_WHITELIST:
            if keep in models:
                filtered[keep] = models[keep]
            else:
                # también intentar detectar por prefijo/contiene (por si el nombre de archivo difiere)
                for k, v in models.items():
                    if keep == k or k.startswith(keep) or keep in k:
                        filtered[k] = v
                        break
        if not filtered:
            # si la whitelist no coincide con nada, devolver todos para no romper la app
            st.warning("MODEL_WHITELIST configurada pero no coincide con modelos en 'modelos/'; mostrando todos.")
            return models
        return filtered

    return models

@st.cache_data(show_spinner=False)
def load_class_names(model_path: str):
    # 1) Buscar archivo con clases al lado del modelo
    candidates = [
        os.path.join(model_path, "classes.txt"),
        os.path.join(model_path, "labels.txt"),
        os.path.join(model_path, "class_indices.json"),
    ]
    # Intentar cargar desde los archivos
    for path in candidates:
        if os.path.isfile(path):
            try:
                if path.endswith(".json"):
                    import json
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    # data puede ser {class_name: index} o [class_name,...]
                    if isinstance(data, dict):
                        # Ordenar por índice
                        return [k for k, _ in sorted(data.items(), key=lambda kv: kv[1])]
                    elif isinstance(data, list):
                        return data
                else:
                    with open(path, "r", encoding="utf-8") as f:
                        lines = [ln.strip() for ln in f if ln.strip()]
                    if lines:
                        return lines
            except Exception:
                pass
    # 2) Fallback: usar clases del dataset de validación , si existe    
    # Sólo intentar crear un dataset si la función está disponible (TF presente)
    if image_dataset_from_directory is not None:
        try:
            _tmp_ds = image_dataset_from_directory(
                os.path.join("datos", "valid"), image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
            )
            return list(_tmp_ds.class_names)
        except Exception:
            pass
    # Si no se puede inferir desde dataset, devolver lista vacía para no romper la UI
    return []

AVAILABLE_MODELS = list_available_models()
if not AVAILABLE_MODELS:
    st.warning("No se encontraron modelos en la carpeta 'modelos/'.")
    st.stop()

# Descripciones breves por especie (editable) este es un arreglo
DESCRIPCIONES = {
    "Ara ararauna": "Guacamaya azul y amarilla; habita selvas tropicales, gran tamaño y vocalizaciones fuertes.",
    "Chroicocephalus ridibundus": "Gaviota reidora; frecuente en humedales y costas, plumaje blanco con capucha estacional.",
    "Eubucco bourcierii": "Barbudito andino; pequeño frugívoro de bosques montanos con colores vivos.",
    "Laterallus albigularis": "Polluela; ave de humedales y vegetación densa, de hábitos esquivos.",
    "Melanerpes formicivorus": "Carpintero bellotero; social, almacena bellotas, típico de bosques abiertos.",
    "Patagioenas subvinacea": "Paloma subvinácea; habita bosques húmedos, de vuelo rápido y llamado profundo.",
    "Platalea ajaja": "Espátula rosada; zancuda de color rosado con pico espatulado, forrajea en aguas someras.",
    "Rynchops niger": "Rayador americano; recorta la superficie del agua con su pico para capturar peces.",
    "Theristicus caudatus": "Ibis de cuello blanco; común en pastizales y humedales, pico curvo para sondar el suelo.",
    "Vultur gryphus": "Cóndor andino; gran carroñero de los Andes, una de las aves voladoras más grandes.",
}

# Nombres comunes por especie (según mapeo proporcionado)
COMMON_NAMES = {
    "Ara ararauna": "Guacamayo Azuliamarillo",
    "Chroicocephalus ridibundus": "Gaviota Reidora",
    "Eubucco bourcierii": "Cabezón Cabecirrojo",
    "Laterallus albigularis": "Polluela Carrasqueadora",
    "Melanerpes formicivorus": "Carpintero Bellotero",
    "Patagioenas subvinacea": "Paloma Vinosa",
    "Platalea ajaja": "Espátula Rosada",
    "Rynchops niger": "Rayador Americano",
    "Theristicus caudatus": "Bandurria Común",
    "Vultur gryphus": "Cóndor Andino",
}

# Funcion para preprocesar imagen según método
def preprocess_image(file_like, method: str, img_size=None) -> np.ndarray:
    if img_size is None:
        img_size = IMG_SIZE
    img = load_img(file_like, target_size=img_size)
    arr = img_to_array(img)
    if method == "VGG16":
        x = np.expand_dims(arr, axis=0)
        x = vgg16_preprocess(x)
    else:
        x = np.expand_dims(arr / 255.0, axis=0)
    return x
# Preprocesar dataset según método
def preprocess_dataset(ds, method: str, img_size=None):
    if img_size is None:
        img_size = IMG_SIZE
    if method == "VGG16":
        def _map(x, y):
            x2 = tf.numpy_function(lambda z: vgg16_preprocess(z), [x], tf.float32)
            x2.set_shape((None, img_size[0], img_size[1], 3))
            return x2, y
        return ds.map(_map)
    else:
        return ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))


def infer_model_image_size(model):
    """Intentar inferir (H, W) desde un modelo/objeto invocable.
    - Para tf.keras.Model, usa input_shape o inputs.
    - Para TFSMLayer o capas personalizadas, intenta buscar input_spec.
    Si no puede inferir, devuelve el valor por defecto IMG_SIZE.
    """
    try:
        shape = None
        # Keras Model
        if hasattr(model, "input_shape") and model.input_shape is not None:
            shape = model.input_shape
        elif hasattr(model, "inputs") and getattr(model, "inputs"):
            shape = model.inputs[0].shape
        # TFSMLayer y otros: intentar input_spec
        elif hasattr(model, "input_spec"):
            spec = getattr(model, "input_spec")
            if isinstance(spec, (list, tuple)):
                spec = spec[0]
            if hasattr(spec, "shape"):
                shape = spec.shape

        if shape is None:
            return IMG_SIZE

        # Normalizar a tupla de ints/None
        if hasattr(shape, "as_list"):
            shape_tuple = tuple(None if x is None else int(x) for x in shape.as_list())
        else:
            shape_tuple = tuple(shape)

        # Buscar H,W en la tupla. Soporta formatos (None,H,W,C) o (H,W,C) o (...,H,W,C)
        if len(shape_tuple) >= 3:
            # preferir los últimos 3 elementos
            H = shape_tuple[-3]
            W = shape_tuple[-2]
            if isinstance(H, int) and isinstance(W, int):
                return (H, W)
    except Exception:
        pass
    return IMG_SIZE
# Predecir probabilidades y asegurar formato
def predict_proba(x: np.ndarray) -> np.ndarray:
    preds = model(x)
    if isinstance(preds, dict):
        preds = list(preds.values())[0]
    if tf.is_tensor(preds):
        preds = preds.numpy()
    # Asegurar probabilidades (softmax) si no está normalizado
    if preds.ndim == 2:
        row_sums = preds.sum(axis=1, keepdims=True)
        if not np.all(np.isfinite(row_sums)) or np.any((row_sums < 0.99) | (row_sums > 1.01)):
            preds = tf.nn.softmax(preds, axis=1).numpy()
    return preds
# Buscar imagen de referencia para una clase
def find_reference_image(class_name: str):
    """Buscar una imagen de referencia para una clase.

    Estrategia (en este orden):
    1. Buscar carpeta exacta `datos/{split}/{class_name}` en splits ['valid','test','train'].
    2. Si `class_name` tiene formato 'class_{i}', mapear al i-ésimo subdirectorio ordenado en `datos/{split}`.
    3. Buscar en carpetas del modelo (`MODEL_PATH`) en subdirs comunes: assets, examples, reference.
    4. Buscar por coincidencia parcial de nombre de archivo en `datos/` y `modelos/` (filename contiene clase o nombre común).
    Devuelve ruta absoluta de la primera imagen encontrada o None si no hay coincidencias.
    """
    img_exts = (".jpg", ".jpeg", ".png")
    # 1) intentos directos en datos/{split}/{class_name}
    for split in ["valid", "test", "train"]:
        candidate_dir = os.path.join("datos", split, class_name)
        if os.path.isdir(candidate_dir):
            for fname in sorted(os.listdir(candidate_dir)):
                if fname.lower().endswith(img_exts):
                    return os.path.join(candidate_dir, fname)

    # 2) si la etiqueta tiene formato class_{i}, mapear al i-ésimo subdirectorio en datos/{split}
    import re
    m = re.match(r"class_(\d+)$", class_name)
    if m:
        idx = int(m.group(1))
        for split in ["valid", "train", "test"]:
            root = os.path.join("datos", split)
            if os.path.isdir(root):
                try:
                    subs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
                    if 0 <= idx < len(subs):
                        candidate_dir = os.path.join(root, subs[idx])
                        for fname in sorted(os.listdir(candidate_dir)):
                            if fname.lower().endswith(img_exts):
                                return os.path.join(candidate_dir, fname)
                except Exception:
                    pass

    # 3) buscar en el directorio del modelo (ej.: modelos/<name>/assets, examples, reference)
    try:
        model_root = MODEL_PATH if 'MODEL_PATH' in globals() else None
        if model_root:
            for sub in ("assets", "examples", "reference", "refs", "imgs"):
                cand = os.path.join(model_root, sub)
                if os.path.isdir(cand):
                    # buscar archivo que contenga el nombre de clase o el nombre común
                    for fname in sorted(os.listdir(cand)):
                        low = fname.lower()
                        if low.endswith(img_exts):
                            if class_name.lower() in low:
                                return os.path.join(cand, fname)
                            # también buscar por nombre común
                            common = COMMON_NAMES.get(class_name, "").lower()
                            if common and common in low:
                                return os.path.join(cand, fname)
                    # si no hay match por nombre, devolver la primera imagen como fallback
                    for fname in sorted(os.listdir(cand)):
                        if fname.lower().endswith(img_exts):
                            return os.path.join(cand, fname)
    except Exception:
        pass

    # 4) búsqueda amplia: archivos en datos/ que contengan el nombre de clase o nombre común
    search_roots = [os.path.join("datos", p) for p in ("valid", "train", "test")] + [os.path.join("modelos", d) for d in os.listdir("modelos") if os.path.isdir(os.path.join("modelos", d))]
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, files in os.walk(root):
            for fname in files:
                low = fname.lower()
                if not low.endswith(img_exts):
                    continue
                if class_name.lower() in low:
                    return os.path.join(dirpath, fname)
                common = COMMON_NAMES.get(class_name, "").lower()
                if common and common in low:
                    return os.path.join(dirpath, fname)

    return None

# ============================
# Utilidades de evaluación
# ============================
def eval_on_dataset(model_layer, ds):
    Y_true, Y_pred = [], []
    for x_batch, y_batch in ds:
        pred = model_layer(x_batch)
        if isinstance(pred, dict):
            pred = list(pred.values())[0]
        if tf.is_tensor(pred):
            pred = pred.numpy()
        # Softmax si es necesario
        if pred.ndim == 2:
            row_sums = pred.sum(axis=1, keepdims=True)
            if not np.all(np.isfinite(row_sums)) or np.any((row_sums < 0.99) | (row_sums > 1.01)):
                pred = tf.nn.softmax(pred, axis=1).numpy()
        Y_pred.append(pred)
        Y_true.append(y_batch.numpy())
    Y_true = np.concatenate(Y_true)
    Y_pred = np.concatenate(Y_pred)
    y_true = Y_true.astype(int)
    y_pred = np.argmax(Y_pred, axis=1)
    # Top-3 accuracy
    top3 = np.mean([
        y_true[i] in np.argsort(Y_pred[i])[-3:]
        for i in range(len(y_true))
    ])
    acc = np.mean(y_true == y_pred)
    return acc, top3, y_true, y_pred, Y_pred

# Cachear evaluación completa por (modelo, split, preprocesamiento)
@st.cache_data(show_spinner=False)
def evaluate_model_path(model_path: str, split: str, preprocess: str):
    mlayer = load_model_generic(model_path)
    # Infer model input size and use it when building the dataset / preprocessing
    try:
        inferred_size = infer_model_image_size(mlayer)
    except Exception:
        inferred_size = IMG_SIZE
    ds = get_dataset(split, img_size=inferred_size)
    ds = preprocess_dataset(ds, preprocess, img_size=inferred_size)
    acc, top3, y_true, y_pred, Y_pred = eval_on_dataset(mlayer, ds)
    labels = load_class_names(model_path)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    return {
        "acc": acc,
        "top3": top3,
        "y_true": y_true,
        "y_pred": y_pred,
        "Y_pred": Y_pred,
        "labels": labels,
        "cm": cm,
        "report": report,
    }

# ============================
# Dashboard: barra lateral + secciones
# ============================
st.sidebar.title("⚙️ Panel de control")
nav = st.sidebar.radio("Navegación", ["🔍 Predicción", "📊 Evaluación"], index=0)

# Configuración común
st.sidebar.markdown("---")
st.sidebar.subheader("Acerca del modelo")
# Selector de modelo
model_name = st.sidebar.selectbox("Modelo", options=list(AVAILABLE_MODELS.keys()), index=0)
MODEL_PATH = AVAILABLE_MODELS[model_name]
model = None
classes = []
classes_source = "desconocida"
if HAS_TF:
    try:
        model = load_model_generic(MODEL_PATH)
    except Exception as e:
        st.sidebar.error(f"No se pudo cargar el modelo: {e}")
        model = None
    try:
        classes = load_class_names(MODEL_PATH)
    except Exception:
        classes = []
else:
    # Sin TensorFlow no podemos cargar el modelo; intentar leer clases desde archivo si existe
    try:
        # Intentar cargar clases desde archivos al lado del modelo (sin TF)
        candidates = [
            os.path.join(MODEL_PATH, "classes.txt"),
            os.path.join(MODEL_PATH, "labels.txt"),
            os.path.join(MODEL_PATH, "class_indices.json"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                if path.endswith('.json'):
                    import json
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        classes = [k for k, _ in sorted(data.items(), key=lambda kv: kv[1])]
                    elif isinstance(data, list):
                        classes = data
                    break
                else:
                    with open(path, 'r', encoding='utf-8') as f:
                        lines = [ln.strip() for ln in f if ln.strip()]
                    if lines:
                        classes = lines
                        break
    except Exception:
        classes = []
classes_source = "archivo del modelo" if any(
    os.path.isfile(os.path.join(MODEL_PATH, name)) for name in ["classes.txt", "labels.txt", "class_indices.json"]
) else ("dataset" if classes else "desconocida")

st.sidebar.write(f"Seleccionado: `{model_name}`")
st.sidebar.write(f"Ruta: `{MODEL_PATH}`")
st.sidebar.write(f"Clases: {len(classes)}")
st.sidebar.caption(f"Fuente de clases: {classes_source}")

# Si no hay TensorFlow/PennyLane en este entorno, mostrar aviso claro
if not HAS_TF or not HAS_PENNYLANE:
    msg_lines = []
    if not HAS_TF:
        msg_lines.append("TensorFlow no está instalado en este despliegue; la evaluación y carga de modelos .keras/.h5 estará deshabilitada.")
    if not HAS_PENNYLANE:
        msg_lines.append("PennyLane no está instalado; los modelos híbridos cuánticos no se podrán ejecutar aquí.")
    st.sidebar.warning("\n".join(msg_lines))

# Diagnóstico opcional de la arquitectura y capas
if st.sidebar.checkbox("Mostrar diagnóstico de modelo", value=False):
    try:
        diag = get_model_diagnostics(model)
        with st.expander("Diagnóstico del modelo (capas)"):
            for d in diag:
                st.write("—" * 40)
                st.write(f"Nombre: {d.get('name')}")
                st.write(f"Tipo: {d.get('class')}")
                st.write(f"input_shape: {d.get('input_shape')}")
                st.write(f"output_shape: {d.get('output_shape')}")
                if d.get("extra"):
                    st.write("Extra:")
                    st.json(d.get("extra"))
    except Exception as e:
        st.warning(f"No se pudo generar el diagnóstico: {e}")

if nav.startswith("🔍"):
    # Controles de predicción
    st.sidebar.markdown("---")
    st.sidebar.subheader("Opciones de predicción")
    prep_method = st.sidebar.selectbox("Preprocesamiento", ["x/255", "VGG16"], index=0)
    num_classes = len(classes)
    # Si no hay suficientes clases para un slider (min == max), evitar el slider y
    # usar un valor por defecto de 1. Esto evita StreamlitAPIException cuando
    # num_classes <= 1 (por ejemplo en despliegues sin TF donde no se pudieron
    # leer las clases del modelo/dataset).
    if num_classes <= 1:
        top_k = 1
        st.sidebar.info("Top-K deshabilitado: menos de 2 clases disponibles.")
    else:
        max_k = min(10, num_classes)
        top_k = st.sidebar.slider("Top-K", min_value=1, max_value=max_k, value=min(5, max_k))
    show_ref = st.sidebar.checkbox("Mostrar imagen de referencia", value=True)

    st.subheader("Sube una imagen de un ave")
    up = st.file_uploader("Elige una imagen (JPG/PNG)", type=["jpg", "jpeg", "png"])
    # Variable que indica si hubo predicción válida
    predicted = False
    sel_class = None
    if up is not None:
        if not HAS_TF or load_img is None:
            st.error("TensorFlow/Keras no están disponibles en este despliegue; no se pueden realizar predicciones aquí. Cargue el modelo localmente o despliegue en un runtime compatible.")
            st.info("Opciones para habilitar predicción: 1) Ejecutar la app localmente con Python 3.10 y TensorFlow/PennyLane instalados, 2) Desplegar un microservicio de inferencia (Docker) con TF/PennyLane y apuntar la UI a ese endpoint.")
        else:
            c1, c2 = st.columns([1, 1])
            with c1:
                st.image(up, caption="Imagen cargada", use_container_width=True)
            # Intentar preprocesar y predecir; cualquier error se captura y muestra
            try:
                x = preprocess_image(up, prep_method)
            except Exception as e:
                st.error(f"Error al preprocesar la imagen: {e}")
                x = None
            if x is not None:
                try:
                    probs_all = predict_proba(x)
                    # validar la salida
                    if probs_all is None:
                        raise ValueError("predict_proba devolvió None")
                    probs = np.asarray(probs_all)
                    # Si la salida tiene batch dim, tomar el primer elemento
                    if probs.ndim == 2 and probs.shape[0] >= 1:
                        probs = probs[0]
                    if probs.ndim != 1:
                        raise ValueError(f"Salida de predict_proba con shape inesperado: {probs.shape}")
                    predicted = True
                except Exception as e:
                    st.error(f"Error al ejecutar la predicción: {e}")
                    st.exception(e)

    # Si hubo predicción válida, construir y mostrar Top-K y detalles
    if predicted:
        # Preparar lista de etiquetas: preferir `classes`, sino intentar cargar desde archivos
        if classes and len(classes) > 0:
            cls_list = classes
            inferred_from = "model/classes file"
        else:
            # 1) intentar load_class_names (intenta clases al lado del modelo o dataset)
            try:
                inferred = load_class_names(MODEL_PATH)
            except Exception:
                inferred = []
            # 2) si sigue vacío, intentar leer las carpetas en datos/valid o datos/train
            if not inferred:
                for candidate in [os.path.join("datos", "valid"), os.path.join("datos", "train")]:
                    if os.path.isdir(candidate):
                        try:
                            inferred = sorted([d for d in os.listdir(candidate) if os.path.isdir(os.path.join(candidate, d))])
                            if inferred:
                                break
                        except Exception:
                            inferred = []
            if inferred:
                cls_list = inferred
                inferred_from = "dataset folders (datos/)"
                st.info(f"Usando nombres de clases inferidos desde {inferred_from}.")
            else:
                # último recurso: construir etiquetas genéricas class_0, class_1, ... según la dimensión de probs
                try:
                    n_out = int(probs.shape[-1])
                except Exception:
                    n_out = int(getattr(probs, 'size', 0))
                cls_list = [f"class_{i}" for i in range(n_out)]
                inferred_from = "generated labels"
                st.warning("No se encontraron nombres de clases; usando etiquetas generadas (class_0, class_1, ...).")

    # Sugerencias Top-3
    suggest_k = min(3, len(cls_list))
    try:
        top_idx_sorted = np.argsort(probs)[-suggest_k:][::-1]
    except Exception:
        top_idx_sorted = np.arange(min(suggest_k, len(cls_list)))
    default_idx = int(top_idx_sorted[0]) if len(top_idx_sorted) > 0 else 0
    default_class = cls_list[default_idx]
    default_conf = float(probs[default_idx]) if probs.size > default_idx else 0.0

    # Construir opciones legibles
    options = []
    idx_map = {}
    for idx in top_idx_sorted:
        sci = cls_list[int(idx)]
        com = COMMON_NAMES.get(sci, sci)
        label = f"{com} ({sci}) — {probs[int(idx)]*100:.1f}%"
        options.append(label)
        idx_map[label] = int(idx)

        with c2:
            st.metric("Predicción sugerida", COMMON_NAMES.get(default_class, default_class))
            st.metric("Confianza", f"{default_conf*100:.2f}%")
            st.markdown("### Opciones Top-3")
            cols = st.columns(suggest_k)
            sel_key = f"sel_idx_pred_{model_name}"
            selected_idx = st.session_state.get(sel_key, default_idx)
            for j, idx in enumerate(top_idx_sorted):
                idx = int(idx)
                sci = cls_list[idx]
                com = COMMON_NAMES.get(sci, sci)
                ref = find_reference_image(sci)
                with cols[j]:
                    if ref:
                        st.image(ref, caption=f"{com} — {probs[idx]*100:.1f}%", use_container_width=True)
                    else:
                        st.caption(f"{com} — {probs[idx]*100:.1f}% (sin imagen de referencia)")
                    if st.button("Elegir", key=f"btn_choose_{model_name}_{int(idx)}"):
                        st.session_state[sel_key] = int(idx)
                        selected_idx = int(idx)
            try:
                default_radio_index = list(top_idx_sorted).index(selected_idx)
            except ValueError:
                default_radio_index = 0
            choice = st.radio("¿Cuál encaja mejor?", options=options, index=default_radio_index)
            sel_idx = idx_map.get(choice, int(top_idx_sorted[0]))
            if sel_idx != selected_idx:
                st.session_state[sel_key] = int(sel_idx)
            sel_class = cls_list[int(sel_idx)]
            sel_common = COMMON_NAMES.get(sel_class, sel_class)
            sel_conf = float(probs[int(sel_idx)])

            st.write(f"Nombre científico: **{sel_class}**")
            st.write(f"Nombre común: **{sel_common}**")
            desc = DESCRIPCIONES.get(sel_class, "Descripción no disponible.")
            st.write(desc)

        # Imagen de referencia para la selección
        if show_ref and sel_class:
            ref_path = find_reference_image(sel_class)
            if ref_path:
                st.image(ref_path, caption=f"Ejemplo de {sel_class}", use_container_width=True)

        # Top-K gráfico: si estamos usando etiquetas inferidas/desde dataset, preferimos mostrar Top-3
        if inferred_from != "model/classes file":
            k = min(3, len(cls_list))
        else:
            k = min(top_k, len(cls_list))
        top_k_idx = np.argsort(probs)[-k:][::-1]
        top_labels = [COMMON_NAMES.get(cls_list[int(i)], cls_list[int(i)]) for i in top_k_idx]
        top_df = pd.DataFrame({
            "Clase (común)": top_labels,
            "Probabilidad": [float(probs[int(i)]) for i in top_k_idx],
        })
        st.bar_chart(top_df.set_index("Clase (común)"))

else:
    # Controles de evaluación en la barra lateral
    st.sidebar.markdown("---")
    st.sidebar.subheader("Opciones de evaluación")
    split = st.sidebar.selectbox("Conjunto", options=["valid", "test"], index=0)
    prep_method_eval = st.sidebar.selectbox("Preprocesamiento", ["x/255", "VGG16"], index=0)
    eval_button = st.sidebar.button("Calcular métricas", use_container_width=True)
    st.sidebar.subheader("Comparación de modelos")
    compare_models = st.sidebar.multiselect("Modelos a comparar", options=list(AVAILABLE_MODELS.keys()), default=[model_name])
    compare_button = st.sidebar.button("Comparar modelos", use_container_width=True)
    # establecer sección de evaluación
    st.subheader("Evaluación del modelo")
    if eval_button:
        if not HAS_TF:
            st.error("TensorFlow no está disponible; la evaluación está deshabilitada en este despliegue.")
        else:
            ds = get_dataset(split)
            ds = preprocess_dataset(ds, prep_method_eval)
        labels = classes
        Y_true, Y_pred = [], []
        progress = st.progress(0, text="Calculando predicciones...")
        # Intentar estimar número total de lotes
        try:
            total_batches = int(tf.data.experimental.cardinality(ds).numpy())
        except Exception:
            total_batches = None
        for batch_idx, (x_batch, y_batch) in enumerate(ds):
            pred = predict_proba(x_batch)
            Y_pred.append(pred)
            Y_true.append(y_batch.numpy())
            if total_batches and total_batches > 0:
                progress.progress((batch_idx + 1) / total_batches)
        progress.progress(1.0)
        progress.empty()

        # Aplanar y calcular métricas
        Y_true = np.concatenate(Y_true)  # Etiquetas enteras (shape: [N,])
        Y_pred = np.concatenate(Y_pred)  # Probabilidades (shape: [N, C])
        y_true = Y_true.astype(int)      # Etiquetas ya son índices de clase
        y_pred = np.argmax(Y_pred, axis=1)

        st.markdown("### Métricas generales")
        acc = np.mean(y_true == y_pred)
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{acc*100:.2f}%")
        c2.metric("Imágenes", str(len(y_true)))
        c3.metric("Clases", str(len(labels)))

        st.markdown("### Matriz de confusión")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_ylabel("Real")
        ax.set_xlabel("Predicción")
        st.pyplot(fig, use_container_width=True)

        st.markdown("### Reporte de clasificación")
        report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        st.dataframe(
            df_report.style.format({
                "precision": "{:.2f}",
                "recall": "{:.2f}",
                "f1-score": "{:.2f}",
                "support": "{:.0f}",
            }),
            use_container_width=True,
        )

        # Descargar CSV
        csv = df_report.to_csv().encode("utf-8")
        st.download_button(
            label=f"Descargar reporte ({split}).csv",
            data=csv,
            file_name=f"reporte_{split}.csv",
            mime="text/csv",
        )
    # ============================
    # Comparación de modelos
    # ============================
    if compare_button and compare_models:
        st.markdown("## Comparación de modelos")
        # Resumen rápido
        rows = []
        progress = st.progress(0, text="Evaluando modelos seleccionados...")
        n_models = len(compare_models)
        eval_results = {}
        for idx, mname in enumerate(compare_models):
            mpath = AVAILABLE_MODELS[mname]
            if not HAS_TF:
                st.error("TensorFlow no está disponible; no se puede evaluar modelos en este despliegue.")
                res = None
            else:
                res = evaluate_model_path(mpath, split, prep_method_eval)
            eval_results[mname] = res
            rows.append({
                "Modelo": mname,
                "Accuracy": res["acc"] if res is not None else None,
                "Top-3": res["top3"] if res is not None else None,
                "Imágenes": len(res["y_true"]) if res is not None else 0,
            })
            progress.progress((idx + 1) / n_models)
        progress.empty()

        df_cmp = pd.DataFrame(rows)
        # Asegurar que las columnas numéricas sean numéricas (None -> NaN)
        for col in ("Accuracy", "Top-3"):
            if col in df_cmp.columns:
                df_cmp[col] = pd.to_numeric(df_cmp[col], errors="coerce")

        # Crear una versión segura para mostrar (convertir NaN a 'N/A' y formatear porcentajes)
        df_display = df_cmp.copy()
        if "Accuracy" in df_display.columns:
            df_display["Accuracy"] = df_display["Accuracy"].apply(lambda v: f"{v:.2%}" if pd.notna(v) else "N/A")
        if "Top-3" in df_display.columns:
            df_display["Top-3"] = df_display["Top-3"].apply(lambda v: f"{v:.2%}" if pd.notna(v) else "N/A")

        st.dataframe(df_display, use_container_width=True)
        # Gráfico de barras de accuracy
        chart_df = df_cmp.set_index("Modelo")["Accuracy"]
        st.bar_chart(chart_df)

        # Detalle por modelo (igual que evaluación individual)
        st.markdown("### Detalle por modelo")
        for mname in compare_models:
            res = eval_results[mname]
            if res is None:
                st.warning(f"No hay resultados para {mname} (TensorFlow no disponible en este despliegue).")
                continue
            labels_m = res["labels"]
            y_true_m = res["y_true"]
            y_pred_m = res["y_pred"]
            cm_m = res["cm"]
            report_m = res["report"]

            with st.expander(f"{mname}", expanded=False):
                c1, c2, c3 = st.columns(3)
                c1.metric("Accuracy", f"{res['acc']*100:.2f}%")
                c2.metric("Top-3", f"{res['top3']*100:.2f}%")
                c3.metric("Imágenes", str(len(y_true_m)))

                st.markdown("**Matriz de confusión**")
                fig_m, ax_m = plt.subplots(figsize=(6, 6))
                sns.heatmap(cm_m, annot=True, fmt="d", cmap="Blues", xticklabels=labels_m, yticklabels=labels_m, ax=ax_m)
                ax_m.set_ylabel("Real")
                ax_m.set_xlabel("Predicción")
                st.pyplot(fig_m, use_container_width=True)

                st.markdown("**Reporte de clasificación**")
                df_rep_m = pd.DataFrame(report_m).transpose()
                st.dataframe(
                    df_rep_m.style.format({
                        "precision": "{:.2f}",
                        "recall": "{:.2f}",
                        "f1-score": "{:.2f}",
                        "support": "{:.0f}",
                    }),
                    use_container_width=True,
                )

                csv_m = df_rep_m.to_csv().encode("utf-8")
                st.download_button(
                    label=f"Descargar reporte ({mname} · {split}).csv",
                    data=csv_m,
                    file_name=f"reporte_{mname}_{split}.csv",
                    mime="text/csv",
                )
    else:
        st.info("Selecciona el conjunto en la barra lateral y pulsa 'Calcular métricas'.")

# Estilos suaves
st.markdown(
    """
    <style>
    .stMetric { text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)
