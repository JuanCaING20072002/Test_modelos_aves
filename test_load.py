#!/usr/bin/env python
"""
Script de prueba para validar la carga de un archivo .keras que contiene una capa PennyLane KerasLayer.
Usa como argumento la ruta al archivo .keras, por ejemplo:
    python test_load.py modelos\hybrid_quantum_10clases_acc55_data46.keras
Si no se pasa argumento, usará la ruta por defecto indicada en el script.
Imprime versiones, intenta importar KerasLayer y carga el modelo pasando custom_objects si es posible.
"""
import sys
import traceback

DEFAULT_PATH = r"modelos\hybrid_quantum_10clases_acc55_data46.keras"

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    print("\n== Entorno y versiones ==")
    try:
        import tensorflow as tf
        print("TensorFlow:", tf.__version__)
    except Exception as e:
        print("No se pudo importar TensorFlow:", e)

    try:
        import pennylane as qml
        print("PennyLane:", qml.__version__)
    except Exception as e:
        print("No se pudo importar PennyLane:", e)

    try:
        from pennylane.qnn.keras import KerasLayer
        print("KerasLayer import OK")
    except Exception as e:
        print("KerasLayer import falló:", e)
        KerasLayer = None

    print(f"\nIntentando cargar modelo: {path}\n")
    try:
        from tensorflow import keras
        custom = {}
        # Construir un objeto fallback para `KerasLayer` que implemente `from_config`
        # y pueda crear una instancia real de pennylane.qnn.keras.KerasLayer en tiempo de carga.
        def build_keraslayer_fallback():
            try:
                import pennylane as _qml
                from pennylane.qnn.keras import KerasLayer as _PLKerasLayer

                class KerasLayerFallback:
                    @classmethod
                    def from_config(cls, config, custom_objects=None):
                        # Intentar inferir n_qubits y n_layers desde weight_shapes si está disponible
                        ws = None
                        try:
                            ws = config.get("weight_shapes", {}).get("weights")
                        except Exception:
                            ws = None

                        n_qubits = 4
                        n_layers = 2
                        if isinstance(ws, (list, tuple)) and len(ws) >= 2:
                            try:
                                n_layers = int(ws[0])
                                n_qubits = int(ws[1])
                            except Exception:
                                pass

                        output_dim = config.get("output_dim", None) or n_qubits

                        dev = _qml.device("default.qubit", wires=n_qubits)

                        def circuit(inputs, weights):
                            _qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
                            _qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                            out = []
                            for i in range(int(output_dim)):
                                wire = i % n_qubits
                                out.append(_qml.expval(_qml.PauliZ(wire)))
                            return out

                        qnode = _qml.QNode(circuit, dev, interface="tf")

                        # Construir la KerasLayer real usando la información de config cuando sea posible
                        weight_shapes = config.get("weight_shapes", None)
                        try:
                            real_layer = _PLKerasLayer(qnode, weight_shapes=weight_shapes, output_dim=int(output_dim))
                            return real_layer
                        except Exception:
                            # último recurso: intentar sin output_dim
                            return _PLKerasLayer(qnode, weight_shapes=weight_shapes)

                return KerasLayerFallback
            except Exception as e:
                print("No se pudo construir fallback KerasLayer:", e)
                return None

        fallback_cls = build_keraslayer_fallback()
        # Añadir al custom_objects un fallback que implemente from_config y cree la capa real.
        if fallback_cls is not None:
            custom["KerasLayer"] = fallback_cls
            custom["pennylane.qnn.keras.KerasLayer"] = fallback_cls
            print("Fallback KerasLayer añadido a custom_objects (heurístico).")
        elif KerasLayer is not None:
            custom["KerasLayer"] = KerasLayer
            custom["pennylane.qnn.keras.KerasLayer"] = KerasLayer
        # Intento de carga con custom_objects primero (si tenemos KerasLayer)
        try:
            model = keras.models.load_model(path, compile=False, custom_objects=custom if custom else None)
        except Exception as e1:
            print("Carga con custom_objects falló, intentando carga sin custom_objects...\n")
            traceback.print_exc()
            # Intentar de nuevo sin custom_objects
            model = keras.models.load_model(path, compile=False)

        print("\nModelo cargado con éxito.")
        try:
            model.summary()
        except Exception:
            print("(No se pudo imprimir model.summary())")
    except Exception as e:
        print("Error cargando el modelo:")
        traceback.print_exc()
        sys.exit(2)

if __name__ == "__main__":
    main()
