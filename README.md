Test_modelos_aves

Repositorio de prueba para la app Streamlit de clasificación de aves.

Contenido relevante
- `app.py` - aplicación Streamlit (interfaz para predicción y evaluación).
- `modelos/` - modelos entrenados (gestionados con Git LFS). No clonarás los binarios si no tienes Git LFS instalado.
- `test_load.py` - script de diagnóstico para probar la carga de `.keras` con PennyLane.
- `requirements.txt` - dependencias para instalar en el entorno virtual.
- `.gitattributes` - reglas para Git LFS (ya configuradas).

Cómo preparar el entorno (Windows / PowerShell)
1. Crear e activar un entorno virtual (recomendado Python 3.10):
```powershell
python -m venv .venv310
.\\.venv310\\Scripts\\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

2. Si vas a clonar este repo y quieres obtener los modelos grandes, instala Git LFS:
```powershell
winget install Git.GitLFS
# o descargar e instalar desde https://git-lfs.github.com/
git lfs install
```

3. Clona y obtén archivos LFS:
```powershell
git clone https://github.com/JuanCaING20072002/Test_modelos_aves.git
cd Test_modelos_aves
git lfs pull --all
```

Cómo ejecutar la app localmente
```powershell
# activar entorno
.\\.venv310\\Scripts\\Activate.ps1
streamlit run app.py
```

Notas sobre despliegue en Streamlit Cloud
- Streamlit Cloud instalará `requirements.txt` en el build. TensorFlow es grande y puede causar tiempos de instalación largos o fallos por timeout. Si el deploy falla, recomiendo convertir modelos a TFLite (float16/int8) o servir los modelos desde un almacenamiento externo (Releases / S3 / GCS).

Qué hacer si hay errores al clonar
- Si `git clone` no descarga los binarios, asegúrate de tener `git lfs` instalado y ejecuta `git lfs pull`.

Contacto
- Si quieres, crea un issue en este repo o escríbeme en la issue/PR y te ayudo a ajustar la app o los modelos.
