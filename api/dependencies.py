"""
Dependencias compartidas del API.

Por qué un módulo separado: permite inyectar dependencias en tests
sin reimportar el app completo (patrón FastAPI Dependency Injection).
"""

import sys
import os

# Asegurar que src/ esté en el path antes de cualquier import de portopt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
