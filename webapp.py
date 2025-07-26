"""
Auto Parts Finder USA - Versión de Debug para identificar problemas
Esta versión incluye logging detallado para identificar exactamente qué está fallando
"""
import os
import asyncio
import hashlib
import io
import json
import logging
import re
import secrets
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import httpx
import numpy as np
import psutil

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field

# ================================================================
# CONFIGURACIÓN CON DEBUG DETALLADO
# ================================================================

class Settings(BaseSettings):
    """Configuración con debug mejorado"""
    
    PROJECT_NAME: str = "Auto Parts Finder USA - DEBUG"
    VERSION: str = "1.0.0-debug"
    
    # Servidor
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=int(os.environ.get("PORT", 8000)), env="PORT")
    ENVIRONMENT: str = Field(default="production", env="ENVIRONMENT")
    
    # SerpAPI - Con validación detallada
    SERPAPI_KEY: str = Field(default="", env="SERPAPI_KEY")
    SERPAPI_TIMEOUT: int = Field(default=30, env="SERPAPI_TIMEOUT")
    
    @property
    def serpapi_configured(self) -> bool:
        if not self.SERPAPI_KEY:
            return False
        if self.SERPAPI_KEY in ["", "TU_SERPAPI_KEY_AQUI", "your_key_here"]:
            return False
        if len(self.SERPAPI_KEY) < 20:  # Las claves SerpAPI son largas
            return False
        return True
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()

# ================================================================
# LOGGING DETALLADO
# ================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

# ================================================================
# CLIENTE SERPAPI SIMPLIFICADO PARA DEBUG
# ================================================================

class DebugSerpAPIClient:
    """Cliente SerpAPI simplificado para debugging"""
    
    def __init__(self):
        self.api_key = settings.SERPAPI_KEY
        self.base_url = "https://serpapi.com/search"
        logger.info(f"🔧 Inicializando SerpAPI Client")
        logger.info(f"🔑 API Key configurada: {'✅ SÍ' if self.api_key else '❌ NO'}")
        if self.api_key:
            logger.info(f"🔑 Longitud de API Key: {len(self.api_key)} caracteres")
            logger.info(f"🔑 Primeros 8 caracteres: {self.api_key[:8]}...")
            logger.info(f"🔑 Últimos 4 caracteres: ...{self.api_key[-4:]}")
        
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.SERPAPI_TIMEOUT),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
    
    async def test_connection(self) -> Dict[str, Any]:
        """Prueba la conexión a SerpAPI con información detallada"""
        logger.info("🧪 Iniciando prueba de conexión a SerpAPI...")
        
        if not settings.serpapi_configured:
            return {
                "success": False,
                "error": "API key no configurada correctamente",
                "details": {
                    "key_exists": bool(self.api_key),
                    "key_length": len(self.api_key) if self.api_key else 0,
                    "key_valid_format": len(self.api_key) >= 20 if self.api_key else False
                }
            }
        
        # Prueba simple con una búsqueda básica
        params = {
            "api_key": self.api_key,
            "engine": "google",
            "q": "test",
            "num": 1
        }
        
        try:
            logger.info(f"🌐 Realizando request a: {self.base_url}")
            logger.info(f"📊 Parámetros: engine=google, q=test, num=1")
            
            response = await self.client.get(self.base_url, params=params)
            
            logger.info(f"📈 Status Code: {response.status_code}")
            logger.info(f"⏱️ Tiempo de respuesta: {response.elapsed}")
            
            if response.status_code == 200:
                data = response.json()
                logger.info("✅ Respuesta exitosa de SerpAPI")
                
                if "error" in data:
                    logger.error(f"❌ Error en respuesta SerpAPI: {data['error']}")
                    return {
                        "success": False,
                        "error": f"SerpAPI Error: {data['error']}",
                        "details": data
                    }
                else:
                    logger.info("🎉 SerpAPI funcionando correctamente")
                    return {
                        "success": True,
                        "message": "SerpAPI configurado correctamente",
                        "search_metadata": data.get("search_metadata", {}),
                        "results_count": len(data.get("organic_results", []))
                    }
            
            elif response.status_code == 401:
                logger.error("🔐 Error 401: API key inválida")
                return {
                    "success": False,
                    "error": "API key inválida (401 Unauthorized)",
                    "details": {
                        "status_code": 401,
                        "suggestion": "Verifica tu API key en https://serpapi.com/manage-api-key"
                    }
                }
            
            elif response.status_code == 429:
                logger.error("🚫 Error 429: Rate limit excedido")
                return {
                    "success": False,
                    "error": "Rate limit excedido (429)",
                    "details": {
                        "status_code": 429,
                        "suggestion": "Espera unos minutos o verifica tu plan SerpAPI"
                    }
                }
            
            else:
                logger.error(f"🚨 Error HTTP {response.status_code}")
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "details": {
                        "status_code": response.status_code,
                        "response_text": response.text[:500]
                    }
                }
        
        except httpx.TimeoutException:
            logger.error("⏰ Timeout al conectar con SerpAPI")
            return {
                "success": False,
                "error": "Timeout al conectar con SerpAPI",
                "details": {
                    "timeout": settings.SERPAPI_TIMEOUT,
                    "suggestion": "Verifica conectividad de red"
                }
            }
        
        except Exception as e:
            logger.error(f"💥 Error inesperado: {str(e)}")
            return {
                "success": False,
                "error": f"Error inesperado: {str(e)}",
                "details": {
                    "exception_type": type(e).__name__
                }
            }
    
    async def search_parts(self, query: str) -> Dict[str, Any]:
        """Búsqueda simplificada para testing"""
        logger.info(f"🔍 Iniciando búsqueda: '{query}'")
        
        # Primero verificar configuración
        test_result = await self.test_connection()
        if not test_result["success"]:
            return test_result
        
        # Realizar búsqueda real
        params = {
            "api_key": self.api_key,
            "engine": "google_shopping",
            "q": f"{query} auto parts",
            "location": "United States",
            "gl": "us",
            "hl": "en",
            "num": 5
        }
        
        try:
            response = await self.client.get(self.base_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if "error" in data:
                    return {
                        "success": False,
                        "error": data["error"],
                        "query": query
                    }
                
                shopping_results = data.get("shopping_results", [])
                
                return {
                    "success": True,
                    "query": query,
                    "results_count": len(shopping_results),
                    "results": shopping_results[:3],  # Solo los primeros 3 para debug
                    "search_metadata": data.get("search_metadata", {})
                }
            
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "query": query
                }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    async def close(self):
        await self.client.aclose()

# ================================================================
# APLICACIÓN FASTAPI PARA DEBUG
# ================================================================

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cliente global
debug_client: Optional[DebugSerpAPIClient] = None

@app.on_event("startup")
async def startup_event():
    global debug_client
    
    logger.info("🚀 INICIANDO AUTO PARTS FINDER - MODO DEBUG")
    logger.info("=" * 60)
    
    # Información del entorno
    logger.info(f"🌍 Entorno: {settings.ENVIRONMENT}")
    logger.info(f"🖥️ Host: {settings.HOST}")
    logger.info(f"🔌 Puerto: {settings.PORT}")
    
    # Variables de entorno críticas
    logger.info("🔧 VERIFICANDO VARIABLES DE ENTORNO:")
    all_env_vars = dict(os.environ)
    
    serpapi_key = all_env_vars.get("SERPAPI_KEY", "")
    logger.info(f"  📋 SERPAPI_KEY exists: {'✅' if serpapi_key else '❌'}")
    if serpapi_key:
        logger.info(f"  📋 SERPAPI_KEY length: {len(serpapi_key)}")
        logger.info(f"  📋 SERPAPI_KEY preview: {serpapi_key[:8]}...{serpapi_key[-4:]}")
    
    logger.info(f"  📋 PORT: {all_env_vars.get('PORT', 'Not set')}")
    logger.info(f"  📋 HOST: {all_env_vars.get('HOST', 'Not set')}")
    logger.info(f"  📋 ENVIRONMENT: {all_env_vars.get('ENVIRONMENT', 'Not set')}")
    
    # Contar todas las variables de entorno
    logger.info(f"  📋 Total env vars: {len(all_env_vars)}")
    
    # Variables que empiecen con SERP
    serp_vars = {k: v for k, v in all_env_vars.items() if k.startswith("SERP")}
    logger.info(f"  📋 SERP* variables: {list(serp_vars.keys())}")
    
    # Inicializar cliente de debug
    debug_client = DebugSerpAPIClient()
    
    logger.info("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    global debug_client
    if debug_client:
        await debug_client.close()

@app.get("/", response_class=HTMLResponse)
async def root():
    """Página principal con información de debug"""
    return HTMLResponse(content=f"""
<!DOCTYPE html>
<html>
<head>
    <title>Auto Parts Finder - DEBUG MODE</title>
    <style>
        body {{ font-family: monospace; margin: 40px; background: #1a1a1a; color: #00ff00; }}
        .container {{ max-width: 800px; margin: 0 auto; }}
        .status {{ padding: 10px; margin: 10px 0; border-radius: 5px; }}
        .success {{ background: #004d00; }}
        .error {{ background: #4d0000; }}
        .info {{ background: #004d4d; }}
        pre {{ background: #333; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        button {{ padding: 10px 20px; margin: 10px 5px; font-size: 16px; cursor: pointer; }}
        .search-box {{ width: 100%; padding: 10px; font-size: 16px; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔧 AUTO PARTS FINDER - DEBUG MODE</h1>
        
        <div class="info status">
            <strong>Versión:</strong> {settings.VERSION}<br>
            <strong>Entorno:</strong> {settings.ENVIRONMENT}<br>
            <strong>SerpAPI Configurado:</strong> {'✅ SÍ' if settings.serpapi_configured else '❌ NO'}
        </div>
        
        <h2>🧪 Panel de Pruebas</h2>
        
        <button onclick="testConnection()">🔌 Probar Conexión SerpAPI</button>
        <button onclick="checkEnvironment()">🌍 Verificar Variables</button>
        <button onclick="testSearch()">🔍 Prueba de Búsqueda</button>
        
        <div style="margin: 20px 0;">
            <input type="text" class="search-box" id="searchQuery" placeholder="Escribe algo para buscar (ej: brake pads)" value="brake pads">
            <button onclick="performSearch()">🚀 Buscar Repuestos</button>
        </div>
        
        <h2>📊 Resultados:</h2>
        <pre id="results">Haz clic en un botón para ver los resultados...</pre>
        
        <h2>📚 Endpoints de Debug:</h2>
        <ul>
            <li><a href="/debug/config">/debug/config</a> - Configuración actual</li>
            <li><a href="/debug/test-serpapi">/debug/test-serpapi</a> - Prueba SerpAPI</li>
            <li><a href="/debug/environment">/debug/environment</a> - Variables de entorno</li>
            <li><a href="/debug/search?q=test">/debug/search?q=test</a> - Búsqueda de prueba</li>
        </ul>
    </div>
    
    <script>
        async function testConnection() {{
            showLoading();
            try {{
                const response = await fetch('/debug/test-serpapi');
                const data = await response.json();
                document.getElementById('results').textContent = JSON.stringify(data, null, 2);
            }} catch (error) {{
                document.getElementById('results').textContent = 'Error: ' + error.message;
            }}
        }}
        
        async function checkEnvironment() {{
            showLoading();
            try {{
                const response = await fetch('/debug/environment');
                const data = await response.json();
                document.getElementById('results').textContent = JSON.stringify(data, null, 2);
            }} catch (error) {{
                document.getElementById('results').textContent = 'Error: ' + error.message;
            }}
        }}
        
        async function testSearch() {{
            showLoading();
            try {{
                const response = await fetch('/debug/search?q=test');
                const data = await response.json();
                document.getElementById('results').textContent = JSON.stringify(data, null, 2);
            }} catch (error) {{
                document.getElementById('results').textContent = 'Error: ' + error.message;
            }}
        }}
        
        async function performSearch() {{
            const query = document.getElementById('searchQuery').value;
            if (!query) {{
                alert('Por favor escribe algo para buscar');
                return;
            }}
            
            showLoading();
            try {{
                const response = await fetch(`/debug/search?q=${{encodeURIComponent(query)}}`);
                const data = await response.json();
                document.getElementById('results').textContent = JSON.stringify(data, null, 2);
            }} catch (error) {{
                document.getElementById('results').textContent = 'Error: ' + error.message;
            }}
        }}
        
        function showLoading() {{
            document.getElementById('results').textContent = 'Cargando... ⏳';
        }}
    </script>
</body>
</html>
    """)

@app.get("/debug/config")
async def debug_config():
    """Información de configuración"""
    return {
        "project_name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "host": settings.HOST,
        "port": settings.PORT,
        "serpapi_configured": settings.serpapi_configured,
        "serpapi_key_length": len(settings.SERPAPI_KEY) if settings.SERPAPI_KEY else 0,
        "serpapi_timeout": settings.SERPAPI_TIMEOUT,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/debug/environment")
async def debug_environment():
    """Variables de entorno relevantes"""
    all_env = dict(os.environ)
    
    # Variables importantes para debugging
    important_vars = [
        "SERPAPI_KEY", "PORT", "HOST", "ENVIRONMENT", 
        "PYTHON_VERSION", "PATH", "PWD"
    ]
    
    result = {
        "total_env_vars": len(all_env),
        "important_vars": {},
        "serp_vars": {},
        "all_var_names": list(all_env.keys())
    }
    
    for var in important_vars:
        value = all_env.get(var, "NOT_SET")
        if var == "SERPAPI_KEY" and value != "NOT_SET":
            # Ocultar la clave pero mostrar info útil
            result["important_vars"][var] = {
                "exists": True,
                "length": len(value),
                "preview": f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "TOO_SHORT"
            }
        else:
            result["important_vars"][var] = value
    
    # Variables que empiecen con SERP
    for key, value in all_env.items():
        if key.startswith("SERP"):
            if key == "SERPAPI_KEY":
                result["serp_vars"][key] = {
                    "length": len(value),
                    "preview": f"{value[:8]}...{value[-4:]}" if len(value) > 12 else value
                }
            else:
                result["serp_vars"][key] = value
    
    return result

@app.get("/debug/test-serpapi")
async def debug_test_serpapi():
    """Prueba de conexión a SerpAPI"""
    global debug_client
    
    if not debug_client:
        return {
            "success": False,
            "error": "Cliente SerpAPI no inicializado"
        }
    
    result = await debug_client.test_connection()
    return result

@app.get("/debug/search")
async def debug_search(q: str = Query(..., description="Término de búsqueda")):
    """Búsqueda de prueba"""
    global debug_client
    
    if not debug_client:
        return {
            "success": False,
            "error": "Cliente SerpAPI no inicializado"
        }
    
    result = await debug_client.search_parts(q)
    return result

@app.get("/api/v1/health")
async def health_check():
    """Health check simplificado"""
    return {
        "status": "healthy" if settings.serpapi_configured else "degraded",
        "timestamp": time.time(),
        "serpapi_configured": settings.serpapi_configured,
        "version": settings.VERSION
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 INICIANDO SERVIDOR EN MODO DEBUG")
    logger.info(f"🌐 URL: http://{settings.HOST}:{settings.PORT}")
    logger.info(f"📚 Docs: http://{settings.HOST}:{settings.PORT}/docs")
    logger.info(f"🔧 Debug: http://{settings.HOST}:{settings.PORT}/debug/config")
    
    uvicorn.run(
        "webapp_debug:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level="info",
        reload=False
    )
