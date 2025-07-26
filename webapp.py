"""
Auto Parts Finder USA - Versión de Debug Final
Esta versión nos dirá exactamente qué está pasando con SerpAPI
"""
import os
import time
import json
import logging
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

# ================================================================
# CONFIGURACIÓN
# ================================================================

SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "")
ENVIRONMENT = os.environ.get("ENVIRONMENT", "production")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8000))

# Logging detallado
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================================================
# FUNCIONES DE VERIFICACIÓN
# ================================================================

def check_serpapi_config() -> Dict[str, Any]:
    """Verifica configuración de SerpAPI"""
    result = {
        "serpapi_key_exists": bool(SERPAPI_KEY),
        "serpapi_key_length": len(SERPAPI_KEY) if SERPAPI_KEY else 0,
        "serpapi_key_preview": "",
        "is_valid": False,
        "issues": []
    }
    
    if not SERPAPI_KEY:
        result["issues"].append("❌ SERPAPI_KEY no está configurada")
        return result
    
    if SERPAPI_KEY in ["", "TU_SERPAPI_KEY_AQUI", "your_key_here"]:
        result["issues"].append("❌ SERPAPI_KEY tiene valor placeholder")
        return result
    
    if len(SERPAPI_KEY) < 20:
        result["issues"].append(f"❌ SERPAPI_KEY muy corta ({len(SERPAPI_KEY)} caracteres)")
        return result
    
    result["serpapi_key_preview"] = f"{SERPAPI_KEY[:8]}...{SERPAPI_KEY[-4:]}"
    result["is_valid"] = True
    result["issues"].append("✅ SERPAPI_KEY configurada correctamente")
    
    return result

async def test_serpapi_simple() -> Dict[str, Any]:
    """Prueba simple de SerpAPI"""
    import httpx
    
    config_check = check_serpapi_config()
    if not config_check["is_valid"]:
        return {
            "success": False,
            "error": "Configuración inválida",
            "details": config_check
        }
    
    url = "https://serpapi.com/search"
    params = {
        "api_key": SERPAPI_KEY,
        "engine": "google",
        "q": "test",
        "num": 1
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            logger.info(f"🌐 Haciendo request a SerpAPI...")
            response = await client.get(url, params=params)
            logger.info(f"📊 Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if "error" in data:
                    return {
                        "success": False,
                        "error": f"SerpAPI Error: {data['error']}",
                        "details": data
                    }
                else:
                    return {
                        "success": True,
                        "message": "✅ SerpAPI funcionando correctamente",
                        "search_id": data.get("search_metadata", {}).get("id", "unknown")
                    }
            
            elif response.status_code == 401:
                return {
                    "success": False,
                    "error": "❌ API key inválida (401)",
                    "suggestion": "Verifica tu clave en https://serpapi.com/manage-api-key"
                }
            
            else:
                return {
                    "success": False,
                    "error": f"❌ HTTP {response.status_code}",
                    "response_preview": response.text[:300]
                }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"❌ Error: {str(e)}",
            "exception_type": type(e).__name__
        }

async def search_auto_parts_debug(query: str) -> Dict[str, Any]:
    """Búsqueda de auto parts con debug completo"""
    import httpx
    
    logger.info(f"🔍 INICIANDO BÚSQUEDA DEBUG: '{query}'")
    
    # 1. Verificar configuración
    config_check = check_serpapi_config()
    if not config_check["is_valid"]:
        return {
            "step": "config_check",
            "success": False,
            "error": "SerpAPI no configurado",
            "details": config_check
        }
    
    logger.info("✅ Configuración SerpAPI OK")
    
    # 2. Preparar parámetros
    url = "https://serpapi.com/search"
    params = {
        "api_key": SERPAPI_KEY,
        "engine": "google_shopping",
        "q": f"{query} auto parts",
        "location": "United States",
        "gl": "us",
        "hl": "en",
        "num": 10
    }
    
    logger.info(f"📋 Parámetros: engine=google_shopping, q='{params['q']}'")
    
    try:
        # 3. Hacer request
        async with httpx.AsyncClient(timeout=45.0) as client:
            logger.info("🌐 Enviando request a SerpAPI...")
            response = await client.get(url, params=params)
            logger.info(f"📊 Response Status: {response.status_code}")
            
            if response.status_code != 200:
                return {
                    "step": "serpapi_request",
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "response_preview": response.text[:500]
                }
            
            # 4. Parsear respuesta
            data = response.json()
            logger.info("📦 Response parseada correctamente")
            
            if "error" in data:
                return {
                    "step": "serpapi_response",
                    "success": False,
                    "error": f"SerpAPI Error: {data['error']}",
                    "full_response": data
                }
            
            # 5. Analizar resultados
            shopping_results = data.get("shopping_results", [])
            search_metadata = data.get("search_metadata", {})
            
            logger.info(f"🛍️ Shopping results encontrados: {len(shopping_results)}")
            
            # 6. Procesar resultados básicamente
            processed_results = []
            for i, result in enumerate(shopping_results[:5]):  # Solo primeros 5
                processed = {
                    "index": i,
                    "title": result.get("title", "Sin título"),
                    "price": result.get("price", "Sin precio"),
                    "source": result.get("source", "Sin fuente"),
                    "link": result.get("link", "Sin link"),
                    "thumbnail": result.get("thumbnail", None)
                }
                processed_results.append(processed)
                logger.info(f"  📦 Resultado {i}: {processed['title'][:50]}... - {processed['price']}")
            
            return {
                "step": "complete",
                "success": True,
                "query": query,
                "enhanced_query": params["q"],
                "total_results": len(shopping_results),
                "processed_results": processed_results,
                "search_metadata": search_metadata,
                "full_serpapi_response_keys": list(data.keys()),
                "sample_raw_result": shopping_results[0] if shopping_results else None
            }
    
    except Exception as e:
        logger.error(f"💥 Error en búsqueda: {str(e)}")
        return {
            "step": "exception",
            "success": False,
            "error": str(e),
            "exception_type": type(e).__name__
        }

# ================================================================
# APLICACIÓN FASTAPI
# ================================================================

app = FastAPI(
    title="Auto Parts Finder - Debug Final",
    version="1.0.0-debug-final",
    description="Diagnóstico completo de SerpAPI"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================================================
# ENDPOINTS
# ================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Página principal de debug"""
    
    config = check_serpapi_config()
    status_class = "success" if config["is_valid"] else "error"
    status_text = "✅ CONFIGURADO" if config["is_valid"] else "❌ NO CONFIGURADO"
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Auto Parts Debug Final</title>
    <meta charset="utf-8">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Monaco', 'Consolas', monospace; 
            background: #1a1a1a; 
            color: #00ff00; 
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        .header {{ 
            background: #2a2a2a; 
            padding: 20px; 
            border-radius: 8px; 
            margin-bottom: 20px;
            border: 2px solid #00ff00;
        }}
        .status {{ 
            padding: 15px; 
            margin: 15px 0; 
            border-radius: 5px; 
            border: 1px solid;
        }}
        .success {{ 
            background: #003300; 
            border-color: #00ff00; 
            color: #00ff00;
        }}
        .error {{ 
            background: #330000; 
            border-color: #ff0000; 
            color: #ff3333;
        }}
        .info {{ 
            background: #003333; 
            border-color: #00ffff; 
            color: #00ffff;
        }}
        button {{ 
            background: #333; 
            color: #00ff00; 
            border: 2px solid #00ff00; 
            padding: 12px 24px; 
            margin: 10px 5px; 
            border-radius: 5px; 
            cursor: pointer; 
            font-family: inherit;
            font-size: 14px;
        }}
        button:hover {{ 
            background: #00ff00; 
            color: #000; 
        }}
        .search-input {{ 
            background: #333; 
            color: #00ff00; 
            border: 2px solid #555; 
            padding: 12px; 
            font-family: inherit; 
            font-size: 16px; 
            width: 300px; 
            margin: 10px 5px;
        }}
        pre {{ 
            background: #000; 
            color: #00ff00; 
            padding: 15px; 
            border-radius: 5px; 
            overflow-x: auto; 
            border: 1px solid #333;
            max-height: 500px;
            overflow-y: auto;
        }}
        h1, h2 {{ color: #00ffff; }}
        .test-section {{ 
            background: #2a2a2a; 
            padding: 20px; 
            margin: 20px 0; 
            border-radius: 8px; 
            border: 1px solid #555;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔧 AUTO PARTS FINDER - DEBUG FINAL</h1>
            <p>Diagnóstico completo para identificar el problema exacto</p>
        </div>
        
        <div class="status {status_class}">
            <strong>🔑 Estado SerpAPI:</strong> {status_text}<br>
            <strong>📊 Clave existe:</strong> {"SÍ" if SERPAPI_KEY else "NO"}<br>
            <strong>📏 Longitud:</strong> {len(SERPAPI_KEY)} caracteres<br>
            {"<strong>👁️ Preview:</strong> " + config.get('serpapi_key_preview', '') if config.get('serpapi_key_preview') else ''}
        </div>
        
        <div class="test-section">
            <h2>🧪 PRUEBAS PASO A PASO</h2>
            <p>Ejecuta estas pruebas en orden para diagnosticar el problema:</p>
            
            <button onclick="step1()">1️⃣ Verificar Variables</button>
            <button onclick="step2()">2️⃣ Probar SerpAPI Simple</button>
            <button onclick="step3()">3️⃣ Búsqueda de Prueba</button>
            <button onclick="step4()">4️⃣ Búsqueda Completa</button>
        </div>
        
        <div class="test-section">
            <h2>🔍 BÚSQUEDA PERSONALIZADA</h2>
            <input type="text" class="search-input" id="customQuery" placeholder="brake pads" value="brake pads">
            <button onclick="searchCustom()">🚀 Buscar Auto Parts</button>
        </div>
        
        <div class="test-section">
            <h2>📊 RESULTADOS:</h2>
            <pre id="results">👆 Haz clic en los botones de arriba para comenzar el diagnóstico...</pre>
        </div>
        
        <div class="info status">
            <strong>ℹ️ Información del Sistema:</strong><br>
            <strong>🌍 Entorno:</strong> {ENVIRONMENT}<br>
            <strong>🖥️ Host:</strong> {HOST}<br>
            <strong>🔌 Puerto:</strong> {PORT}<br>
            <strong>⏰ Timestamp:</strong> {time.strftime('%Y-%m-%d %H:%M:%S UTC')}
        </div>
    </div>
    
    <script>
        function showLoading(message) {{
            document.getElementById('results').textContent = message + ' ⏳\\n\\nEspera un momento...';
        }}
        
        function showResult(data) {{
            document.getElementById('results').textContent = JSON.stringify(data, null, 2);
        }}
        
        async function step1() {{
            showLoading('🔍 PASO 1: Verificando variables de entorno');
            try {{
                const response = await fetch('/debug/environment');
                const data = await response.json();
                showResult(data);
            }} catch (error) {{
                document.getElementById('results').textContent = '❌ Error: ' + error.message;
            }}
        }}
        
        async function step2() {{
            showLoading('🔍 PASO 2: Probando conexión SerpAPI');
            try {{
                const response = await fetch('/debug/test-serpapi');
                const data = await response.json();
                showResult(data);
            }} catch (error) {{
                document.getElementById('results').textContent = '❌ Error: ' + error.message;
            }}
        }}
        
        async function step3() {{
            showLoading('🔍 PASO 3: Búsqueda simple de prueba');
            try {{
                const response = await fetch('/debug/search-simple?q=test');
                const data = await response.json();
                showResult(data);
            }} catch (error) {{
                document.getElementById('results').textContent = '❌ Error: ' + error.message;
            }}
        }}
        
        async function step4() {{
            showLoading('🔍 PASO 4: Búsqueda completa de auto parts');
            try {{
                const response = await fetch('/debug/search-full?q=brake+pads');
                const data = await response.json();
                showResult(data);
            }} catch (error) {{
                document.getElementById('results').textContent = '❌ Error: ' + error.message;
            }}
        }}
        
        async function searchCustom() {{
            const query = document.getElementById('customQuery').value.trim();
            if (!query) {{
                alert('❌ Por favor escribe algo para buscar');
                return;
            }}
            
            showLoading(`🔍 BÚSQUEDA PERSONALIZADA: "${{query}}"`);
            try {{
                const response = await fetch(`/debug/search-full?q=${{encodeURIComponent(query)}}`);
                const data = await response.json();
                showResult(data);
            }} catch (error) {{
                document.getElementById('results').textContent = '❌ Error: ' + error.message;
            }}
        }}
    </script>
</body>
</html>
    """
    
    return HTMLResponse(content=html)

@app.get("/debug/environment")
async def debug_environment():
    """Variables de entorno detalladas"""
    env_vars = dict(os.environ)
    
    important = {}
    for key in ["SERPAPI_KEY", "PORT", "HOST", "ENVIRONMENT", "PYTHON_VERSION"]:
        value = env_vars.get(key, "❌ NOT_SET")
        if key == "SERPAPI_KEY" and value != "❌ NOT_SET":
            important[key] = {
                "exists": True,
                "length": len(value),
                "preview": f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "⚠️ TOO_SHORT",
                "is_valid": len(value) >= 20
            }
        else:
            important[key] = value
    
    return {
        "step": "environment_check",
        "total_variables": len(env_vars),
        "important_variables": important,
        "serpapi_config": check_serpapi_config(),
        "environment_summary": {
            "serpapi_key_exists": bool(SERPAPI_KEY),
            "serpapi_key_length": len(SERPAPI_KEY),
            "render_environment": ENVIRONMENT,
            "port": PORT
        }
    }

@app.get("/debug/test-serpapi")
async def debug_test_serpapi():
    """Prueba básica de SerpAPI"""
    result = await test_serpapi_simple()
    logger.info(f"🧪 Test SerpAPI result: {result.get('success', False)}")
    return result

@app.get("/debug/search-simple")
async def debug_search_simple(q: str = Query(...)):
    """Búsqueda simple sin auto parts"""
    import httpx
    
    logger.info(f"🔍 Búsqueda simple: '{q}'")
    
    url = "https://serpapi.com/search"
    params = {
        "api_key": SERPAPI_KEY,
        "engine": "google",
        "q": q,
        "num": 3
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                organic_results = data.get("organic_results", [])
                
                return {
                    "step": "simple_search",
                    "success": True,
                    "query": q,
                    "total_results": len(organic_results),
                    "sample_results": organic_results[:2],
                    "search_metadata": data.get("search_metadata", {})
                }
            else:
                return {
                    "step": "simple_search",
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "response": response.text[:300]
                }
    
    except Exception as e:
        return {
            "step": "simple_search",
            "success": False,
            "error": str(e)
        }

@app.get("/debug/search-full")
async def debug_search_full(q: str = Query(...)):
    """Búsqueda completa de auto parts con debug"""
    result = await search_auto_parts_debug(q)
    return result

@app.get("/health")
async def health():
    """Health check"""
    config = check_serpapi_config()
    return {
        "status": "healthy" if config["is_valid"] else "degraded",
        "timestamp": time.time(),
        "serpapi_configured": config["is_valid"]
    }

# ================================================================
# STARTUP EVENT
# ================================================================

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 AUTO PARTS FINDER DEBUG FINAL - INICIADO")
    logger.info(f"🌍 Entorno: {ENVIRONMENT}")
    logger.info(f"🔌 Puerto: {PORT}")
    
    config = check_serpapi_config()
    logger.info(f"🔑 SerpAPI Status: {config['is_valid']}")
    
    if config["is_valid"]:
        logger.info("✅ Sistema listo para debugging")
    else:
        logger.warning("⚠️ SerpAPI no configurado - debugging limitado")

# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Iniciando Auto Parts Debug Final...")
    print(f"🌐 URL: http://{HOST}:{PORT}")
    print(f"🔑 SerpAPI: {'✅ OK' if SERPAPI_KEY else '❌ NO'}")
    
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info"
    )
