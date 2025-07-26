"""
Auto Parts Finder USA - Sistema Completo Unificado (FIXED for Render.com)
Sistema profesional de búsqueda de repuestos automotrices en EE.UU.
"""
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
import os

import httpx
import numpy as np
import psutil
try:
    import pytesseract
    import cv2
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️ OCR libraries not available - image search disabled")

from bs4 import BeautifulSoup
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field, HttpUrl, field_validator

# ================================================================
# CONFIGURACIÓN Y SETTINGS (FIXED)
# ================================================================

class Settings(BaseSettings):
    """Configuración principal del sistema"""
    
    # Configuración básica
    PROJECT_NAME: str = "Auto Parts Finder USA"
    VERSION: str = "1.0.1"
    DESCRIPTION: str = "Sistema profesional de búsqueda de repuestos automotrices en EE.UU."
    
    # Servidor
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=int(os.environ.get("PORT", 8000)), env="PORT")
    DEBUG: bool = Field(default=False, env="DEBUG")
    ENVIRONMENT: str = Field(default="production", env="ENVIRONMENT")
    
    # Seguridad
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32), env="SECRET_KEY")
    
    # SerpAPI (OBLIGATORIO) - FIXED
    SERPAPI_KEY: str = Field(default="", env="SERPAPI_KEY")
    SERPAPI_TIMEOUT: int = Field(default=45, env="SERPAPI_TIMEOUT")  # Increased timeout
    SERPAPI_MAX_RETRIES: int = Field(default=3, env="SERPAPI_MAX_RETRIES")
    
    # Redis (opcional)
    REDIS_URL: Optional[str] = Field(default=None, env="REDIS_URL")
    
    # Búsqueda
    MAX_RESULTS_PER_SEARCH: int = Field(default=20, env="MAX_RESULTS_PER_SEARCH")
    DEFAULT_RESULTS_LIMIT: int = Field(default=5, env="DEFAULT_RESULTS_LIMIT")
    MIN_QUERY_LENGTH: int = Field(default=3, env="MIN_QUERY_LENGTH")
    
    # Precios
    MIN_VALID_PRICE: float = Field(default=0.50, env="MIN_VALID_PRICE")
    MAX_VALID_PRICE: float = Field(default=50000.0, env="MAX_VALID_PRICE")
    PRICE_VERIFICATION_ENABLED: bool = Field(default=True, env="PRICE_VERIFICATION_ENABLED")
    
    # Imágenes
    MAX_IMAGE_SIZE_MB: int = Field(default=10, env="MAX_IMAGE_SIZE_MB")
    OCR_ENABLED: bool = Field(default=OCR_AVAILABLE, env="OCR_ENABLED")
    TESSERACT_CMD: Optional[str] = Field(default=None, env="TESSERACT_CMD")
    
    # Caché
    CACHE_ENABLED: bool = Field(default=True, env="CACHE_ENABLED")
    CACHE_TTL_SECONDS: int = Field(default=3600, env="CACHE_TTL_SECONDS")
    CACHE_MAX_SIZE: int = Field(default=1000, env="CACHE_MAX_SIZE")
    
    # Rate limiting
    RATE_LIMIT_ENABLED: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    RATE_LIMIT_REQUESTS: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    RATE_LIMIT_WINDOW: int = Field(default=3600, env="RATE_LIMIT_WINDOW")
    
    # Logs
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Dominios confiables de EE.UU.
    TRUSTED_US_DOMAINS: Dict[str, int] = {
        "autozone.com": 5,
        "oreillyauto.com": 5,
        "advanceautoparts.com": 5,
        "napaonline.com": 5,
        "pepboys.com": 5,
        "rockauto.com": 4,
        "partsgeek.com": 4,
        "carparts.com": 4,
        "1aauto.com": 4,
        "amazon.com": 3,
        "ebay.com": 3,
        "walmart.com": 3,
    }
    
    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == "development"
    
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"
    
    @property
    def serpapi_configured(self) -> bool:
        return bool(self.SERPAPI_KEY and self.SERPAPI_KEY != "TU_SERPAPI_KEY_AQUI" and len(self.SERPAPI_KEY) > 10)
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()

# ================================================================
# EXCEPCIONES PERSONALIZADAS (ENHANCED)
# ================================================================

class AutoPartsBaseException(Exception):
    """Excepción base para todas las excepciones de la aplicación"""
    
    def __init__(self, message: str, error_code: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": True,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details
        }

class ConfigurationError(AutoPartsBaseException):
    def __init__(self, missing_config: str):
        super().__init__(
            message=f"Configuración faltante: {missing_config}",
            error_code="CONFIG_ERROR"
        )

class InvalidSearchQueryError(AutoPartsBaseException):
    def __init__(self, query: str, reason: str):
        super().__init__(
            message=f"Consulta inválida: {reason}",
            error_code="INVALID_SEARCH_QUERY",
            details={"query": query, "reason": reason}
        )

class NoResultsFoundError(AutoPartsBaseException):
    def __init__(self, query: str, reason: str = "No se encontraron resultados"):
        super().__init__(
            message=reason,
            error_code="NO_RESULTS_FOUND",
            details={"query": query}
        )

class SerpAPIError(AutoPartsBaseException):
    def __init__(self, status_code: int, response_data: Dict[str, Any], original_error: str = ""):
        error_msg = f"Error en SerpAPI (código {status_code})"
        if original_error:
            error_msg += f": {original_error}"
        
        super().__init__(
            message=error_msg,
            error_code="SERPAPI_ERROR",
            details={"status_code": status_code, "response_data": response_data, "original_error": original_error}
        )

class InvalidImageError(AutoPartsBaseException):
    def __init__(self, reason: str):
        super().__init__(
            message=f"Imagen inválida: {reason}",
            error_code="INVALID_IMAGE"
        )

class OCRProcessingError(AutoPartsBaseException):
    def __init__(self, reason: str):
        super().__init__(
            message=f"Error en OCR: {reason}",
            error_code="OCR_ERROR"
        )

def to_http_exception(exc: AutoPartsBaseException) -> HTTPException:
    """Convierte excepción personalizada a HTTPException"""
    status_code_mapping = {
        "CONFIG_ERROR": 503,
        "INVALID_SEARCH_QUERY": 400,
        "NO_RESULTS_FOUND": 404,
        "SERPAPI_ERROR": 502,
        "INVALID_IMAGE": 400,
        "OCR_ERROR": 422,
    }
    
    status_code = status_code_mapping.get(exc.error_code, 500)
    return HTTPException(status_code=status_code, detail=exc.to_dict())

# ================================================================
# MODELOS DE DATOS (UNCHANGED)
# ================================================================

class SortOption(str, Enum):
    PRICE_LOW_TO_HIGH = "price_asc"
    PRICE_HIGH_TO_LOW = "price_desc"
    RELEVANCE = "relevance"
    RELIABILITY = "reliability"

class AvailabilityStatus(str, Enum):
    IN_STOCK = "in_stock"
    OUT_OF_STOCK = "out_of_stock"
    LIMITED_STOCK = "limited_stock"

class ShippingSpeed(str, Enum):
    SAME_DAY = "same_day"
    NEXT_DAY = "next_day"
    TWO_DAY = "two_day"
    STANDARD = "standard"

class PriceRange(BaseModel):
    min_price: Optional[Decimal] = Field(None, ge=0)
    max_price: Optional[Decimal] = Field(None, ge=0)

class VendorFilter(BaseModel):
    excluded_vendors: List[str] = Field(default_factory=list)
    min_reliability_score: int = Field(default=1, ge=1, le=5)

class SearchFilters(BaseModel):
    price_range: Optional[PriceRange] = None
    vendor: VendorFilter = Field(default_factory=VendorFilter)

class TextSearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=200)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    sort_by: SortOption = Field(default=SortOption.PRICE_LOW_TO_HIGH)
    max_results: int = Field(default=5, ge=1, le=50)
    
    @field_validator('query')
    def validate_query(cls, v):
        cleaned = v.strip()
        if len(cleaned) < 3:
            raise ValueError('La consulta debe tener al menos 3 caracteres')
        return cleaned
    
    def get_enhanced_query(self) -> str:
        query = self.query
        if not any(term in query.lower() for term in ['auto', 'car', 'part']):
            query += " auto part"
        return query

class SearchMetrics(BaseModel):
    search_time_ms: int = Field(..., ge=0)
    total_found: int = Field(..., ge=0)
    total_filtered: int = Field(..., ge=0)
    sources_queried: int = Field(..., ge=0)
    cache_hit: bool = Field(default=False)

class SearchResponse(BaseModel):
    search_type: str
    original_query: str
    processed_query: str
    search_id: str
    products: List[Dict[str, Any]]
    total_count: int
    applied_filters: SearchFilters
    sort_applied: SortOption
    metrics: SearchMetrics
    suggestions: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class OCRResult(BaseModel):
    extracted_text: str
    confidence: float = Field(..., ge=0, le=1)
    part_numbers: List[str] = Field(default_factory=list)
    brands: List[str] = Field(default_factory=list)
    processing_time_ms: int = Field(..., ge=0)
    
    def is_reliable(self) -> bool:
        return self.confidence >= 0.7 and len(self.extracted_text.strip()) >= 3

class ProductSummary(BaseModel):
    title: str = Field(..., min_length=5, max_length=200)
    price: Decimal = Field(..., ge=0)
    vendor_name: str = Field(..., max_length=100)
    vendor_domain: str = Field(..., max_length=100)
    reliability_score: int = Field(..., ge=1, le=5)
    availability: AvailabilityStatus = AvailabilityStatus.IN_STOCK
    product_url: HttpUrl
    is_on_sale: bool = False
    shipping_speed: ShippingSpeed = ShippingSpeed.STANDARD
    primary_image_url: Optional[HttpUrl] = None
    
    class Config:
        json_encoders = {
            Decimal: lambda v: float(v)
        }

# ================================================================
# CLIENTE SERPAPI (ENHANCED WITH BETTER ERROR HANDLING)
# ================================================================

class SerpAPIClient:
    """Cliente optimizado para SerpAPI con mejor manejo de errores"""
    
    def __init__(self):
        if not settings.serpapi_configured:
            raise ConfigurationError("SERPAPI_KEY no configurado correctamente")
        
        self.api_key = settings.SERPAPI_KEY
        self.base_url = "https://serpapi.com/search"
        
        # Enhanced timeout and connection settings for Render.com
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,  # Connection timeout
                read=settings.SERPAPI_TIMEOUT,  # Read timeout
                write=10.0,  # Write timeout
                pool=5.0   # Pool timeout
            ),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            follow_redirects=True
        )
        
        self._cache: Dict[str, Tuple[Dict, datetime]] = {}
        self._cache_ttl = timedelta(seconds=settings.CACHE_TTL_SECONDS)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def close(self):
        if self.client:
            await self.client.aclose()
    
    def _generate_cache_key(self, params: Dict) -> str:
        sorted_params = dict(sorted(params.items()))
        sorted_params.pop('api_key', None)
        return hashlib.md5(json.dumps(sorted_params, sort_keys=True).encode()).hexdigest()
    
    def _enhance_query(self, query: str) -> str:
        query = query.strip()
        query_lower = query.lower()
        
        if not any(term in query_lower for term in ['auto', 'car', 'vehicle', 'part']):
            query += " auto parts"
        
        if not any(term in query_lower for term in ['genuine', 'oem', 'new']):
            query += " OEM genuine"
        
        return query
    
    def _validate_us_result(self, result: Dict) -> bool:
        """Enhanced US validation with more permissive rules"""
        try:
            link = result.get("link", "")
            if not link:
                return False
            
            parsed_url = urlparse(link)
            domain = parsed_url.netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
            
            # Check trusted domains first
            if domain in settings.TRUSTED_US_DOMAINS:
                return True
            
            # More permissive US validation
            us_extensions = [".com", ".us", ".org", ".net"]
            if any(domain.endswith(ext) for ext in us_extensions):
                # Only exclude obvious international domains
                international_indicators = [
                    ".ca", ".uk", ".de", ".fr", ".au", ".mx", ".cn", ".jp",
                    "canada", "europe", "china", "japan"
                ]
                if not any(indicator in domain for indicator in international_indicators):
                    return True
            
            return False
        except Exception as e:
            logger.warning(f"Error validando resultado US: {e}")
            return False
    
    async def _make_request_with_retries(self, params: Dict) -> Dict:
        """Make request with retries and better error handling"""
        last_error = None
        
        for attempt in range(settings.SERPAPI_MAX_RETRIES):
            try:
                logger.info(f"SerpAPI request attempt {attempt + 1}/{settings.SERPAPI_MAX_RETRIES}")
                
                response = await self.client.get(self.base_url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check for API errors
                    if "error" in data:
                        error_msg = data.get("error", "Unknown error")
                        raise SerpAPIError(200, data, error_msg)
                    
                    return data
                
                elif response.status_code == 401:
                    raise SerpAPIError(401, {"error": "Invalid API key"}, "API key inválida")
                
                elif response.status_code == 429:
                    # Rate limit - wait longer on retries
                    wait_time = (attempt + 1) * 2
                    logger.warning(f"Rate limit hit, waiting {wait_time}s before retry")
                    await asyncio.sleep(wait_time)
                    continue
                
                elif response.status_code >= 500:
                    # Server error - retry
                    wait_time = (attempt + 1) * 1
                    logger.warning(f"Server error {response.status_code}, retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue
                
                else:
                    raise SerpAPIError(response.status_code, {"error": f"HTTP {response.status_code}"})
            
            except httpx.TimeoutException as e:
                last_error = SerpAPIError(408, {"error": "Request timeout"}, str(e))
                if attempt < settings.SERPAPI_MAX_RETRIES - 1:
                    wait_time = (attempt + 1) * 2
                    logger.warning(f"Timeout on attempt {attempt + 1}, retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue
            
            except httpx.ConnectError as e:
                last_error = SerpAPIError(503, {"error": "Connection error"}, str(e))
                if attempt < settings.SERPAPI_MAX_RETRIES - 1:
                    wait_time = (attempt + 1) * 3
                    logger.warning(f"Connection error on attempt {attempt + 1}, retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue
            
            except Exception as e:
                last_error = SerpAPIError(500, {"error": "Unexpected error"}, str(e))
                if attempt < settings.SERPAPI_MAX_RETRIES - 1:
                    wait_time = (attempt + 1) * 2
                    logger.warning(f"Unexpected error on attempt {attempt + 1}, retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue
        
        # All retries failed
        if last_error:
            raise last_error
        else:
            raise SerpAPIError(500, {"error": "Max retries exceeded"})
    
    async def search_auto_parts(
        self,
        query: str,
        max_results: int = 20,
        use_cache: bool = True
    ) -> Tuple[List[Dict], SearchMetrics]:
        start_time = time.time()
        
        enhanced_query = self._enhance_query(query)
        
        params = {
            "api_key": self.api_key,
            "engine": "google_shopping",
            "q": enhanced_query,
            "location": "United States",
            "gl": "us",
            "hl": "en",
            "num": min(max_results * 2, 100),
            "tbm": "shop"
        }
        
        cache_key = self._generate_cache_key(params)
        if use_cache and cache_key in self._cache:
            cached_data, cached_time = self._cache[cache_key]
            if datetime.utcnow() - cached_time < self._cache_ttl:
                logger.debug(f"Cache hit para: {query}")
                shopping_results = cached_data.get("shopping_results", [])
                validated_results = [r for r in shopping_results if self._validate_us_result(r)]
                
                metrics = SearchMetrics(
                    search_time_ms=int((time.time() - start_time) * 1000),
                    total_found=len(shopping_results),
                    total_filtered=len(validated_results),
                    sources_queried=1,
                    cache_hit=True
                )
                return validated_results[:max_results], metrics
        
        try:
            data = await self._make_request_with_retries(params)
            
            if use_cache:
                self._cache[cache_key] = (data, datetime.utcnow())
                # Clean old cache entries
                if len(self._cache) > 100:
                    oldest_key = min(self._cache.keys(), 
                                   key=lambda k: self._cache[k][1])
                    del self._cache[oldest_key]
            
            shopping_results = data.get("shopping_results", [])
            
            # More permissive validation
            validated_results = []
            for result in shopping_results:
                if self._validate_us_result(result):
                    validated_results.append(result)
            
            # If no results with strict validation, try with basic validation
            if not validated_results and shopping_results:
                logger.warning("No strict US results found, using basic validation")
                for result in shopping_results[:max_results]:
                    link = result.get("link", "")
                    if link and not any(bad in link.lower() for bad in [".ca", ".uk", ".de", ".fr", ".mx"]):
                        validated_results.append(result)
            
            search_time_ms = int((time.time() - start_time) * 1000)
            
            metrics = SearchMetrics(
                search_time_ms=search_time_ms,
                total_found=len(shopping_results),
                total_filtered=len(validated_results),
                sources_queried=1,
                cache_hit=False
            )
            
            logger.info(f"Búsqueda completada: '{query}' -> {len(validated_results)} resultados en {search_time_ms}ms")
            return validated_results[:max_results], metrics
        
        except Exception as e:
            logger.error(f"Error en búsqueda SerpAPI: {e}")
            if isinstance(e, SerpAPIError):
                raise
            raise SerpAPIError(500, {"error": str(e)})

# ================================================================
# RESTO DEL CÓDIGO (PROCESADOR DE IMÁGENES, SERVICIOS, ETC.)
# ================================================================

class ImageProcessor:
    """Procesador de imágenes con OCR optimizado"""
    
    def __init__(self):
        if OCR_AVAILABLE:
            self._verify_tesseract()
        
        self.part_number_patterns = [
            r'\b[A-Z]{2,4}[-\s]?\d{3,8}[-\s]?[A-Z]?\b',
            r'\b\d{5,10}\b',
            r'\b[A-Z]\d{6,8}\b',
            r'\b\d{2,4}[-\s][A-Z]{2,3}[-\s]\d{3,6}\b',
        ]
        
        self.brand_patterns = [
            r'\b(?:AC\s+DELCO|ACDELCO)\b',
            r'\b(?:BOSCH)\b',
            r'\b(?:DENSO)\b',
            r'\b(?:NGK)\b',
            r'\b(?:CHAMPION)\b',
            r'\b(?:GATES)\b',
            r'\b(?:MONROE)\b',
            r'\b(?:MOOG)\b',
        ]
    
    def _verify_tesseract(self):
        if not OCR_AVAILABLE:
            raise OCRProcessingError("OCR libraries not available")
        
        try:
            if settings.TESSERACT_CMD:
                pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
            
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")
        except Exception as e:
            logger.error(f"Error verificando Tesseract: {e}")
            raise OCRProcessingError(f"Tesseract no disponible: {e}")
    
    async def process_image_upload(self, image_data: bytes, filename: str) -> dict:
        file_size_mb = len(image_data) / (1024 * 1024)
        if file_size_mb > settings.MAX_IMAGE_SIZE_MB:
            raise InvalidImageError(f"Imagen muy grande: {file_size_mb:.1f}MB > {settings.MAX_IMAGE_SIZE_MB}MB")
        
        try:
            image = Image.open(io.BytesIO(image_data))
            
            if image.format and image.format.lower() not in ['jpeg', 'jpg', 'png', 'webp', 'bmp']:
                raise InvalidImageError(f"Formato no soportado: {image.format}")
            
            if image.width < 50 or image.height < 50:
                raise InvalidImageError("Imagen muy pequeña (mínimo 50x50)")
            
            return {
                "filename": filename,
                "format": image.format,
                "size_bytes": len(image_data),
                "width": image.width,
                "height": image.height,
                "valid": True
            }
        
        except Exception as e:
            if isinstance(e, InvalidImageError):
                raise
            raise InvalidImageError(f"Imagen corrupta: {e}")
    
    def _preprocess_image(self, image_array: np.ndarray) -> np.ndarray:
        if not OCR_AVAILABLE:
            return image_array
        
        try:
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array.copy()
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            denoised = cv2.fastNlMeansDenoising(enhanced)
            
            binary = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            return binary
        
        except Exception as e:
            logger.warning(f"Error en preprocesamiento: {e}")
            return image_array
    
    def _extract_text_with_ocr(self, image_array: np.ndarray) -> Tuple[str, float]:
        if not OCR_AVAILABLE:
            return "", 0.0
        
        try:
            config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-./() '
            
            data = pytesseract.image_to_data(
                image_array, 
                config=config, 
                output_type=pytesseract.Output.DICT
            )
            
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            words = [
                data['text'][i] for i in range(len(data['text']))
                if int(data['conf'][i]) > 30
            ]
            
            text = ' '.join(words).strip()
            avg_confidence = np.mean(confidences) / 100.0 if confidences else 0.0
            
            return text, avg_confidence
        
        except Exception as e:
            logger.error(f"Error en OCR: {e}")
            return "", 0.0
    
    def _extract_structured_info(self, text: str) -> dict:
        result = {
            'part_numbers': [],
            'brands': []
        }
        
        if not text:
            return result
        
        text_upper = text.upper()
        
        for pattern in self.part_number_patterns:
            matches = re.findall(pattern, text_upper)
            result['part_numbers'].extend(matches)
        
        for pattern in self.brand_patterns:
            matches = re.findall(pattern, text_upper)
            result['brands'].extend(matches)
        
        result['part_numbers'] = list(set(result['part_numbers']))
        result['brands'] = list(set(result['brands']))
        
        return result
    
    async def extract_text_from_image(
        self,
        image_data: bytes,
        enhance_image: bool = True,
        confidence_threshold: float = 0.5
    ) -> OCRResult:
        start_time = time.time()
        
        if not OCR_AVAILABLE:
            raise OCRProcessingError("OCR not available - missing libraries")
        
        try:
            image_pil = Image.open(io.BytesIO(image_data))
            image_array = np.array(image_pil)
            
            if enhance_image:
                image_array = self._preprocess_image(image_array)
            
            extracted_text, confidence = self._extract_text_with_ocr(image_array)
            structured_info = self._extract_structured_info(extracted_text)
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            if confidence < confidence_threshold and not extracted_text.strip():
                logger.warning(f"OCR confidence {confidence:.2f} below threshold {confidence_threshold}")
            
            result = OCRResult(
                extracted_text=extracted_text,
                confidence=confidence,
                part_numbers=structured_info['part_numbers'],
                brands=structured_info['brands'],
                processing_time_ms=processing_time_ms
            )
            
            logger.info(f"OCR completado: '{extracted_text[:50]}...' (confianza: {confidence:.2f}) en {processing_time_ms}ms")
            return result
        
        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Error en OCR: {e}")
            raise OCRProcessingError(f"Error procesando imagen: {e}")

# ================================================================
# SERVICIOS AUXILIARES (UNCHANGED)
# ================================================================

class SimpleGeoFilter:
    def __init__(self):
        self.trusted_domains = settings.TRUSTED_US_DOMAINS
        self.international_indicators = [
            ".ca", ".uk", ".de", ".fr", ".au", ".mx", ".cn",
            "canada", "europe", "international", "global", "china"
        ]
    
    def validate_us_vendor(self, result: Dict[str, Any]) -> bool:
        try:
            link = result.get("link", "")
            if not link:
                return False
            
            domain = urlparse(link).netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
            
            if domain in self.trusted_domains:
                return True
            
            for indicator in self.international_indicators:
                if indicator in domain or indicator in result.get("source", "").lower():
                    return False
            
            return domain.endswith((".com", ".org", ".net", ".us"))
        
        except Exception as e:
            logger.warning(f"Error validando geolocalización: {e}")
            return False
    
    async def filter_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered = [r for r in results if self.validate_us_vendor(r)]
        logger.info(f"Filtrado geográfico: {len(filtered)}/{len(results)} resultados válidos")
        return filtered

class SimplePriceValidator:
    def __init__(self):
        self.min_price = settings.MIN_VALID_PRICE
        self.max_price = settings.MAX_VALID_PRICE
    
    def validate_price(self, price_str: str) -> Optional[float]:
        try:
            clean_price = re.sub(r'[^\d.]', '', str(price_str))
            price = float(clean_price)
            
            if self.min_price <= price <= self.max_price:
                return price
            
            logger.warning(f"Precio fuera de rango: ${price}")
            return None
        
        except (ValueError, TypeError):
            logger.warning(f"Precio inválido: {price_str}")
            return None
    
    def is_suspicious_price(self, price: float, title: str) -> bool:
        if price < 5.0 and any(term in title.lower() for term in ['oem', 'genuine', 'premium']):
            return True
        
        cents = int((price % 1) * 100)
        if cents in [37, 63, 73, 83, 87, 93, 97]:
            return True
        
        return False

class SimpleAntiFraud:
    def __init__(self):
        self.fraud_keywords = [
            "fake", "replica", "copy", "knockoff", "imitation", "counterfeit",
            "cheap", "clone", "unauthorized", "bootleg", "generic aftermarket"
        ]
        
        self.trust_keywords = [
            "oem", "genuine", "original", "authentic", "certified",
            "warranty", "guaranteed", "brand new", "factory", "manufacturer"
        ]
    
    def analyze_product(self, result: Dict[str, Any]) -> Dict[str, Any]:
        title = result.get("title", "").lower()
        snippet = result.get("snippet", "").lower()
        source = result.get("source", "").lower()
        
        risk_score = 0.0
        warnings = []
        
        for keyword in self.fraud_keywords:
            if keyword in title:
                risk_score += 0.3
                warnings.append(f"Palabra sospechosa en título: {keyword}")
        
        trust_score = 0
        for keyword in self.trust_keywords:
            if keyword in title or keyword in snippet:
                trust_score += 1
        
        if trust_score == 0:
            risk_score += 0.2
            warnings.append("Sin indicadores de autenticidad")
        
        generic_names = ["auto parts", "car parts", "generic parts"]
        if any(name in source for name in generic_names):
            risk_score += 0.2
            warnings.append("Vendedor con nombre genérico")
        
        price_str = result.get("price", "")
        if price_str:
            validator = SimplePriceValidator()
            price = validator.validate_price(price_str)
            if price and validator.is_suspicious_price(price, title):
                risk_score += 0.3
                warnings.append("Precio sospechoso")
        
        is_suspicious = risk_score >= 0.5
        should_block = risk_score >= 0.8
        
        return {
            "risk_score": min(1.0, risk_score),
            "is_suspicious": is_suspicious,
            "should_block": should_block,
            "warnings": warnings
        }

class SimpleCache:
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        return self.cache.get(key)
    
    def set(self, key: str, value: Any):
        if len(self.cache) >= self.max_size:
            keys_to_remove = list(self.cache.keys())[:self.max_size // 5]
            for k in keys_to_remove:
                del self.cache[k]
        
        self.cache[key] = value
    
    def clear(self):
        self.cache.clear()

# ================================================================
# SERVICIOS GLOBALES (ENHANCED)
# ================================================================

# Instancias globales
_serpapi_client_instance: Optional[SerpAPIClient] = None
_image_processor_instance: Optional[ImageProcessor] = None
_geo_filter = SimpleGeoFilter()
_price_validator = SimplePriceValidator()
_antifraud = SimpleAntiFraud()
_cache = SimpleCache()

async def get_serpapi_client() -> SerpAPIClient:
    global _serpapi_client_instance
    if _serpapi_client_instance is None:
        _serpapi_client_instance = SerpAPIClient()
    return _serpapi_client_instance

async def get_image_processor() -> ImageProcessor:
    global _image_processor_instance
    if _image_processor_instance is None:
        _image_processor_instance = ImageProcessor()
    return _image_processor_instance

async def get_geo_filter():
    return _geo_filter

async def get_price_validator():
    return _price_validator

async def get_antifraud_analyzer():
    return _antifraud

async def get_cache_manager():
    return _cache

async def cleanup_serpapi_client():
    global _serpapi_client_instance
    if _serpapi_client_instance is not None:
        await _serpapi_client_instance.close()
        _serpapi_client_instance = None

# ================================================================
# SERVICIO DE BÚSQUEDA (ENHANCED)
# ================================================================

class SearchService:
    def __init__(self):
        self.search_id_prefix = "search_"
    
    def generate_search_id(self) -> str:
        return f"{self.search_id_prefix}{uuid.uuid4().hex[:12]}"
    
    def _convert_raw_to_product(self, raw_result: dict) -> Optional[ProductSummary]:
        try:
            price_str = raw_result.get("price", "")
            if not price_str:
                return None
            
            price_validator = _price_validator
            price = price_validator.validate_price(price_str)
            if not price:
                return None
            
            link = raw_result.get("link", "")
            domain = ""
            if link:
                try:
                    parsed = urlparse(link)
                    domain = parsed.netloc.lower()
                    if domain.startswith("www."):
                        domain = domain[4:]
                except:
                    domain = ""
            
            reliability_score = settings.TRUSTED_US_DOMAINS.get(domain, 3)
            
            snippet = raw_result.get("snippet", "").lower()
            if any(term in snippet for term in ["out of stock", "not available"]):
                availability = AvailabilityStatus.OUT_OF_STOCK
            elif any(term in snippet for term in ["limited", "few left"]):
                availability = AvailabilityStatus.LIMITED_STOCK
            else:
                availability = AvailabilityStatus.IN_STOCK
            
            text = (raw_result.get("title", "") + " " + snippet).lower()
            is_on_sale = any(term in text for term in ["sale", "discount", "off", "deal"])
            
            if any(term in text for term in ["same day", "today"]):
                shipping_speed = ShippingSpeed.SAME_DAY
            elif any(term in text for term in ["next day", "overnight"]):
                shipping_speed = ShippingSpeed.NEXT_DAY
            elif any(term in text for term in ["2 day", "two day"]):
                shipping_speed = ShippingSpeed.TWO_DAY
            else:
                shipping_speed = ShippingSpeed.STANDARD
            
            return ProductSummary(
                title=raw_result.get("title", "")[:200],
                price=Decimal(str(price)),
                vendor_name=raw_result.get("source", "")[:100],
                vendor_domain=domain,
                reliability_score=reliability_score,
                availability=availability,
                product_url=link,
                is_on_sale=is_on_sale,
                shipping_speed=shipping_speed,
                primary_image_url=raw_result.get("thumbnail")
            )
        
        except Exception as e:
            logger.warning(f"Error convirtiendo producto: {e}")
            return None
    
    def _apply_filters(self, products: List[ProductSummary], filters) -> List[ProductSummary]:
        filtered = []
        
        for product in products:
            if filters.price_range:
                if filters.price_range.min_price and product.price < filters.price_range.min_price:
                    continue
                if filters.price_range.max_price and product.price > filters.price_range.max_price:
                    continue
            
            if filters.vendor.excluded_vendors:
                if any(excluded.lower() in product.vendor_domain.lower() or 
                      excluded.lower() in product.vendor_name.lower() 
                      for excluded in filters.vendor.excluded_vendors):
                    continue
            
            if product.reliability_score < filters.vendor.min_reliability_score:
                continue
            
            filtered.append(product)
        
        return filtered
    
    def _apply_sorting(self, products: List[ProductSummary], sort_by: SortOption) -> List[ProductSummary]:
        if sort_by == SortOption.PRICE_LOW_TO_HIGH:
            return sorted(products, key=lambda p: p.price)
        elif sort_by == SortOption.PRICE_HIGH_TO_LOW:
            return sorted(products, key=lambda p: p.price, reverse=True)
        elif sort_by == SortOption.RELIABILITY:
            return sorted(products, key=lambda p: p.reliability_score, reverse=True)
        else:
            return sorted(products, key=lambda p: (p.price, -p.reliability_score))
    
    async def process_text_search(self, request: TextSearchRequest, search_id: str) -> SearchResponse:
        start_time = time.time()
        
        try:
            # Check API configuration first
            if not settings.serpapi_configured:
                raise ConfigurationError("SerpAPI not configured")
            
            cache = _cache
            cache_key = f"search_{hash(request.query + str(request.filters.dict()))}"
            cached_result = cache.get(cache_key)
            
            if cached_result:
                logger.info(f"Cache hit para: {request.query}")
                return cached_result
            
            serpapi_client = await get_serpapi_client()
            
            try:
                raw_results, search_metrics = await serpapi_client.search_auto_parts(
                    query=request.get_enhanced_query(),
                    max_results=request.max_results * 2
                )
            except SerpAPIError as e:
                logger.error(f"SerpAPI error: {e}")
                # Provide helpful error message based on error type
                if "API key" in str(e).lower():
                    raise NoResultsFoundError(request.query, "API key de SerpAPI inválida o no configurada")
                elif "rate limit" in str(e).lower():
                    raise NoResultsFoundError(request.query, "Límite de requests excedido. Intente en unos minutos")
                elif "timeout" in str(e).lower():
                    raise NoResultsFoundError(request.query, "Tiempo de espera agotado. Intente nuevamente")
                else:
                    raise NoResultsFoundError(request.query, f"Error en búsqueda: {str(e)[:100]}")
            
            if not raw_results:
                raise NoResultsFoundError(request.query, "No se encontraron productos para esta búsqueda")
            
            # More permissive geo filtering
            geo_filter = _geo_filter
            us_results = await geo_filter.filter_results(raw_results)
            
            if not us_results:
                # If no strict US results, use all results
                logger.warning("No US results found, using all results")
                us_results = raw_results
            
            products = []
            antifraud = _antifraud
            
            for raw_result in us_results:
                product = self._convert_raw_to_product(raw_result)
                if not product:
                    continue
                
                fraud_analysis = antifraud.analyze_product(raw_result)
                if fraud_analysis["should_block"]:
                    logger.info(f"Producto bloqueado por fraude: {product.title[:50]}")
                    continue
                
                products.append(product)
            
            if not products:
                raise NoResultsFoundError(request.query, "No se encontraron productos válidos")
            
            filtered_products = self._apply_filters(products, request.filters)
            if not filtered_products:
                # If filters are too restrictive, use unfiltered products
                logger.warning("Filters too restrictive, using unfiltered products")
                filtered_products = products
            
            sorted_products = self._apply_sorting(filtered_products, request.sort_by)
            final_products = sorted_products[:request.max_results]
            
            response = SearchResponse(
                search_type="text",
                original_query=request.query,
                processed_query=request.get_enhanced_query(),
                search_id=search_id,
                products=[p.dict() for p in final_products],
                total_count=len(final_products),
                applied_filters=request.filters,
                sort_applied=request.sort_by,
                metrics=SearchMetrics(
                    search_time_ms=int((time.time() - start_time) * 1000),
                    total_found=len(raw_results),
                    total_filtered=len(final_products),
                    sources_queried=1,
                    cache_hit=False
                )
            )
            
            cache.set(cache_key, response)
            
            logger.info(f"Búsqueda completada: '{request.query}' -> {len(final_products)} productos")
            return response
        
        except (NoResultsFoundError, InvalidSearchQueryError, ConfigurationError):
            raise
        except Exception as e:
            logger.error(f"Error en búsqueda: {e}")
            raise NoResultsFoundError(request.query, f"Error interno: {str(e)[:100]}")

# ================================================================
# CONFIGURACIÓN DE LOGGING (ENHANCED)
# ================================================================

# Enhanced logging configuration for Render.com
log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # Console output for Render.com
    ]
)

logger = logging.getLogger(__name__)

# Reduce noise from httpx
logging.getLogger("httpx").setLevel(logging.WARNING)

# ================================================================
# APLICACIÓN FASTAPI (ENHANCED)
# ================================================================

# Instancia del servicio de búsqueda
search_service = SearchService()

# Contexto de ciclo de vida
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Iniciando Auto Parts Finder USA API...")
    
    # Check essential configuration
    if not settings.serpapi_configured:
        logger.error("❌ SERPAPI_KEY no configurado correctamente")
        logger.error("💡 Configura la variable de entorno SERPAPI_KEY con tu clave de SerpAPI")
        # Don't raise error immediately, let the app start and show helpful error messages
    else:
        logger.info("✅ SERPAPI_KEY configurado correctamente")
    
    if settings.OCR_ENABLED and OCR_AVAILABLE:
        try:
            pytesseract.get_tesseract_version()
            logger.info("✅ Tesseract OCR disponible")
        except Exception as e:
            logger.warning(f"⚠️ Tesseract no disponible: {e}")
    
    logger.info(f"🌍 Servidor configurado en: {settings.HOST}:{settings.PORT}")
    logger.info(f"📊 Entorno: {settings.ENVIRONMENT}")
    logger.info("🎯 API lista para recibir requests")
    
    yield
    
    logger.info("⏹️ Cerrando Auto Parts Finder USA API...")
    await cleanup_serpapi_client()
    logger.info("🔚 API cerrada correctamente")

# Crear aplicación FastAPI
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
    docs_url=None,
    redoc_url=None,
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.is_development else ["*"],  # Cambiar en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting simple (unchanged)
if settings.RATE_LIMIT_ENABLED:
    request_counts = defaultdict(list)
    
    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        request_counts[client_ip] = [
            req_time for req_time in request_counts[client_ip]
            if current_time - req_time < settings.RATE_LIMIT_WINDOW
        ]
        
        if len(request_counts[client_ip]) >= settings.RATE_LIMIT_REQUESTS:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Máximo {settings.RATE_LIMIT_REQUESTS} requests por hora"
                }
            )
        
        request_counts[client_ip].append(current_time)
        response = await call_next(request)
        return response

# ================================================================
# ENDPOINTS (ENHANCED WITH BETTER ERROR HANDLING)
# ================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Página principal del sitio web"""
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Auto Parts Finder USA</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🔧</text></svg>">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 20px 0;
            margin-bottom: 40px;
        }
        
        .header h1 {
            color: white;
            text-align: center;
            font-size: 2.5rem;
            font-weight: 700;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            color: rgba(255, 255, 255, 0.9);
            text-align: center;
            font-size: 1.2rem;
            margin-top: 10px;
        }
        
        .config-notice {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 15px 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: none;
        }
        
        .config-notice.show {
            display: block;
        }
        
        .search-section {
            background: white;
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            margin-bottom: 40px;
        }
        
        .search-tabs {
            display: flex;
            margin-bottom: 30px;
            gap: 10px;
        }
        
        .tab-button {
            flex: 1;
            padding: 15px 25px;
            border: none;
            background: #f8f9fa;
            color: #666;
            border-radius: 10px;
            cursor: pointer;
            font-size: 1rem;
            transition: all 0.3s ease;
        }
        
        .tab-button.active {
            background: #667eea;
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        }
        
        .search-form {
            display: none;
        }
        
        .search-form.active {
            display: block;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        
        .form-group input, 
        .form-group select {
            width: 100%;
            padding: 15px;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            font-size: 1rem;
            transition: all 0.3s ease;
        }
        
        .form-group input:focus,
        .form-group select:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        
        .search-button {
            width: 100%;
            padding: 15px 30px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .search-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        }
        
        .search-button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }
        
        .spinner {
            width: 50px;
            height: 50px;
            border: 5px solid #f3f3f3;
            border-top: 5px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .results {
            display: none;
        }
        
        .results.show {
            display: block;
        }
        
        .results-header {
            background: white;
            padding: 20px 40px;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        }
        
        .product-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
        }
        
        .product-card {
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
        }
        
        .product-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0,0,0,0.15);
        }
        
        .product-image {
            width: 100%;
            height: 200px;
            object-fit: cover;
            background: #f8f9fa;
        }
        
        .product-info {
            padding: 20px;
        }
        
        .product-title {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 10px;
            line-height: 1.4;
            color: #333;
        }
        
        .product-price {
            font-size: 1.5rem;
            font-weight: 700;
            color: #28a745;
            margin-bottom: 10px;
        }
        
        .product-vendor {
            color: #666;
            margin-bottom: 15px;
        }
        
        .product-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 15px;
        }
        
        .badge {
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 500;
        }
        
        .badge-sale {
            background: #dc3545;
            color: white;
        }
        
        .badge-shipping {
            background: #17a2b8;
            color: white;
        }
        
        .badge-reliability {
            background: #ffc107;
            color: #333;
        }
        
        .product-button {
            width: 100%;
            padding: 12px 20px;
            background: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 8px;
            text-align: center;
            display: inline-block;
            transition: all 0.3s ease;
        }
        
        .product-button:hover {
            background: #5a6fd8;
            transform: translateY(-1px);
        }
        
        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            border-left: 4px solid #dc3545;
        }
        
        .features {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 40px;
            border-radius: 20px;
            margin-bottom: 40px;
        }
        
        .features h2 {
            color: white;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2rem;
        }
        
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 30px;
        }
        
        .feature {
            text-align: center;
            color: white;
        }
        
        .feature-icon {
            font-size: 3rem;
            margin-bottom: 15px;
        }
        
        .feature h3 {
            margin-bottom: 10px;
            font-size: 1.3rem;
        }
        
        .feature p {
            opacity: 0.9;
            line-height: 1.6;
        }
        
        .footer {
            background: rgba(0, 0, 0, 0.2);
            color: white;
            padding: 30px 0;
            text-align: center;
        }
        
        .image-preview {
            max-width: 200px;
            max-height: 200px;
            border-radius: 10px;
            margin-top: 10px;
            border: 2px solid #e9ecef;
        }
        
        @media (max-width: 768px) {
            .form-row {
                grid-template-columns: 1fr;
            }
            
            .search-tabs {
                flex-direction: column;
            }
            
            .product-grid {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .search-section {
                padding: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <h1>🔧 Auto Parts Finder USA</h1>
            <p>Find genuine auto parts from trusted US retailers</p>
        </div>
    </div>

    <div class="container">
        <div class="config-notice" id="configNotice">
            <strong>⚠️ Configuration Notice:</strong> SerpAPI key not configured. Search functionality will be limited. 
            Please configure the SERPAPI_KEY environment variable with your SerpAPI key.
        </div>

        <div class="search-section">
            <div class="search-tabs">
                <button class="tab-button active" onclick="switchTab('text')">🔍 Text Search</button>
                <button class="tab-button" onclick="switchTab('image')">📷 Image Search</button>
            </div>

            <!-- Text Search Form -->
            <form id="textSearchForm" class="search-form active">
                <div class="form-group">
                    <label for="searchQuery">Search for auto parts:</label>
                    <input type="text" id="searchQuery" placeholder="e.g., brake pads, oil filter, spark plugs..." required>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label for="minPrice">Min Price ($):</label>
                        <input type="number" id="minPrice" placeholder="0" min="0" step="0.01">
                    </div>
                    <div class="form-group">
                        <label for="maxPrice">Max Price ($):</label>
                        <input type="number" id="maxPrice" placeholder="1000" min="0" step="0.01">
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label for="sortBy">Sort by:</label>
                        <select id="sortBy">
                            <option value="price_asc">Price: Low to High</option>
                            <option value="price_desc">Price: High to Low</option>
                            <option value="reliability">Reliability</option>
                            <option value="relevance">Relevance</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="maxResults">Max Results:</label>
                        <select id="maxResults">
                            <option value="5">5 results</option>
                            <option value="10">10 results</option>
                            <option value="15">15 results</option>
                            <option value="20">20 results</option>
                        </select>
                    </div>
                </div>
                
                <button type="submit" class="search-button">Search Parts</button>
            </form>

            <!-- Image Search Form -->
            <form id="imageSearchForm" class="search-form">
                <div class="form-group">
                    <label for="imageFile">Upload part image:</label>
                    <input type="file" id="imageFile" accept="image/*" required>
                    <img id="imagePreview" class="image-preview" style="display: none;">
                </div>
                
                <div class="form-group">
                    <label for="fallbackText">Fallback text (optional):</label>
                    <input type="text" id="fallbackText" placeholder="Enter part number if image doesn't work...">
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label for="enhanceImage">Enhance image:</label>
                        <select id="enhanceImage">
                            <option value="true">Yes (recommended)</option>
                            <option value="false">No</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="confidenceThreshold">OCR Confidence:</label>
                        <select id="confidenceThreshold">
                            <option value="0.3">Low (30%)</option>
                            <option value="0.5" selected>Medium (50%)</option>
                            <option value="0.7">High (70%)</option>
                        </select>
                    </div>
                </div>
                
                <button type="submit" class="search-button">Search by Image</button>
            </form>
        </div>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Searching for parts...</p>
        </div>

        <div class="results" id="results">
            <div class="results-header">
                <h2 id="resultsTitle">Search Results</h2>
                <p id="resultsInfo"></p>
            </div>
            <div class="product-grid" id="productGrid"></div>
        </div>

        <div class="features">
            <div class="container">
                <h2>Why Choose Auto Parts Finder USA?</h2>
                <div class="features-grid">
                    <div class="feature">
                        <div class="feature-icon">🔍</div>
                        <h3>Smart Search</h3>
                        <p>Advanced search algorithms find the exact parts you need from trusted US retailers.</p>
                    </div>
                    <div class="feature">
                        <div class="feature-icon">🛡️</div>
                        <h3>Anti-Fraud Protection</h3>
                        <p>Built-in fraud detection ensures you only see genuine, authentic auto parts.</p>
                    </div>
                    <div class="feature">
                        <div class="feature-icon">📷</div>
                        <h3>Image Recognition</h3>
                        <p>Upload a photo of your part and let our OCR technology identify it instantly.</p>
                    </div>
                    <div class="feature">
                        <div class="feature-icon">💰</div>
                        <h3>Best Prices</h3>
                        <p>Compare prices across multiple retailers to find the best deals on quality parts.</p>
                    </div>
                    <div class="feature">
                        <div class="feature-icon">🇺🇸</div>
                        <h3>US Only</h3>
                        <p>Exclusively searches US-based retailers for faster shipping and better service.</p>
                    </div>
                    <div class="feature">
                        <div class="feature-icon">⚡</div>
                        <h3>Fast Results</h3>
                        <p>Get instant results with cached searches and optimized performance.</p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="footer">
        <div class="container">
            <p>&copy; 2024 Auto Parts Finder USA. All rights reserved. | Professional auto parts search system.</p>
        </div>
    </div>

    <script>
        // Check configuration on page load
        fetch('/api/v1/health/detailed')
            .then(response => response.json())
            .then(data => {
                if (data.components && data.components.serpapi && !data.components.serpapi.api_key_configured) {
                    document.getElementById('configNotice').classList.add('show');
                }
            })
            .catch(err => {
                console.log('Health check failed:', err);
            });

        // Tab switching
        function switchTab(tab) {
            // Update tab buttons
            document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            // Update forms
            document.querySelectorAll('.search-form').forEach(form => form.classList.remove('active'));
            document.getElementById(tab + 'SearchForm').classList.add('active');
        }

        // Image preview
        document.getElementById('imageFile').addEventListener('change', function(e) {
            const file = e.target.files[0];
            const preview = document.getElementById('imagePreview');
            
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                };
                reader.readAsDataURL(file);
            } else {
                preview.style.display = 'none';
            }
        });

        // Text search form submission
        document.getElementById('textSearchForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const query = document.getElementById('searchQuery').value;
            const minPrice = document.getElementById('minPrice').value;
            const maxPrice = document.getElementById('maxPrice').value;
            const sortBy = document.getElementById('sortBy').value;
            const maxResults = document.getElementById('maxResults').value;
            
            if (!query.trim()) {
                alert('Please enter a search query');
                return;
            }
            
            await performSearch('text', {
                query: query,
                min_price: minPrice ? parseFloat(minPrice) : null,
                max_price: maxPrice ? parseFloat(maxPrice) : null,
                sort_by: sortBy,
                max_results: parseInt(maxResults)
            });
        });

        // Image search form submission
        document.getElementById('imageSearchForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const imageFile = document.getElementById('imageFile').files[0];
            const fallbackText = document.getElementById('fallbackText').value;
            const enhanceImage = document.getElementById('enhanceImage').value === 'true';
            const confidenceThreshold = parseFloat(document.getElementById('confidenceThreshold').value);
            
            if (!imageFile) {
                alert('Please select an image file');
                return;
            }
            
            const formData = new FormData();
            formData.append('image', imageFile);
            formData.append('max_results', '5');
            formData.append('enhance_image', enhanceImage);
            formData.append('confidence_threshold', confidenceThreshold);
            if (fallbackText) {
                formData.append('ocr_fallback_text', fallbackText);
            }
            
            await performSearch('image', formData);
        });

        // Perform search function
        async function performSearch(type, data) {
            const loading = document.getElementById('loading');
            const results = document.getElementById('results');
            const searchButton = document.querySelector('.search-button');
            
            // Show loading
            loading.style.display = 'block';
            results.classList.remove('show');
            searchButton.disabled = true;
            searchButton.textContent = 'Searching...';
            
            try {
                let response;
                
                if (type === 'text') {
                    // Build query string for GET request
                    const params = new URLSearchParams();
                    params.append('q', data.query);
                    if (data.min_price !== null) params.append('min_price', data.min_price);
                    if (data.max_price !== null) params.append('max_price', data.max_price);
                    params.append('sort_by', data.sort_by);
                    params.append('max_results', data.max_results);
                    
                    response = await fetch(`/api/v1/search/text?${params}`);
                } else {
                    // Image search with FormData
                    response = await fetch('/api/v1/search/image', {
                        method: 'POST',
                        body: data
                    });
                }
                
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail?.message || errorData.message || 'Search failed');
                }
                
                const result = await response.json();
                displayResults(result);
                
            } catch (error) {
                console.error('Search error:', error);
                displayError(error.message);
            } finally {
                // Hide loading
                loading.style.display = 'none';
                searchButton.disabled = false;
                searchButton.textContent = type === 'text' ? 'Search Parts' : 'Search by Image';
            }
        }

        // Display results function
        function displayResults(data) {
            const results = document.getElementById('results');
            const resultsTitle = document.getElementById('resultsTitle');
            const resultsInfo = document.getElementById('resultsInfo');
            const productGrid = document.getElementById('productGrid');
            
            // Update header
            resultsTitle.textContent = `Found ${data.total_count} parts`;
            resultsInfo.innerHTML = `
                <strong>Query:</strong> ${data.original_query} |
                <strong>Search Time:</strong> ${data.metrics.search_time_ms}ms |
                <strong>Source:</strong> ${data.search_type} search
            `;
            
            // Clear and populate product grid
            productGrid.innerHTML = '';
            
            if (data.products.length === 0) {
                productGrid.innerHTML = '<div class="error">No parts found. Try a different search term or check if SerpAPI is configured properly.</div>';
            } else {
                data.products.forEach(product => {
                    const productCard = createProductCard(product);
                    productGrid.appendChild(productCard);
                });
            }
            
            // Show results
            results.classList.add('show');
            
            // Scroll to results
            results.scrollIntoView({ behavior: 'smooth' });
        }

        // Create product card function
        function createProductCard(product) {
            const card = document.createElement('div');
            card.className = 'product-card';
            
            const badges = [];
            if (product.is_on_sale) {
                badges.push('<span class="badge badge-sale">ON SALE</span>');
            }
            if (product.shipping_speed !== 'standard') {
                badges.push(`<span class="badge badge-shipping">${product.shipping_speed.replace('_', ' ').toUpperCase()}</span>`);
            }
            if (product.reliability_score >= 4) {
                badges.push(`<span class="badge badge-reliability">${'⭐'.repeat(product.reliability_score)}</span>`);
            }
            
            card.innerHTML = `
                ${product.primary_image_url 
                    ? `<img src="${product.primary_image_url}" alt="${product.title}" class="product-image" onerror="this.style.display='none'">` 
                    : '<div class="product-image" style="display: flex; align-items: center; justify-content: center; background: #f8f9fa; color: #666;">No Image</div>'
                }
                <div class="product-info">
                    <h3 class="product-title">${product.title}</h3>
                    <div class="product-price">$${product.price}</div>
                    <div class="product-vendor">Sold by: ${product.vendor_name}</div>
                    <div class="product-badges">
                        ${badges.join('')}
                    </div>
                    <a href="${product.product_url}" target="_blank" class="product-button">
                        View Product
                    </a>
                </div>
            `;
            
            return card;
        }

        // Display error function
        function displayError(message) {
            const results = document.getElementById('results');
            const productGrid = document.getElementById('productGrid');
            
            document.getElementById('resultsTitle').textContent = 'Search Error';
            document.getElementById('resultsInfo').textContent = '';
            
            // Enhanced error messages
            let errorDetails = '';
            if (message.includes('API key')) {
                errorDetails = `
                    <br><br>
                    <strong>Configuration Issue:</strong> This appears to be a SerpAPI configuration problem.
                    <br>Please ensure the SERPAPI_KEY environment variable is set correctly.
                `;
            } else if (message.includes('rate limit') || message.includes('límite')) {
                errorDetails = `
                    <br><br>
                    <strong>Rate Limit:</strong> Too many requests. Please wait a few minutes before trying again.
                `;
            } else if (message.includes('timeout') || message.includes('tiempo')) {
                errorDetails = `
                    <br><br>
                    <strong>Timeout:</strong> The search took too long. Please try again with a simpler search term.
                `;
            }
            
            productGrid.innerHTML = `
                <div class="error">
                    <strong>Error:</strong> ${message}
                    ${errorDetails}
                    <br><br>
                    Please try again with a different search term or check your internet connection.
                </div>
            `;
            
            results.classList.add('show');
        }

        // Auto-focus search input
        document.getElementById('searchQuery').focus();
    </script>
</body>
</html>
    """, media_type="text/html")

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Interactive Documentation",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui.css",
    )

@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - ReDoc Documentation",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@2.1.0/bundles/redoc.standalone.js",
    )

# ENDPOINTS DE BÚSQUEDA (ENHANCED)
@app.post("/api/v1/search/text", response_model=SearchResponse)
async def search_parts_by_text(request: TextSearchRequest):
    """
    🔍 Búsqueda de repuestos automotrices por texto
    
    Realiza búsqueda inteligente con:
    - ✅ Filtrado geográfico (solo EE.UU.)
    - ✅ Validación de precios
    - ✅ Detección de fraude
    - ✅ Ordenamiento por precio/confiabilidad
    """
    if len(request.query.strip()) < 3:
        raise InvalidSearchQueryError(request.query, "Consulta muy corta")
    
    search_id = search_service.generate_search_id()
    return await search_service.process_text_search(request, search_id)

@app.get("/api/v1/search/text", response_model=SearchResponse)
async def search_parts_by_text_get(
    q: str = Query(..., min_length=3, max_length=200),
    max_results: int = Query(5, ge=1, le=20),
    min_price: Optional[float] = Query(None, ge=0),
    max_price: Optional[float] = Query(None, ge=0),
    sort_by: SortOption = Query(SortOption.PRICE_LOW_TO_HIGH)
):
    """🔍 Búsqueda por GET (navegador)"""
    filters = SearchFilters()
    if min_price is not None or max_price is not None:
        filters.price_range = PriceRange(min_price=min_price, max_price=max_price)
    
    search_request = TextSearchRequest(
        query=q,
        max_results=max_results,
        filters=filters,
        sort_by=sort_by
    )
    
    return await search_parts_by_text(search_request)

@app.post("/api/v1/search/image", response_model=SearchResponse)
async def search_parts_by_image(
    image: UploadFile = File(..., description="Imagen del repuesto"),
    max_results: int = Form(5, ge=1, le=20),
    ocr_fallback_text: Optional[str] = Form(None),
    enhance_image: bool = Form(True),
    confidence_threshold: float = Form(0.5, ge=0.0, le=1.0)
):
    """
    📷 Búsqueda de repuestos por imagen
    
    Características:
    - 🔍 OCR optimizado para números de parte
    - 🖼️ Preprocesamiento automático de imágenes
    - 🏷️ Detección de marcas y especificaciones
    - 📝 Texto de respaldo opcional
    """
    if not OCR_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Image search not available - OCR libraries not installed"
        )
    
    search_id = search_service.generate_search_id()
    
    try:
        image_processor = await get_image_processor()
        image_data = await image.read()
        
        image_info = await image_processor.process_image_upload(image_data, image.filename or "image")
        
        ocr_result = await image_processor.extract_text_from_image(
            image_data=image_data,
            enhance_image=enhance_image,
            confidence_threshold=confidence_threshold
        )
        
        search_text = ""
        if ocr_result.extracted_text and ocr_result.confidence >= confidence_threshold:
            search_text = ocr_result.extracted_text
        elif ocr_fallback_text:
            search_text = ocr_fallback_text
        else:
            raise OCRProcessingError("No se pudo extraer texto de la imagen")
        
        search_request = TextSearchRequest(
            query=search_text,
            max_results=max_results,
            filters=SearchFilters()
        )
        
        response = await search_service.process_text_search(search_request, search_id)
        
        response.search_type = "image"
        response.original_query = f"Imagen: {image.filename}"
        response.processed_query = search_text
        
        return response
    
    except (InvalidImageError, OCRProcessingError):
        raise
    except Exception as e:
        logger.error(f"Error en búsqueda por imagen: {e}")
        raise HTTPException(status_code=500, detail="Error procesando imagen")

@app.get("/api/v1/search/suggestions")
async def get_search_suggestions(
    q: str = Query(..., min_length=1, max_length=100),
    limit: int = Query(5, ge=1, le=20)
):
    """💡 Sugerencias de búsqueda inteligentes"""
    suggestions = []
    q_lower = q.lower()
    
    auto_parts_suggestions = {
        "brake": ["brake pads", "brake rotors", "brake calipers", "brake fluid"],
        "engine": ["engine oil", "engine filter", "engine mount", "engine belt"],
        "alternator": ["alternator belt", "alternator pulley"],
        "starter": ["starter motor", "starter relay"],
        "filter": ["oil filter", "air filter", "fuel filter", "cabin filter"],
        "belt": ["timing belt", "serpentine belt", "drive belt"],
        "pump": ["water pump", "fuel pump", "power steering pump"],
        "sensor": ["oxygen sensor", "temperature sensor", "pressure sensor"],
    }
    
    for category, parts in auto_parts_suggestions.items():
        if category.startswith(q_lower) or q_lower in category:
            suggestions.extend(parts)
        else:
            for part in parts:
                if q_lower in part:
                    suggestions.append(part)
    
    unique_suggestions = list(dict.fromkeys(suggestions))
    
    return {
        "query": q,
        "suggestions": unique_suggestions[:limit],
        "total_suggestions": len(unique_suggestions[:limit])
    }

# ENDPOINTS DE SALUD (ENHANCED)
@app.get("/api/v1/health")
async def health_check():
    """❤️ Verificación básica de salud del sistema"""
    try:
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "service": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": time.time(),
                "error": str(e),
                "service": settings.PROJECT_NAME
            }
        )

@app.get("/api/v1/health/detailed")
async def detailed_health_check():
    """🔍 Verificación detallada de salud de componentes"""
    components = {}
    overall_status = "healthy"
    
    # Check SerpAPI
    try:
        if settings.serpapi_configured:
            components["serpapi"] = {
                "status": "healthy",
                "message": "SerpAPI key configured",
                "api_key_configured": True
            }
        else:
            components["serpapi"] = {
                "status": "degraded",
                "message": "SerpAPI key not configured properly",
                "api_key_configured": False
            }
            overall_status = "degraded"
    except Exception as e:
        components["serpapi"] = {
            "status": "unhealthy",
            "message": f"SerpAPI error: {str(e)[:100]}"
        }
        overall_status = "degraded"
    
    # Check OCR
    try:
        if settings.OCR_ENABLED and OCR_AVAILABLE:
            pytesseract.get_tesseract_version()
            components["ocr"] = {
                "status": "healthy",
                "message": "Tesseract OCR available"
            }
        else:
            components["ocr"] = {
                "status": "disabled" if not settings.OCR_ENABLED else "unavailable",
                "message": "OCR disabled in configuration" if not settings.OCR_ENABLED else "OCR libraries not available"
            }
    except Exception as e:
        components["ocr"] = {
            "status": "unhealthy",
            "message": f"OCR error: {str(e)[:100]}"
        }
        overall_status = "degraded"
    
    # Check cache
    try:
        cache = _cache
        test_key = f"health_test_{time.time()}"
        cache.set(test_key, {"test": True})
        retrieved = cache.get(test_key)
        
        components["cache"] = {
            "status": "healthy" if retrieved else "degraded",
            "message": "Memory cache functional"
        }
    except Exception as e:
        components["cache"] = {
            "status": "unhealthy",
            "message": f"Cache error: {str(e)[:100]}"
        }
        overall_status = "degraded"
    
    # Check system resources
    try:
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        system_status = "healthy"
        if cpu_percent > 90 or memory.percent > 90:
            system_status = "degraded"
            overall_status = "degraded"
        
        components["system"] = {
            "status": system_status,
            "message": f"CPU: {cpu_percent:.1f}%, Memory: {memory.percent:.1f}%",
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": round(memory.available / (1024**3), 2)
        }
    except Exception as e:
        components["system"] = {
            "status": "unknown",
            "message": f"System metrics error: {str(e)[:100]}"
        }
    
    return {
        "overall_status": overall_status,
        "timestamp": time.time(),
        "components": components,
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "configuration": {
            "serpapi_configured": settings.serpapi_configured,
            "ocr_enabled": settings.OCR_ENABLED,
            "cache_enabled": settings.CACHE_ENABLED,
            "rate_limiting_enabled": settings.RATE_LIMIT_ENABLED,
            "environment": settings.ENVIRONMENT
        }
    }

@app.get("/api/v1/health/metrics")
async def get_system_metrics():
    """📊 Métricas del sistema"""
    try:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        process = psutil.Process()
        
        metrics = {
            "timestamp": time.time(),
            "system": {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "memory_used_gb": round(memory.used / (1024**3), 2),
                "memory_percent": memory.percent,
                "disk_total_gb": round(disk.total / (1024**3), 2),
                "disk_used_gb": round(disk.used / (1024**3), 2),
                "disk_percent": round(disk.used / disk.total * 100, 1)
            },
            "process": {
                "pid": process.pid,
                "cpu_percent": process.cpu_percent(),
                "memory_percent": process.memory_percent(),
                "memory_rss_mb": round(process.memory_info().rss / (1024**2), 2),
                "threads": process.num_threads(),
                "uptime_seconds": round(time.time() - process.create_time(), 1)
            },
            "application": {
                "name": settings.PROJECT_NAME,
                "version": settings.VERSION,
                "environment": settings.ENVIRONMENT,
                "features": {
                    "ocr_enabled": settings.OCR_ENABLED and OCR_AVAILABLE,
                    "cache_enabled": settings.CACHE_ENABLED,
                    "rate_limiting_enabled": settings.RATE_LIMIT_ENABLED,
                    "price_verification_enabled": settings.PRICE_VERIFICATION_ENABLED
                }
            }
        }
        
        return metrics
    
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error obteniendo métricas: {e}")

# MANEJO DE ERRORES (ENHANCED)
@app.exception_handler(AutoPartsBaseException)
async def auto_parts_exception_handler(request: Request, exc: AutoPartsBaseException):
    http_exc = to_http_exception(exc)
    logger.warning(f"AutoParts exception: {exc.error_code} - {exc.message}")
    return JSONResponse(
        status_code=http_exc.status_code,
        content=http_exc.detail
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Error no controlado en {request.method} {request.url}: {exc}")
    
    # More helpful error messages based on common issues
    error_message = "Ha ocurrido un error inesperado. Por favor, intente nuevamente."
    
    if "SerpAPI" in str(exc) or "serpapi" in str(exc).lower():
        error_message = "Error de configuración de SerpAPI. Verifique la clave API."
    elif "timeout" in str(exc).lower():
        error_message = "Tiempo de espera agotado. Intente nuevamente en unos momentos."
    elif "connection" in str(exc).lower():
        error_message = "Error de conexión. Verifique su conexión a internet."
    
    if settings.is_development:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Error interno del servidor",
                "message": error_message,
                "detail": str(exc),
                "type": type(exc).__name__
            }
        )
    else:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Error interno del servidor",
                "message": error_message
            }
        )

# ================================================================
# FUNCIÓN PRINCIPAL (ENHANCED)
# ================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Enhanced configuration for Render.com
    uvicorn_config = {
        "app": "webapp:app",
        "host": settings.HOST,
        "port": settings.PORT,
        "log_level": settings.LOG_LEVEL.lower(),
        "access_log": True,
        "use_colors": False,  # Better for Render.com logs
        "loop": "asyncio"
    }
    
    # Only enable reload in development
    if settings.is_development:
        uvicorn_config["reload"] = True
        uvicorn_config["reload_dirs"] = ["."]
    
    # Production optimizations
    if settings.is_production:
        uvicorn_config["workers"] = 1  # Single worker for Render.com
        uvicorn_config["backlog"] = 2048
        uvicorn_config["limit_concurrency"] = 100
        uvicorn_config["limit_max_requests"] = 1000
    
    logger.info(f"🚀 Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    logger.info(f"🌍 Environment: {settings.ENVIRONMENT}")
    logger.info(f"🔗 Server will start on: http://{settings.HOST}:{settings.PORT}")
    
    if not settings.serpapi_configured:
        logger.warning("⚠️ SERPAPI_KEY not configured - search functionality will be limited")
        logger.info("💡 Set SERPAPI_KEY environment variable to enable full functionality")
    
    uvicorn.run(**uvicorn_config)
