"""
Auto Parts Finder USA - Sistema Completo Unificado
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

import cv2
import httpx
import numpy as np
import psutil
import pytesseract
from bs4 import BeautifulSoup
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, BaseSettings, Field, HttpUrl, validator

# ================================================================
# CONFIGURACIÓN Y SETTINGS
# ================================================================

class Settings(BaseSettings):
    """Configuración principal del sistema"""
    
    # Configuración básica
    PROJECT_NAME: str = "Auto Parts Finder USA"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "Sistema profesional de búsqueda de repuestos automotrices en EE.UU."
    
    # Servidor
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    DEBUG: bool = Field(default=False, env="DEBUG")
    ENVIRONMENT: str = Field(default="production", env="ENVIRONMENT")
    
    # Seguridad
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32), env="SECRET_KEY")
    
    # SerpAPI (OBLIGATORIO)
    SERPAPI_KEY: str = Field(..., env="SERPAPI_KEY")
    SERPAPI_TIMEOUT: int = Field(default=30, env="SERPAPI_TIMEOUT")
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
    OCR_ENABLED: bool = Field(default=True, env="OCR_ENABLED")
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
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()

# ================================================================
# EXCEPCIONES PERSONALIZADAS
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
    def __init__(self, query: str):
        super().__init__(
            message="No se encontraron resultados",
            error_code="NO_RESULTS_FOUND",
            details={"query": query}
        )

class SerpAPIError(AutoPartsBaseException):
    def __init__(self, status_code: int, response_data: Dict[str, Any]):
        super().__init__(
            message=f"Error en SerpAPI (código {status_code})",
            error_code="SERPAPI_ERROR",
            details={"status_code": status_code, "response_data": response_data}
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
# MODELOS DE DATOS
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
    
    @validator('query')
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
# CLIENTE SERPAPI
# ================================================================

class SerpAPIClient:
    """Cliente optimizado para SerpAPI"""
    
    def __init__(self):
        self.api_key = settings.SERPAPI_KEY
        self.base_url = "https://serpapi.com/search"
        
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.SERPAPI_TIMEOUT),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
        )
        
        self._cache: Dict[str, Tuple[Dict, datetime]] = {}
        self._cache_ttl = timedelta(seconds=settings.CACHE_TTL_SECONDS)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def close(self):
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
        try:
            link = result.get("link", "")
            if not link:
                return False
            
            parsed_url = urlparse(link)
            domain = parsed_url.netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
            
            if domain in settings.TRUSTED_US_DOMAINS:
                return True
            
            us_extensions = [".com", ".us", ".org", ".net"]
            if any(domain.endswith(ext) for ext in us_extensions):
                international_indicators = [
                    ".ca", ".uk", ".de", ".fr", ".au", ".mx",
                    "canada", "europe", "international", "global"
                ]
                if not any(indicator in domain for indicator in international_indicators):
                    return True
            
            return False
        except Exception as e:
            logger.warning(f"Error validando resultado US: {e}")
            return False
    
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
            response = await self.client.get(self.base_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if "error" in data:
                    raise SerpAPIError(200, data)
                
                if use_cache:
                    self._cache[cache_key] = (data, datetime.utcnow())
                    if len(self._cache) > 100:
                        oldest_key = min(self._cache.keys(), 
                                       key=lambda k: self._cache[k][1])
                        del self._cache[oldest_key]
                
                shopping_results = data.get("shopping_results", [])
                validated_results = [r for r in shopping_results if self._validate_us_result(r)]
                
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
            
            else:
                raise SerpAPIError(response.status_code, {"error": f"HTTP {response.status_code}"})
        
        except httpx.TimeoutException:
            raise SerpAPIError(408, {"error": "Timeout"})
        except Exception as e:
            if isinstance(e, SerpAPIError):
                raise
            raise SerpAPIError(500, {"error": str(e)})

# ================================================================
# PROCESADOR DE IMÁGENES Y OCR
# ================================================================

class ImageProcessor:
    """Procesador de imágenes con OCR optimizado"""
    
    def __init__(self):
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
            
            if image.format.lower() not in ['jpeg', 'jpg', 'png', 'webp', 'bmp']:
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
# SERVICIOS AUXILIARES
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
# SERVICIOS GLOBALES
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
# SERVICIO DE BÚSQUEDA
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
            
            price_validator = get_price_validator()
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
            cache = await get_cache_manager()
            cache_key = f"search_{hash(request.query + str(request.filters.dict()))}"
            cached_result = cache.get(cache_key)
            
            if cached_result:
                logger.info(f"Cache hit para: {request.query}")
                return cached_result
            
            serpapi_client = await get_serpapi_client()
            raw_results, search_metrics = await serpapi_client.search_auto_parts(
                query=request.get_enhanced_query(),
                max_results=request.max_results * 2
            )
            
            if not raw_results:
                raise NoResultsFoundError(request.query)
            
            geo_filter = await get_geo_filter()
            us_results = await geo_filter.filter_results(raw_results)
            
            if not us_results:
                raise NoResultsFoundError(request.query)
            
            products = []
            antifraud = await get_antifraud_analyzer()
            
            for raw_result in us_results:
                product = self._convert_raw_to_product(raw_result)
                if not product:
                    continue
                
                fraud_analysis = antifraud.analyze_product(raw_result)
                if fraud_analysis["should_block"]:
                    logger.info(f"Producto bloqueado por fraude: {product.title[:50]}")
                    continue
                
                products.append(product)
            
            filtered_products = self._apply_filters(products, request.filters)
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
        
        except Exception as e:
            if isinstance(e, (NoResultsFoundError, InvalidSearchQueryError)):
                raise
            logger.error(f"Error en búsqueda: {e}")
            raise HTTPException(status_code=500, detail="Error interno en búsqueda")

# ================================================================
# CONFIGURACIÓN DE LOGGING
# ================================================================

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# ================================================================
# APLICACIÓN FASTAPI
# ================================================================

# Instancia del servicio de búsqueda
search_service = SearchService()

# Contexto de ciclo de vida
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Iniciando Auto Parts Finder USA API...")
    
    if not settings.SERPAPI_KEY or settings.SERPAPI_KEY == "TU_SERPAPI_KEY_AQUI":
        logger.error("❌ SERPAPI_KEY no configurado correctamente")
        raise RuntimeError("SERPAPI_KEY requerido")
    
    if settings.OCR_ENABLED:
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            logger.info("✅ Tesseract OCR disponible")
        except Exception as e:
            logger.warning(f"⚠️ Tesseract no disponible: {e}")
    
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
    allow_origins=["*"] if settings.is_development else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting simple
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
# ENDPOINTS
# ================================================================

@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "description": settings.DESCRIPTION,
        "status": "operational",
        "documentation": "/docs",
        "health_check": "/api/v1/health",
        "features": [
            "🔍 Búsqueda inteligente de repuestos automotrices",
            "🌍 Filtrado geográfico estricto (solo EE.UU.)",
            "💰 Verificación de precios en tiempo real",
            "🛡️ Sistema antifraude avanzado",
            "📷 Búsqueda por imagen con OCR",
            "⚡ Caché para performance optimizada"
        ],
        "endpoints": {
            "text_search": "/api/v1/search/text",
            "image_search": "/api/v1/search/image",
            "suggestions": "/api/v1/search/suggestions",
            "health": "/api/v1/health",
            "metrics": "/api/v1/health/metrics"
        }
    }

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Documentación Interactiva",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui.css",
    )

@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Documentación ReDoc",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@2.1.0/bundles/redoc.standalone.js",
    )

# ENDPOINTS DE BÚSQUEDA
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
    search_id = search_service.generate_search_id()
    
    try:
        image_processor = await get_image_processor()
        image_data = await image.read()
        
        image_info = await image_processor.process_image_upload(image_data, image.filename)
        
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

# ENDPOINTS DE SALUD
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
    
    try:
        client = await get_serpapi_client()
        components["serpapi"] = {
            "status": "healthy",
            "message": "SerpAPI client initialized",
            "api_key_configured": bool(settings.SERPAPI_KEY and settings.SERPAPI_KEY != "TU_SERPAPI_KEY_AQUI")
        }
    except Exception as e:
        components["serpapi"] = {
            "status": "unhealthy",
            "message": f"SerpAPI error: {str(e)[:100]}"
        }
        overall_status = "degraded"
    
    try:
        if settings.OCR_ENABLED:
            import pytesseract
            pytesseract.get_tesseract_version()
            components["ocr"] = {
                "status": "healthy",
                "message": "Tesseract OCR available"
            }
        else:
            components["ocr"] = {
                "status": "disabled",
                "message": "OCR disabled in configuration"
            }
    except Exception as e:
        components["ocr"] = {
            "status": "unhealthy",
            "message": f"OCR error: {str(e)[:100]}"
        }
        overall_status = "degraded"
    
    try:
        cache = await get_cache_manager()
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
        "version": settings.VERSION
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
                    "ocr_enabled": settings.OCR_ENABLED,
                    "cache_enabled": settings.CACHE_ENABLED,
                    "rate_limiting_enabled": settings.RATE_LIMIT_ENABLED,
                    "price_verification_enabled": settings.PRICE_VERIFICATION_ENABLED
                }
            }
        }
        
        return metrics
    
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error obteniendo métricas: {e}")

# MANEJO DE ERRORES
@app.exception_handler(AutoPartsBaseException)
async def auto_parts_exception_handler(request: Request, exc: AutoPartsBaseException):
    http_exc = to_http_exception(exc)
    return JSONResponse(
        status_code=http_exc.status_code,
        content=http_exc.detail
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Error no controlado en {request.method} {request.url}: {exc}")
    
    if settings.is_development:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Error interno del servidor",
                "detail": str(exc),
                "type": type(exc).__name__
            }
        )
    else:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Error interno del servidor",
                "message": "Ha ocurrido un error inesperado. Por favor, intente nuevamente."
            }
        )

# ================================================================
# FUNCIÓN PRINCIPAL
# ================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "webapp:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.is_development,
        log_level="info"
    )