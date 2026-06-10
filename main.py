from datetime import datetime
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="CLARUM Invest API",
    description="Backend académico en Python para cálculos financieros de CLARUM Invest.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_origin_regex=r"https://.*\.lovable\.app",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def inicio():
    return {
        "mensaje": "CLARUM Invest API está funcionando correctamente.",
        "estado": "ok",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "servicio": "CLARUM Invest API",
        "version": "0.1.0",
        "fecha_utc": datetime.utcnow().isoformat(),
    }


@app.post("/perfil-inversionista")
def perfil_inversionista(payload: Dict[str, Any]):
    return {
        "status": "ok",
        "modulo": "perfil-inversionista",
        "perfil_estimado": "moderado",
        "nivel_conocimiento": "básico",
        "mensaje": (
            "Este es un resultado de prueba. Más adelante este módulo clasificará "
            "el perfil del inversionista según conocimiento, tolerancia al riesgo, "
            "horizonte temporal, liquidez y objetivo financiero."
        ),
        "datos_recibidos": payload,
    }


@app.post("/analizar-portafolio")
def analizar_portafolio(payload: Dict[str, Any]):
    return {
        "status": "ok",
        "modulo": "analizar-portafolio",
        "resultado_demo": {
            "rentabilidad_anual_estimada": "8.00%",
            "volatilidad_estimada": "12.50%",
            "perfil_riesgo": "moderado",
            "explicacion": (
                "Este es un resultado de demostración. En la siguiente fase se conectarán "
                "librerías financieras de Python para calcular rentabilidad, volatilidad, "
                "correlación, drawdown, VaR y CVaR."
            ),
        },
        "datos_recibidos": payload,
    }


@app.post("/calcular-riesgo")
def calcular_riesgo(payload: Dict[str, Any]):
    return {
        "status": "ok",
        "modulo": "calcular-riesgo",
        "resultado_demo": {
            "nivel_riesgo": "medio",
            "var_demo": "5.20%",
            "cvar_demo": "7.80%",
            "explicacion": (
                "Este resultado es provisional. Posteriormente se calculará el riesgo "
                "con datos reales de mercado y series históricas."
            ),
        },
        "datos_recibidos": payload,
    }
