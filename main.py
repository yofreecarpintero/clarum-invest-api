from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from io import StringIO

import numpy as np
import pandas as pd
import requests
import math
import yfinance as yf

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI(
    title="CLARUM Invest API",
    description="Backend académico en Python para cálculos financieros de CLARUM Invest.",
    version="0.2.13",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://clarum-invest.lovable.app",
        "https://lovable.dev",
    ],
    allow_origin_regex=r"https://.*\.lovable\.app",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# MÓDULO PERFIL ACADÉMICO, ENCUESTA Y MATRIZ DE RESTRICCIONES
# ============================================================

class ProfileRequest(BaseModel):
    respuestas: Optional[List[str]] = None
    answers: Optional[Dict[str, str]] = None


RISK_QUESTIONS = ["q1", "q2", "q3", "q4", "q5"]
KNOWLEDGE_QUESTIONS = ["q6", "q7", "q8", "q9", "q10"]

ANSWER_SCORE = {
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
}


RISK_KNOWLEDGE_MATRIX = {
    "Conservador": {
        "Básico": "Protección alta",
        "Intermedio": "Protección moderada",
        "Avanzado": "Protección flexible",
    },
    "Moderado": {
        "Básico": "Crecimiento limitado",
        "Intermedio": "Crecimiento prudente",
        "Avanzado": "Crecimiento balanceado",
    },
    "Dinámico": {
        "Básico": "Riesgo condicionado",
        "Intermedio": "Crecimiento dinámico controlado",
        "Avanzado": "Crecimiento dinámico",
    },
    "Agresivo": {
        "Básico": "Riesgo no habilitado",
        "Intermedio": "Riesgo alto restringido",
        "Avanzado": "Riesgo alto permitido",
    },
}


def normalizar_respuestas_perfil(request: ProfileRequest) -> Dict[str, str]:
    """
    Permite recibir las respuestas en dos formatos:
    1. Formato Lovable:
       {"respuestas": ["A","B","C","D","A","B","C","D","A","B"]}

    2. Formato estructurado:
       {"answers": {"q1": "A", "q2": "B", ...}}
    """

    if request.answers:
        return request.answers

    if request.respuestas:
        if len(request.respuestas) != 10:
            raise HTTPException(
                status_code=400,
                detail="La encuesta debe contener exactamente 10 respuestas.",
            )

        return {
            f"q{index + 1}": answer
            for index, answer in enumerate(request.respuestas)
        }

    raise HTTPException(
        status_code=400,
        detail="El cuerpo de la solicitud debe incluir 'respuestas' o 'answers'.",
    )


def calcular_puntaje_perfil(answers: Dict[str, str], questions: List[str]) -> int:
    total = 0

    for question in questions:
        answer = str(answers.get(question, "")).strip().upper()

        if answer not in ANSWER_SCORE:
            raise HTTPException(
                status_code=400,
                detail=f"La respuesta de {question} no es válida. Debe ser A, B, C o D.",
            )

        total += ANSWER_SCORE[answer]

    return total


def clasificar_perfil_riesgo(score: int) -> str:
    if score <= 8:
        return "Conservador"
    elif score <= 12:
        return "Moderado"
    elif score <= 16:
        return "Dinámico"
    return "Agresivo"


def clasificar_nivel_conocimiento(score: int) -> str:
    if score <= 8:
        return "Básico"
    elif score <= 14:
        return "Intermedio"
    return "Avanzado"


def obtener_restricciones_academicas(
    risk_profile: str,
    knowledge_level: str,
) -> Dict[str, Any]:

    matrix_category = RISK_KNOWLEDGE_MATRIX[risk_profile][knowledge_level]

    restrictions = {
        "categoria_matriz": matrix_category,
        "allowed_asset_classes": [],
        "blocked_asset_classes": [
            "Derivados",
            "Criptomonedas",
            "ETF apalancados",
            "ETF inversos",
            "Productos estructurados complejos",
        ],
        "max_equity_percentage": 0,
        "min_fixed_income_percentage": 0,
        "max_weight_per_asset": 0,
        "short_positions_allowed": False,
        "leverage_allowed": False,
        "minimum_history_years": 3,
        "allowed_simulations": [],
        "academic_warning": "",
    }

    if matrix_category == "Protección alta":
        restrictions.update({
            "allowed_asset_classes": [
                "ETF renta fija corto plazo",
                "ETF renta fija agregada",
                "Cartera modelo conservadora",
            ],
            "max_equity_percentage": 20,
            "min_fixed_income_percentage": 80,
            "max_weight_per_asset": 15,
            "allowed_simulations": [
                "Cartera equiponderada",
                "Cartera de mínima varianza",
            ],
            "academic_warning": (
                "El análisis debe priorizar instrumentos simples, diversificados "
                "y de menor volatilidad histórica."
            ),
        })

    elif matrix_category in ["Protección moderada", "Crecimiento limitado"]:
        restrictions.update({
            "allowed_asset_classes": [
                "ETF renta fija corto plazo",
                "ETF renta fija agregada",
                "ETF renta variable diversificada",
                "Cartera modelo conservadora",
                "Cartera modelo moderada",
            ],
            "max_equity_percentage": 35,
            "min_fixed_income_percentage": 65,
            "max_weight_per_asset": 20,
            "allowed_simulations": [
                "Cartera equiponderada",
                "Cartera de mínima varianza",
                "Comparación riesgo-rentabilidad",
            ],
            "academic_warning": (
                "El usuario puede analizar una exposición limitada a renta variable, "
                "siempre dentro de una estructura diversificada."
            ),
        })

    elif matrix_category in ["Crecimiento prudente", "Crecimiento balanceado"]:
        restrictions.update({
            "allowed_asset_classes": [
                "ETF renta fija",
                "ETF renta variable diversificada",
                "Acciones individuales de baja o media volatilidad",
                "Cartera modelo moderada",
            ],
            "max_equity_percentage": 60,
            "min_fixed_income_percentage": 40,
            "max_weight_per_asset": 20,
            "allowed_simulations": [
                "Cartera equiponderada",
                "Cartera de mínima varianza",
                "Cartera máximo Sharpe restringida",
                "Frontera eficiente académica",
            ],
            "academic_warning": (
                "El usuario puede analizar una cartera balanceada entre renta fija "
                "y renta variable, manteniendo control de concentración."
            ),
        })

    elif matrix_category in ["Riesgo condicionado", "Crecimiento dinámico controlado"]:
        restrictions.update({
            "allowed_asset_classes": [
                "ETF renta fija",
                "ETF renta variable diversificada",
                "Acciones individuales de media volatilidad",
                "Acciones individuales de alta volatilidad con límite reducido",
            ],
            "max_equity_percentage": 75,
            "min_fixed_income_percentage": 25,
            "max_weight_per_asset": 25,
            "allowed_simulations": [
                "Cartera equiponderada",
                "Cartera de mínima varianza",
                "Cartera máximo Sharpe restringida",
                "Monte Carlo académico",
                "Frontera eficiente académica",
            ],
            "academic_warning": (
                "El usuario puede analizar mayor exposición a renta variable, "
                "pero con límites de concentración."
            ),
        })

    elif matrix_category == "Riesgo no habilitado":
        restrictions.update({
            "allowed_asset_classes": [
                "ETF renta fija",
                "ETF renta variable diversificada con límite bajo",
                "Cartera modelo moderada",
            ],
            "max_equity_percentage": 40,
            "min_fixed_income_percentage": 60,
            "max_weight_per_asset": 15,
            "allowed_simulations": [
                "Cartera equiponderada",
                "Cartera de mínima varianza",
                "Simulación educativa comparativa",
            ],
            "academic_warning": (
                "Aunque el usuario manifiesta alta tolerancia al riesgo, su bajo "
                "conocimiento financiero limita el acceso académico a activos complejos "
                "o muy volátiles."
            ),
        })

    elif matrix_category == "Riesgo alto restringido":
        restrictions.update({
            "allowed_asset_classes": [
                "ETF renta fija",
                "ETF renta variable diversificada",
                "Acciones individuales de media volatilidad",
                "Acciones individuales de alta volatilidad con límite reducido",
            ],
            "max_equity_percentage": 80,
            "min_fixed_income_percentage": 20,
            "max_weight_per_asset": 25,
            "allowed_simulations": [
                "Cartera equiponderada",
                "Cartera máximo Sharpe restringida",
                "Monte Carlo académico",
                "Frontera eficiente académica",
            ],
            "academic_warning": (
                "El usuario puede analizar activos de mayor riesgo, pero el sistema "
                "mantiene restricciones por su nivel de conocimiento intermedio."
            ),
        })

    elif matrix_category == "Riesgo alto permitido":
        restrictions.update({
            "allowed_asset_classes": [
                "ETF renta fija",
                "ETF renta variable diversificada",
                "Acciones individuales",
                "Acciones individuales de alta volatilidad",
            ],
            "max_equity_percentage": 100,
            "min_fixed_income_percentage": 0,
            "max_weight_per_asset": 30,
            "allowed_simulations": [
                "Cartera equiponderada",
                "Cartera de mínima varianza",
                "Cartera máximo Sharpe restringida",
                "Monte Carlo académico",
                "Frontera eficiente académica",
            ],
            "academic_warning": (
                "El usuario presenta mayor tolerancia al riesgo y mayor conocimiento "
                "financiero. Aun así, el modelo no permite apalancamiento ni posiciones cortas."
            ),
        })

    return restrictions


def generar_mensaje_perfil_academico(
    risk_profile: str,
    knowledge_level: str,
    matrix_category: str,
    restrictions: Dict[str, Any],
) -> str:
    return (
        f"De acuerdo con las respuestas registradas, el usuario presenta un perfil de riesgo "
        f"{risk_profile.lower()} y un nivel de conocimiento financiero {knowledge_level.lower()}. "
        f"Al cruzar ambas variables, el modelo lo clasifica dentro de la categoría "
        f"'{matrix_category}'. Para este perfil, la exposición máxima permitida a renta variable "
        f"dentro del modelo académico es del {restrictions['max_equity_percentage']}%, "
        f"mientras que el peso máximo por activo será del {restrictions['max_weight_per_asset']}%. "
        f"Estas restricciones no constituyen una recomendación personalizada de inversión, "
        f"sino una regla académica de prudencia para orientar la simulación."
    )


@app.post("/api/profile/evaluate")
def evaluar_perfil_academico(request: ProfileRequest):
    answers = normalizar_respuestas_perfil(request)

    risk_score = calcular_puntaje_perfil(answers, RISK_QUESTIONS)
    knowledge_score = calcular_puntaje_perfil(answers, KNOWLEDGE_QUESTIONS)

    risk_profile = clasificar_perfil_riesgo(risk_score)
    knowledge_level = clasificar_nivel_conocimiento(knowledge_score)

    matrix_category = RISK_KNOWLEDGE_MATRIX[risk_profile][knowledge_level]

    restrictions = obtener_restricciones_academicas(
        risk_profile=risk_profile,
        knowledge_level=knowledge_level,
    )

    restricciones_resumen = [
    f"Exposición máxima a renta variable: {restrictions['max_equity_percentage']}%",
    f"Exposición mínima sugerida a renta fija: {restrictions['min_fixed_income_percentage']}%",
    f"Peso máximo por activo: {restrictions['max_weight_per_asset']}%",
    f"Historial mínimo requerido: {restrictions['minimum_history_years']} años",
    "Posiciones cortas: No permitidas" if not restrictions["short_positions_allowed"] else "Posiciones cortas: Permitidas",
    "Apalancamiento: No permitido" if not restrictions["leverage_allowed"] else "Apalancamiento: Permitido",
    ]

    message = generar_mensaje_perfil_academico(
        risk_profile=risk_profile,
        knowledge_level=knowledge_level,
        matrix_category=matrix_category,
        restrictions=restrictions,
    )

    resultado = {
        "risk_score": risk_score,
        "knowledge_score": knowledge_score,
        "perfil_riesgo": risk_profile,
        "nivel_conocimiento": knowledge_level,
        "matriz_riesgo_conocimiento": matrix_category,
        "restricciones": restrictions,
        "mensaje_academico": message,
        "restricciones_resumen": restricciones_resumen,
        "fecha_evaluacion_utc": datetime.now(timezone.utc).isoformat(),
        "profile_version": "1.0",
        "profile_type": "perfil_academico_inversionista",
    }

    return {
        "status": "ok",
        "modulo": "profile-evaluate",
        "resultado": resultado,
        **resultado,
    }
# =============================================================

def limpiar_tickers(activos: List[str]) -> List[str]:
    tickers = []

    for activo in activos:
        ticker = str(activo).strip().upper()
        if ticker:
            tickers.append(ticker)

    tickers_unicos = list(dict.fromkeys(tickers))

    if len(tickers_unicos) < 1:
        raise HTTPException(status_code=400, detail="Debes enviar al menos un activo válido.")

    if len(tickers_unicos) > 10:
        raise HTTPException(status_code=400, detail="Por ahora el análisis permite máximo 10 activos.")

    return tickers_unicos


def descargar_con_yahoo_chart(ticker: str, periodo: str) -> pd.Series:
    rango = periodo if periodo else "2y"

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

    params = {
        "range": rango,
        "interval": "1d",
        "includePrePost": "false",
        "events": "div,splits",
    }

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }

    respuesta = requests.get(url, params=params, headers=headers, timeout=25)

    if respuesta.status_code != 200:
        raise ValueError(f"Yahoo Chart respondió con estado {respuesta.status_code}: {respuesta.text[:200]}")

    data = respuesta.json()

    result = data.get("chart", {}).get("result")

    if not result:
        error = data.get("chart", {}).get("error")
        raise ValueError(f"Yahoo Chart no devolvió resultados. Error: {error}")

    result = result[0]
    timestamps = result.get("timestamp", [])
    quote = result.get("indicators", {}).get("quote", [{}])[0]
    closes = quote.get("close", [])

    if not timestamps or not closes:
        raise ValueError("Yahoo Chart no devolvió fechas o precios de cierre.")

    fechas = pd.to_datetime(timestamps, unit="s").tz_localize("UTC").tz_convert(None)
    serie = pd.Series(closes, index=fechas, name=ticker)
    serie = pd.to_numeric(serie, errors="coerce").dropna()

    if serie.empty or len(serie) < 30:
        raise ValueError("Yahoo Chart devolvió datos insuficientes.")

    return serie


def descargar_con_yfinance(ticker: str, periodo: str) -> pd.Series:
    data = yf.download(
        ticker,
        period=periodo,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if data.empty:
        raise ValueError("yfinance no devolvió datos.")

    if isinstance(data.columns, pd.MultiIndex):
        columnas = data.columns.tolist()

        posibles = [
            ("Close", ticker),
            ("Adj Close", ticker),
        ]

        for columna in posibles:
            if columna in columnas:
                serie = data[columna]
                serie.name = ticker
                serie = pd.to_numeric(serie, errors="coerce").dropna()

                if not serie.empty:
                    return serie

        raise ValueError(f"yfinance no encontró columna de cierre. Columnas: {columnas[:10]}")

    if "Close" in data.columns:
        serie = data["Close"]
    elif "Adj Close" in data.columns:
        serie = data["Adj Close"]
    else:
        raise ValueError(f"yfinance no encontró columna Close. Columnas: {list(data.columns)}")

    serie.name = ticker
    serie = pd.to_numeric(serie, errors="coerce").dropna()

    if serie.empty:
        raise ValueError("yfinance devolvió serie vacía después de limpiar.")

    return serie


def descargar_con_stooq(ticker: str, periodo: str) -> pd.Series:
    ticker_stooq = ticker.lower()

    if "." not in ticker_stooq:
        ticker_stooq = f"{ticker_stooq}.us"

    url = f"https://stooq.com/q/d/l/?s={ticker_stooq}&i=d"

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,text/plain,*/*",
    }

    respuesta = requests.get(url, headers=headers, timeout=25)

    if respuesta.status_code != 200:
        raise ValueError(f"Stooq respondió con estado {respuesta.status_code}: {respuesta.text[:200]}")

    texto = respuesta.text.strip()

    if not texto or "No data" in texto:
        raise ValueError(f"Stooq no devolvió datos. Respuesta: {texto[:200]}")

    data = pd.read_csv(StringIO(texto))

    if data.empty:
        raise ValueError(f"Stooq devolvió CSV vacío. Respuesta: {texto[:200]}")

    data.columns = [str(col).strip().title() for col in data.columns]

    if "Date" not in data.columns or "Close" not in data.columns:
        raise ValueError(f"Stooq no devolvió columnas válidas. Columnas: {list(data.columns)}. Respuesta: {texto[:200]}")

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data = data.dropna(subset=["Date", "Close"])
    data = data.set_index("Date").sort_index()

    serie = data["Close"]
    serie.name = ticker

    if periodo.endswith("y"):
        anios = int(periodo.replace("y", ""))
        fecha_inicio = serie.index.max() - pd.DateOffset(years=anios)
        serie = serie[serie.index >= fecha_inicio]

    if serie.empty or len(serie) < 30:
        raise ValueError("Stooq devolvió datos insuficientes después de filtrar periodo.")

    return serie


def descargar_precio_activo(ticker: str, periodo: str) -> Dict[str, Any]:
    errores = {}

    fuentes = [
        ("yahoo_chart", descargar_con_yahoo_chart),
        ("yfinance", descargar_con_yfinance),
        ("stooq", descargar_con_stooq),
    ]

    for nombre_fuente, funcion in fuentes:
        try:
            serie = funcion(ticker, periodo)
            return {
                "ticker": ticker,
                "fuente": nombre_fuente,
                "serie": serie,
            }
        except Exception as error:
            errores[nombre_fuente] = str(error)

    raise HTTPException(
        status_code=400,
        detail={
            "mensaje": f"No se pudieron obtener datos para el activo {ticker}.",
            "errores": errores,
        },
    )


def obtener_precios(activos: List[str], periodo: str = "2y") -> Dict[str, Any]:
    series = []
    fuentes_utilizadas = {}

    for ticker in activos:
        resultado = descargar_precio_activo(ticker, periodo)
        series.append(resultado["serie"])
        fuentes_utilizadas[ticker] = resultado["fuente"]

    precios_df = pd.concat(series, axis=1).dropna()

    if precios_df.empty or len(precios_df) < 30:
        raise HTTPException(
            status_code=400,
            detail="No hay suficientes datos históricos comunes para calcular el análisis.",
        )

    return {
        "precios": precios_df,
        "fuentes_utilizadas": fuentes_utilizadas,
    }


def obtener_pesos(payload: Dict[str, Any], activos: List[str]) -> np.ndarray:
    pesos_recibidos = payload.get("pesos")

    if pesos_recibidos is None:
        return np.array([1 / len(activos)] * len(activos))

    if not isinstance(pesos_recibidos, list):
        raise HTTPException(status_code=400, detail="El campo 'pesos' debe ser una lista de números.")

    if len(pesos_recibidos) != len(activos):
        raise HTTPException(status_code=400, detail="La cantidad de pesos debe coincidir con la cantidad de activos.")

    pesos = np.array(pesos_recibidos, dtype=float)

    if np.any(pesos < 0):
        raise HTTPException(status_code=400, detail="Los pesos no pueden ser negativos.")

    suma_pesos = pesos.sum()

    if suma_pesos <= 0:
        raise HTTPException(status_code=400, detail="La suma de los pesos debe ser mayor que cero.")

    return pesos / suma_pesos


def generar_explicacion(
    rentabilidad_anual: float,
    volatilidad_anual: float,
    max_drawdown: float,
    var_95: float,
    cvar_95: float,
    activos: List[str],
) -> str:
    rentabilidad_pct = rentabilidad_anual * 100
    volatilidad_pct = volatilidad_anual * 100
    drawdown_pct = max_drawdown * 100
    var_pct = var_95 * 100
    cvar_pct = cvar_95 * 100

    if volatilidad_anual < 0.10:
        nivel_riesgo = "bajo"
    elif volatilidad_anual < 0.20:
        nivel_riesgo = "medio"
    else:
        nivel_riesgo = "alto"

    return (
        f"El portafolio analizado con los activos {', '.join(activos)} presenta una "
        f"rentabilidad anualizada aproximada de {rentabilidad_pct:.2f}% y una volatilidad "
        f"anualizada de {volatilidad_pct:.2f}%. En términos sencillos, la rentabilidad "
        f"muestra cuánto habría crecido el portafolio en promedio anual según los datos "
        f"históricos, mientras que la volatilidad indica qué tanto pueden variar sus resultados. "
        f"Con base en la volatilidad observada, el nivel de riesgo histórico se clasifica como "
        f"{nivel_riesgo}. El máximo drawdown fue de {drawdown_pct:.2f}%. El VaR histórico "
        f"diario al 95% fue de {var_pct:.2f}% y el CVaR fue de {cvar_pct:.2f}%."
    )


def calcular_metricas_financieras(payload: Dict[str, Any]) -> Dict[str, Any]:
    activos = limpiar_tickers(payload.get("activos", []))
    periodo = str(payload.get("periodo", "2y"))

    resultado_precios = obtener_precios(activos, periodo)
    precios = resultado_precios["precios"]
    fuentes_utilizadas = resultado_precios["fuentes_utilizadas"]

    pesos = obtener_pesos(payload, activos)

    retornos_diarios = precios.pct_change().dropna()

    if retornos_diarios.empty:
        raise HTTPException(status_code=400, detail="No fue posible calcular retornos diarios.")

    retornos_portafolio = retornos_diarios.dot(pesos)

    rentabilidad_anual = float(retornos_portafolio.mean() * 252)
    volatilidad_anual = float(retornos_portafolio.std() * np.sqrt(252))

    acumulado = (1 + retornos_portafolio).cumprod()
    maximo_acumulado = acumulado.cummax()
    drawdown = acumulado / maximo_acumulado - 1
    max_drawdown = float(drawdown.min())

    percentil_5 = float(np.percentile(retornos_portafolio, 5))
    var_95 = max(0.0, -percentil_5)

    retornos_en_cola = retornos_portafolio[retornos_portafolio <= percentil_5]
    cvar_95 = max(0.0, -float(retornos_en_cola.mean())) if len(retornos_en_cola) > 0 else 0.0

    correlacion = retornos_diarios.corr().round(4).to_dict()

    rentabilidades_individuales = (retornos_diarios.mean() * 252).round(6).to_dict()
    volatilidades_individuales = (retornos_diarios.std() * np.sqrt(252)).round(6).to_dict()

    pesos_dict = {
        activo: round(float(peso), 4)
        for activo, peso in zip(activos, pesos)
    }

    explicacion = generar_explicacion(
        rentabilidad_anual=rentabilidad_anual,
        volatilidad_anual=volatilidad_anual,
        max_drawdown=max_drawdown,
        var_95=var_95,
        cvar_95=cvar_95,
        activos=activos,
    )

    return {
        "activos": activos,
        "periodo": periodo,
        "fecha_inicio": str(precios.index.min().date()),
        "fecha_fin": str(precios.index.max().date()),
        "numero_observaciones": int(len(precios)),
        "fuentes_utilizadas": fuentes_utilizadas,
        "pesos_utilizados": pesos_dict,
        "metricas_portafolio": {
            "rentabilidad_anualizada": round(rentabilidad_anual, 6),
            "volatilidad_anualizada": round(volatilidad_anual, 6),
            "max_drawdown": round(max_drawdown, 6),
            "var_95_historico_diario": round(var_95, 6),
            "cvar_95_historico_diario": round(cvar_95, 6),
        },
        "metricas_individuales": {
            "rentabilidad_anualizada": rentabilidades_individuales,
            "volatilidad_anualizada": volatilidades_individuales,
        },
        "matriz_correlacion": correlacion,
        "explicacion_lenguaje_natural": explicacion,
    }


@app.get("/")
def inicio():
    return {
        "mensaje": "CLARUM Invest API está funcionando correctamente.",
        "estado": "ok",
        "version": "0.2.13",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "servicio": "CLARUM Invest API",
        "version": "0.2.13",
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/diagnostico-activo/{ticker}")
def diagnostico_activo(ticker: str):
    ticker = ticker.upper()
    periodo = "1y"

    resultado = {
        "ticker": ticker,
        "periodo": periodo,
        "fuentes": {},
    }

    pruebas = [
        ("yahoo_chart", descargar_con_yahoo_chart),
        ("yfinance", descargar_con_yfinance),
        ("stooq", descargar_con_stooq),
    ]

    for nombre, funcion in pruebas:
        try:
            serie = funcion(ticker, periodo)
            resultado["fuentes"][nombre] = {
                "status": "ok",
                "observaciones": int(len(serie)),
                "fecha_inicio": str(serie.index.min().date()),
                "fecha_fin": str(serie.index.max().date()),
                "ultimo_precio": round(float(serie.iloc[-1]), 4),
            }
        except Exception as error:
            resultado["fuentes"][nombre] = {
                "status": "error",
                "detalle": str(error),
            }

    return resultado


@app.post("/perfil-inversionista")
def perfil_inversionista(payload: Dict[str, Any]):
    conocimiento = str(payload.get("conocimiento", "basico")).lower()
    tolerancia = str(payload.get("tolerancia_riesgo", "media")).lower()
    horizonte = int(payload.get("horizonte_anios", 5))

    if conocimiento in ["bajo", "basico", "básico"] or tolerancia in ["baja"]:
        perfil = "conservador"
    elif tolerancia in ["alta"] and horizonte >= 5:
        perfil = "dinamico"
    else:
        perfil = "moderado"

    return {
        "status": "ok",
        "modulo": "perfil-inversionista",
        "perfil_estimado": perfil,
        "nivel_conocimiento": conocimiento,
        "mensaje": (
            "Este perfil es una clasificación académica preliminar. No constituye "
            "asesoría financiera personalizada ni recomendación de inversión."
        ),
        "datos_recibidos": payload,
    }


@app.post("/analizar-portafolio")
def analizar_portafolio(payload: Dict[str, Any]):
    resultado = calcular_metricas_financieras(payload)

    return {
        "status": "ok",
        "modulo": "analizar-portafolio",
        "resultado": resultado,
        "advertencia": (
            "Los cálculos se basan en datos históricos y tienen finalidad académica. "
            "No constituyen asesoría financiera ni garantizan resultados futuros."
        ),
    }


@app.post("/calcular-riesgo")
def calcular_riesgo(payload: Dict[str, Any]):
    resultado = calcular_metricas_financieras(payload)
    metricas = resultado["metricas_portafolio"]

    return {
        "status": "ok",
        "modulo": "calcular-riesgo",
        "resultado": {
            "activos": resultado["activos"],
            "periodo": resultado["periodo"],
            "fuentes_utilizadas": resultado["fuentes_utilizadas"],
            "volatilidad_anualizada": metricas["volatilidad_anualizada"],
            "max_drawdown": metricas["max_drawdown"],
            "var_95_historico_diario": metricas["var_95_historico_diario"],
            "cvar_95_historico_diario": metricas["cvar_95_historico_diario"],
            "explicacion_lenguaje_natural": resultado["explicacion_lenguaje_natural"],
        },
        "advertencia": (
            "El análisis de riesgo se calcula con datos históricos. No representa "
            "una predicción exacta del comportamiento futuro del mercado."
        ),
    }

#==========================================
# MATRIZ RIESGO CONOCIMIENTO
#==========================================

@app.get("/api/profile/risk-knowledge-matrix")
def obtener_matriz_riesgo_conocimiento():
    return {
        "status": "ok",
        "modulo": "risk-knowledge-matrix",
        "risk_profiles": [
            "Conservador",
            "Moderado",
            "Dinámico",
            "Agresivo",
        ],
        "knowledge_levels": [
            "Básico",
            "Intermedio",
            "Avanzado",
        ],
        "matrix": RISK_KNOWLEDGE_MATRIX,
        "descripcion": (
            "La matriz riesgo-conocimiento cruza el perfil de riesgo del usuario "
            "con su nivel de conocimiento financiero. Su finalidad es definir "
            "restricciones académicas prudenciales para la selección de activos "
            "y la simulación de carteras."
        ),
        "advertencia": (
            "Esta matriz tiene finalidad académica y no constituye asesoría "
            "financiera personalizada."
        ),
    }
#===========================================

# ============================================================
# MÓDULO CATÁLOGO MAESTRO DE ACTIVOS Y FILTRO POR PERFIL
# ============================================================

class AllowedAssetsRequest(BaseModel):
    resultado: Optional[Dict[str, Any]] = None
    restricciones: Optional[Dict[str, Any]] = None
    perfil_riesgo: Optional[str] = None
    nivel_conocimiento: Optional[str] = None
    search: Optional[str] = None


ASSET_CATALOG = [
    {
        "ticker": "SHY",
        "name": "iShares 1-3 Year Treasury Bond ETF",
        "asset_class": "ETF renta fija corto plazo",
        "asset_type": "Renta fija ETF",
        "risk_level": "Bajo",
        "complexity": "Baja",
        "currency": "USD",
        "region": "Estados Unidos",
        "data_source": "yfinance",
        "description": "ETF que representa bonos del Tesoro estadounidense de corto plazo.",
        "academic_note": "Puede servir para explicar renta fija de baja duración dentro de una simulación académica.",
    },
    {
        "ticker": "VGSH",
        "name": "Vanguard Short-Term Treasury ETF",
        "asset_class": "ETF renta fija corto plazo",
        "asset_type": "Renta fija ETF",
        "risk_level": "Bajo",
        "complexity": "Baja",
        "currency": "USD",
        "region": "Estados Unidos",
        "data_source": "yfinance",
        "description": "ETF de bonos del Tesoro de corto plazo.",
        "academic_note": "Se utiliza como representación simple de renta fija de corto plazo.",
    },
    {
        "ticker": "AGG",
        "name": "iShares Core U.S. Aggregate Bond ETF",
        "asset_class": "ETF renta fija agregada",
        "asset_type": "Renta fija ETF",
        "risk_level": "Bajo",
        "complexity": "Baja",
        "currency": "USD",
        "region": "Estados Unidos",
        "data_source": "yfinance",
        "description": "ETF diversificado de bonos del mercado estadounidense.",
        "academic_note": "Permite explicar una exposición diversificada a renta fija.",
    },
    {
        "ticker": "BND",
        "name": "Vanguard Total Bond Market ETF",
        "asset_class": "ETF renta fija agregada",
        "asset_type": "Renta fija ETF",
        "risk_level": "Bajo",
        "complexity": "Baja",
        "currency": "USD",
        "region": "Estados Unidos",
        "data_source": "yfinance",
        "description": "ETF amplio de renta fija estadounidense.",
        "academic_note": "Representa una canasta diversificada de bonos para fines educativos.",
    },
    {
        "ticker": "IEF",
        "name": "iShares 7-10 Year Treasury Bond ETF",
        "asset_class": "ETF renta fija",
        "asset_type": "Renta fija ETF",
        "risk_level": "Medio",
        "complexity": "Media",
        "currency": "USD",
        "region": "Estados Unidos",
        "data_source": "yfinance",
        "description": "ETF de bonos del Tesoro estadounidense de mediano plazo.",
        "academic_note": "Permite explicar sensibilidad a tasas de interés en renta fija.",
    },
    {
        "ticker": "LQD",
        "name": "iShares iBoxx Investment Grade Corporate Bond ETF",
        "asset_class": "ETF renta fija",
        "asset_type": "Renta fija ETF",
        "risk_level": "Medio",
        "complexity": "Media",
        "currency": "USD",
        "region": "Estados Unidos",
        "data_source": "yfinance",
        "description": "ETF de bonos corporativos con grado de inversión.",
        "academic_note": "Útil para explicar que no toda renta fija tiene el mismo nivel de riesgo.",
    },
    {
        "ticker": "VOO",
        "name": "Vanguard S&P 500 ETF",
        "asset_class": "ETF renta variable diversificada",
        "asset_type": "Renta variable ETF",
        "risk_level": "Medio",
        "complexity": "Media",
        "currency": "USD",
        "region": "Estados Unidos",
        "data_source": "yfinance",
        "description": "ETF que replica el comportamiento de empresas del índice S&P 500.",
        "academic_note": "Permite analizar renta variable diversificada sin escoger una sola acción.",
    },
    {
        "ticker": "SPY",
        "name": "SPDR S&P 500 ETF Trust",
        "asset_class": "ETF renta variable diversificada",
        "asset_type": "Renta variable ETF",
        "risk_level": "Medio",
        "complexity": "Media",
        "currency": "USD",
        "region": "Estados Unidos",
        "data_source": "yfinance",
        "description": "ETF ampliamente utilizado para representar el índice S&P 500.",
        "academic_note": "Sirve como activo de referencia para comparar renta variable diversificada.",
    },
    {
        "ticker": "VTI",
        "name": "Vanguard Total Stock Market ETF",
        "asset_class": "ETF renta variable diversificada",
        "asset_type": "Renta variable ETF",
        "risk_level": "Medio",
        "complexity": "Media",
        "currency": "USD",
        "region": "Estados Unidos",
        "data_source": "yfinance",
        "description": "ETF que representa una exposición amplia al mercado accionario estadounidense.",
        "academic_note": "Permite explicar diversificación dentro de renta variable.",
    },
    {
        "ticker": "VT",
        "name": "Vanguard Total World Stock ETF",
        "asset_class": "ETF renta variable diversificada",
        "asset_type": "Renta variable ETF",
        "risk_level": "Medio",
        "complexity": "Media",
        "currency": "USD",
        "region": "Global",
        "data_source": "yfinance",
        "description": "ETF con exposición global a mercados accionarios.",
        "academic_note": "Útil para explicar diversificación geográfica.",
    },
    {
        "ticker": "JNJ",
        "name": "Johnson & Johnson",
        "asset_class": "Acciones individuales de baja volatilidad",
        "asset_type": "Acción individual",
        "risk_level": "Medio",
        "complexity": "Media",
        "currency": "USD",
        "region": "Estados Unidos",
        "data_source": "yfinance",
        "description": "Acción individual de una compañía del sector salud.",
        "academic_note": "Una acción individual tiene riesgo específico, aunque pertenezca a una empresa reconocida.",
    },
    {
        "ticker": "PG",
        "name": "Procter & Gamble",
        "asset_class": "Acciones individuales de baja volatilidad",
        "asset_type": "Acción individual",
        "risk_level": "Medio",
        "complexity": "Media",
        "currency": "USD",
        "region": "Estados Unidos",
        "data_source": "yfinance",
        "description": "Acción individual de una compañía de consumo defensivo.",
        "academic_note": "Permite explicar diferencia entre acción individual y ETF diversificado.",
    },
    {
        "ticker": "AAPL",
        "name": "Apple Inc.",
        "asset_class": "Acciones individuales de media volatilidad",
        "asset_type": "Acción individual",
        "risk_level": "Medio-alto",
        "complexity": "Media",
        "currency": "USD",
        "region": "Estados Unidos",
        "data_source": "yfinance",
        "description": "Acción individual de una compañía tecnológica.",
        "academic_note": "Debe analizarse con límites de concentración por tratarse de una acción individual.",
    },
    {
        "ticker": "MSFT",
        "name": "Microsoft Corporation",
        "asset_class": "Acciones individuales de media volatilidad",
        "asset_type": "Acción individual",
        "risk_level": "Medio-alto",
        "complexity": "Media",
        "currency": "USD",
        "region": "Estados Unidos",
        "data_source": "yfinance",
        "description": "Acción individual de una compañía tecnológica.",
        "academic_note": "Su análisis debe considerar volatilidad, concentración y riesgo específico.",
    },
    {
        "ticker": "TSLA",
        "name": "Tesla Inc.",
        "asset_class": "Acciones individuales de alta volatilidad",
        "asset_type": "Acción individual",
        "risk_level": "Alto",
        "complexity": "Alta",
        "currency": "USD",
        "region": "Estados Unidos",
        "data_source": "yfinance",
        "description": "Acción individual con variaciones históricas fuertes.",
        "academic_note": "Solo debe habilitarse para perfiles con mayor tolerancia y conocimiento, y con límite de concentración.",
    },
    {
        "ticker": "NVDA",
        "name": "NVIDIA Corporation",
        "asset_class": "Acciones individuales de alta volatilidad",
        "asset_type": "Acción individual",
        "risk_level": "Alto",
        "complexity": "Alta",
        "currency": "USD",
        "region": "Estados Unidos",
        "data_source": "yfinance",
        "description": "Acción tecnológica con alto crecimiento histórico y elevada variabilidad.",
        "academic_note": "Debe analizarse con especial cuidado por su posible volatilidad y concentración sectorial.",
    },
]


def extraer_restricciones_desde_request(request: AllowedAssetsRequest) -> Dict[str, Any]:
    if request.restricciones:
        return request.restricciones

    if request.resultado and isinstance(request.resultado, dict):
        restricciones = request.resultado.get("restricciones")
        if restricciones:
            return restricciones

        perfil_riesgo = request.resultado.get("perfil_riesgo")
        nivel_conocimiento = request.resultado.get("nivel_conocimiento")

        if perfil_riesgo and nivel_conocimiento:
            return obtener_restricciones_academicas(
                risk_profile=perfil_riesgo,
                knowledge_level=nivel_conocimiento,
            )

    if request.perfil_riesgo and request.nivel_conocimiento:
        return obtener_restricciones_academicas(
            risk_profile=request.perfil_riesgo,
            knowledge_level=request.nivel_conocimiento,
        )

    raise HTTPException(
        status_code=400,
        detail="No fue posible obtener restricciones. Envía resultado, restricciones o perfil_riesgo/nivel_conocimiento.",
    )


def clase_activo_permitida(asset_class: str, allowed_classes: List[str]) -> bool:
    asset_class_norm = asset_class.strip().lower()
    allowed_norm = [item.strip().lower() for item in allowed_classes]

    if asset_class_norm in allowed_norm:
        return True

    for allowed in allowed_norm:
        if allowed == "etf renta fija" and asset_class_norm.startswith("etf renta fija"):
            return True

        if allowed == "acciones individuales" and asset_class_norm.startswith("acciones individuales"):
            return True

        if allowed == "acciones individuales de baja o media volatilidad":
            if asset_class_norm in [
                "acciones individuales de baja volatilidad",
                "acciones individuales de media volatilidad",
            ]:
                return True

        if allowed == "acciones individuales de alta volatilidad con límite reducido":
            if asset_class_norm == "acciones individuales de alta volatilidad":
                return True

        if allowed == "etf renta variable diversificada":
            if asset_class_norm == "etf renta variable diversificada":
                return True

    return False


def aplicar_busqueda_activos(activos: List[Dict[str, Any]], search: Optional[str]) -> List[Dict[str, Any]]:
    if not search:
        return activos

    term = search.strip().lower()

    if not term:
        return activos

    filtrados = []

    for activo in activos:
        texto = " ".join([
            str(activo.get("ticker", "")),
            str(activo.get("name", "")),
            str(activo.get("asset_class", "")),
            str(activo.get("asset_type", "")),
            str(activo.get("risk_level", "")),
            str(activo.get("region", "")),
        ]).lower()

        if term in texto:
            filtrados.append(activo)

    return filtrados


def enriquecer_activo_con_restricciones(
    activo: Dict[str, Any],
    restricciones: Dict[str, Any],
) -> Dict[str, Any]:

    max_weight = restricciones.get("max_weight_per_asset", 0)

    peso_maximo_aplicable = max_weight

    if activo["asset_class"] == "Acciones individuales de alta volatilidad":
        peso_maximo_aplicable = min(max_weight, 10)

    activo_enriquecido = {
        **activo,
        "max_weight_allowed": peso_maximo_aplicable,
        "selection_allowed": True,
        "restriction_note": (
            f"Peso máximo sugerido para este activo dentro del modelo: {peso_maximo_aplicable}%."
        ),
    }

    if activo["asset_class"] == "Acciones individuales de alta volatilidad":
        activo_enriquecido["restriction_note"] += (
            " Al ser un activo de alta volatilidad, el sistema aplica un límite reducido."
        )

    return activo_enriquecido


@app.post("/api/assets/allowed")
def obtener_activos_permitidos(request: AllowedAssetsRequest):
    restricciones = extraer_restricciones_desde_request(request)

    allowed_classes = restricciones.get("allowed_asset_classes", [])

    activos_permitidos = []
    activos_bloqueados = []

    for activo in ASSET_CATALOG:
        permitido = clase_activo_permitida(
            asset_class=activo["asset_class"],
            allowed_classes=allowed_classes,
        )

        if permitido:
            activos_permitidos.append(
                enriquecer_activo_con_restricciones(activo, restricciones)
            )
        else:
            activos_bloqueados.append({
                **activo,
                "selection_allowed": False,
                "block_reason": (
                    "Este activo no está habilitado para la categoría actual de la matriz riesgo-conocimiento."
                ),
            })

    activos_permitidos = aplicar_busqueda_activos(
        activos=activos_permitidos,
        search=request.search,
    )

    perfil_riesgo = None
    nivel_conocimiento = None
    matriz_categoria = restricciones.get("categoria_matriz")

    if request.resultado:
        perfil_riesgo = request.resultado.get("perfil_riesgo")
        nivel_conocimiento = request.resultado.get("nivel_conocimiento")
        matriz_categoria = request.resultado.get("matriz_riesgo_conocimiento", matriz_categoria)

    return {
        "status": "ok",
        "modulo": "assets-allowed",
        "perfil_riesgo": perfil_riesgo,
        "nivel_conocimiento": nivel_conocimiento,
        "matriz_riesgo_conocimiento": matriz_categoria,
        "restricciones": restricciones,
        "activos_permitidos": activos_permitidos,
        "activos_bloqueados": activos_bloqueados,
        "total_permitidos": len(activos_permitidos),
        "total_bloqueados": len(activos_bloqueados),
        "resumen": (
            "Los activos mostrados fueron filtrados por la matriz riesgo-conocimiento "
            "y por las restricciones académicas del usuario. Este catálogo no representa "
            "una recomendación personalizada de inversión."
        ),
    }


@app.get("/api/assets/catalog")
def obtener_catalogo_activos():
    return {
        "status": "ok",
        "modulo": "asset-catalog",
        "total_activos": len(ASSET_CATALOG),
        "catalogo": ASSET_CATALOG,
        "advertencia": (
            "Este catálogo tiene finalidad académica. Los activos incluidos son referencias "
            "para simulación y no constituyen recomendación de inversión."
        ),
    }

#==================================================

# ============================================================
# MÓDULO MÉTRICAS HISTÓRICAS INDIVIDUALES POR ACTIVO
# Endpoint: POST /api/assets/risk-metrics
# ============================================================

class AssetRiskMetricsRequest(BaseModel):
    tickers: List[str]
    lookback_observations: int = 250
    timeframe: str = "1d"
    confidence_level: float = 0.95


ASSET_RISK_METRICS_CACHE: Dict[str, Dict[str, Any]] = {}

ASSET_RISK_CACHE_TTL_OK_SECONDS = 12 * 60 * 60
ASSET_RISK_CACHE_TTL_ERROR_SECONDS = 5 * 60


def limpiar_tickers_metricas(tickers: List[str]) -> List[str]:
    tickers_limpios = []
    vistos = set()

    for ticker in tickers:
        ticker_limpio = str(ticker).strip().upper()

        if not ticker_limpio:
            continue

        if ticker_limpio not in vistos:
            tickers_limpios.append(ticker_limpio)
            vistos.add(ticker_limpio)

    return tickers_limpios


def generar_cache_key_metricas(
    ticker: str,
    lookback_observations: int,
    timeframe: str,
    confidence_level: float,
) -> str:
    return (
        f"{ticker}|"
        f"{lookback_observations}|"
        f"{timeframe}|"
        f"{confidence_level}"
    )


def obtener_metricas_desde_cache(cache_key: str) -> Optional[Dict[str, Any]]:
    item = ASSET_RISK_METRICS_CACHE.get(cache_key)

    if not item:
        return None

    created_at = float(item.get("created_at", 0))
    ttl_seconds = float(item.get("ttl_seconds", ASSET_RISK_CACHE_TTL_OK_SECONDS))

    if (datetime.now(timezone.utc).timestamp() - created_at) > ttl_seconds:
        ASSET_RISK_METRICS_CACHE.pop(cache_key, None)
        return None

    return item.get("data")


def guardar_metricas_en_cache(
    cache_key: str,
    data: Dict[str, Any],
    ttl_seconds: int,
) -> None:
    ASSET_RISK_METRICS_CACHE[cache_key] = {
        "created_at": datetime.now(timezone.utc).timestamp(),
        "ttl_seconds": ttl_seconds,
        "data": data,
    }


def preparar_serie_para_metricas(
    ticker: str,
    lookback_observations: int,
) -> Dict[str, Any]:
    """
    Usa la arquitectura actual del backend:
    descargar_precio_activo intenta yahoo_chart, yfinance y stooq.

    Para obtener 250 retornos se necesitan al menos 251 precios.
    Por prudencia se descarga un periodo amplio.
    """

    if lookback_observations <= 250:
        periodo_descarga = "2y"
    elif lookback_observations <= 500:
        periodo_descarga = "3y"
    else:
        periodo_descarga = "7y"

    resultado_descarga = descargar_precio_activo(
        ticker=ticker,
        periodo=periodo_descarga,
    )

    serie = resultado_descarga["serie"]
    fuente = resultado_descarga["fuente"]

    serie = pd.to_numeric(serie, errors="coerce").dropna()
    serie = serie.sort_index()

    # Evita que se use cualquier fecha futura si alguna fuente la entrega por error.
    hoy = pd.Timestamp(datetime.now().date())
    serie = serie[serie.index <= hoy]

    if serie.empty or len(serie) < 61:
        raise ValueError("No hay suficientes precios históricos disponibles.")

    return {
        "ticker": ticker,
        "fuente": fuente,
        "serie": serie,
    }


def calcular_metricas_individuales_activo(
    ticker: str,
    lookback_observations: int,
    confidence_level: float,
) -> Dict[str, Any]:
    resultado_serie = preparar_serie_para_metricas(
        ticker=ticker,
        lookback_observations=lookback_observations,
    )

    serie = resultado_serie["serie"]
    fuente = resultado_serie["fuente"]

    # Para N retornos se necesitan N + 1 precios.
    precios_ventana = serie.tail(lookback_observations + 1)

    retornos = precios_ventana.pct_change().dropna()
    retornos = retornos.tail(lookback_observations)

    if retornos.empty or len(retornos) < 60:
        return {
            "status": "unavailable",
            "reason": "No hay suficientes observaciones históricas para calcular métricas confiables.",
        }

    # Alinea precios usados para drawdown con el rango de retornos.
    fecha_inicio_retornos = retornos.index.min()
    fecha_fin_retornos = retornos.index.max()

    precios_drawdown = precios_ventana[
        (precios_ventana.index >= fecha_inicio_retornos) &
        (precios_ventana.index <= fecha_fin_retornos)
    ]

    if precios_drawdown.empty or len(precios_drawdown) < 2:
        precios_drawdown = precios_ventana.tail(len(retornos) + 1)

    volatilidad_anualizada = float(retornos.std(ddof=1) * np.sqrt(252))

    tail_probability = 1.0 - confidence_level

    var_daily = float(np.percentile(retornos, tail_probability * 100))

    retornos_cola = retornos[retornos <= var_daily]

    if len(retornos_cola) > 0:
        cvar_daily = float(retornos_cola.mean())
    else:
        cvar_daily = var_daily

    maximo_acumulado = precios_drawdown.cummax()
    drawdown = precios_drawdown / maximo_acumulado - 1
    max_drawdown = float(drawdown.min())

    return {
        "status": "ok",
        "observations": int(len(retornos)),
        "start_date": str(retornos.index.min().date()),
        "end_date": str(retornos.index.max().date()),
        "annualized_volatility": round(volatilidad_anualizada, 6),

        # Se devuelven negativos porque representan pérdida.
        "var_95_daily": round(var_daily, 6),
        "cvar_95_daily": round(cvar_daily, 6),
        "max_drawdown_250d": round(max_drawdown, 6),

        "last_price": round(float(serie.iloc[-1]), 6),
        "price_source": fuente,
        "data_source": "backend_historical_prices",
    }


@app.post("/api/assets/risk-metrics")
@app.post("/api/assets/metrics250")
def obtener_metricas_riesgo_activos(request: AssetRiskMetricsRequest):
    tickers = limpiar_tickers_metricas(request.tickers or [])

    if not tickers:
        raise HTTPException(
            status_code=422,
            detail={
                "status": "error",
                "message": "Debe enviar al menos un ticker.",
            },
        )

    if len(tickers) > 50:
        raise HTTPException(
            status_code=422,
            detail={
                "status": "error",
                "message": "La solicitud supera el máximo de 50 tickers.",
            },
        )

    lookback_observations = int(request.lookback_observations or 250)

    if lookback_observations < 60 or lookback_observations > 1000:
        raise HTTPException(
            status_code=422,
            detail={
                "status": "error",
                "message": "lookback_observations debe estar entre 60 y 1000.",
            },
        )

    timeframe = str(request.timeframe or "1d").strip().lower()

    if timeframe != "1d":
        raise HTTPException(
            status_code=422,
            detail={
                "status": "error",
                "message": "Por ahora solo se soporta timeframe diario 1d para métricas individuales.",
            },
        )

    confidence_level = float(request.confidence_level or 0.95)

    if confidence_level < 0.90 or confidence_level > 0.99:
        raise HTTPException(
            status_code=422,
            detail={
                "status": "error",
                "message": "confidence_level debe estar entre 0.90 y 0.99.",
            },
        )

    inicio_calculo = datetime.now(timezone.utc)

    print(
        "[asset-risk-metrics] "
        f"tickers={tickers} "
        f"lookback={lookback_observations} "
        f"timeframe={timeframe} "
        f"confidence={confidence_level}"
    )

    metrics = {}
    unavailable_tickers = []

    for ticker in tickers:
        cache_key = generar_cache_key_metricas(
            ticker=ticker,
            lookback_observations=lookback_observations,
            timeframe=timeframe,
            confidence_level=confidence_level,
        )

        cached = obtener_metricas_desde_cache(cache_key)

        if cached:
            metrics[ticker] = cached

            if cached.get("status") != "ok":
                unavailable_tickers.append(ticker)

            continue

        try:
            resultado = calcular_metricas_individuales_activo(
                ticker=ticker,
                lookback_observations=lookback_observations,
                confidence_level=confidence_level,
            )

            metrics[ticker] = resultado

            if resultado.get("status") == "ok":
                guardar_metricas_en_cache(
                    cache_key=cache_key,
                    data=resultado,
                    ttl_seconds=ASSET_RISK_CACHE_TTL_OK_SECONDS,
                )
            else:
                unavailable_tickers.append(ticker)
                guardar_metricas_en_cache(
                    cache_key=cache_key,
                    data=resultado,
                    ttl_seconds=ASSET_RISK_CACHE_TTL_ERROR_SECONDS,
                )

        except Exception as error:
            print(f"[asset-risk-metrics] error en {ticker}: {str(error)}")

            resultado_error = {
                "status": "unavailable",
                "reason": "No se pudieron obtener datos históricos para este activo.",
                "technical_detail": str(error),
            }

            metrics[ticker] = resultado_error
            unavailable_tickers.append(ticker)

            guardar_metricas_en_cache(
                cache_key=cache_key,
                data=resultado_error,
                ttl_seconds=ASSET_RISK_CACHE_TTL_ERROR_SECONDS,
            )

    fin_calculo = datetime.now(timezone.utc)
    elapsed_seconds = round((fin_calculo - inicio_calculo).total_seconds(), 3)

    respuesta = {
        "status": "ok",
        "modulo": "asset_risk_metrics",
        "lookback_observations": lookback_observations,
        "timeframe": timeframe,
        "confidence_level": confidence_level,
        "data_source": "backend_historical_prices",
        "generated_at": fin_calculo.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "metrics": metrics,
        "unavailable_tickers": unavailable_tickers,
        "advertencia": (
            "Estas métricas se calculan con datos históricos y tienen finalidad académica. "
            "No constituyen recomendación de inversión ni garantizan resultados futuros."
        ),
    }

    return limpiar_valores_json(respuesta)

# ============================================================
# MÓDULO VALIDACIÓN DE PORTAFOLIO SEGÚN RESTRICCIONES
# ============================================================

class PortfolioValidationRequest(BaseModel):
    resultado: Optional[Dict[str, Any]] = None
    selected_assets: List[Dict[str, Any]]
    weights: Optional[Dict[str, float]] = None


def obtener_clase_general_activo(asset_class: str) -> str:
    clase = str(asset_class).lower()

    if "renta fija" in clase:
        return "renta_fija"

    if "renta variable" in clase or "acciones individuales" in clase:
        return "renta_variable"

    return "otro"


def validar_activo_con_catalogo(ticker: str) -> Optional[Dict[str, Any]]:
    ticker_norm = str(ticker).strip().upper()

    for activo in ASSET_CATALOG:
        if activo["ticker"].upper() == ticker_norm:
            return activo

    return None


def normalizar_pesos_portafolio(
    selected_assets: List[Dict[str, Any]],
    weights: Optional[Dict[str, float]],
) -> Dict[str, float]:

    tickers = [str(asset.get("ticker", "")).upper() for asset in selected_assets]

    if not weights:
        peso_igual = round(100 / len(tickers), 6)
        return {ticker: peso_igual for ticker in tickers}

    pesos_limpios = {}

    for ticker in tickers:
        peso = weights.get(ticker)

        if peso is None:
            peso = weights.get(ticker.lower())

        if peso is None:
            peso = 0

        try:
            peso_float = float(peso)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail=f"El peso del activo {ticker} no es un número válido.",
            )

        pesos_limpios[ticker] = peso_float

    return pesos_limpios


@app.post("/api/portfolio/validate-selection")
def validar_portafolio_academico(request: PortfolioValidationRequest):
    if not request.selected_assets:
        raise HTTPException(
            status_code=400,
            detail="Debes enviar al menos un activo seleccionado.",
        )

    if len(request.selected_assets) > 10:
        raise HTTPException(
            status_code=400,
            detail="El MVP académico permite máximo 10 activos por portafolio.",
        )

    if not request.resultado:
        raise HTTPException(
            status_code=400,
            detail="Debes enviar el resultado del perfil académico para validar restricciones.",
        )

    perfil_riesgo = request.resultado.get("perfil_riesgo")
    nivel_conocimiento = request.resultado.get("nivel_conocimiento")

    restricciones = request.resultado.get("restricciones")

    if not restricciones and perfil_riesgo and nivel_conocimiento:
        restricciones = obtener_restricciones_academicas(
            risk_profile=perfil_riesgo,
            knowledge_level=nivel_conocimiento,
        )

    if not restricciones:
        raise HTTPException(
            status_code=400,
            detail="No fue posible obtener restricciones del perfil académico.",
        )

    allowed_classes = restricciones.get("allowed_asset_classes", [])
    max_equity = float(restricciones.get("max_equity_percentage", 0))
    min_fixed_income = float(restricciones.get("min_fixed_income_percentage", 0))
    max_weight_per_asset = float(restricciones.get("max_weight_per_asset", 0))
    short_allowed = bool(restricciones.get("short_positions_allowed", False))
    leverage_allowed = bool(restricciones.get("leverage_allowed", False))

    errors = []
    warnings = []

    pesos = normalizar_pesos_portafolio(
        selected_assets=request.selected_assets,
        weights=request.weights,
    )

    suma_pesos = round(sum(pesos.values()), 6)

    if not leverage_allowed and suma_pesos > 100.0001:
        errors.append(
            f"La suma de los pesos es {suma_pesos:.2f}%, superior al 100%. "
            "El perfil no permite apalancamiento."
        )

    if abs(suma_pesos - 100) > 0.01:
        errors.append(
            f"La suma de los pesos debe ser 100%. Actualmente es {suma_pesos:.2f}%."
        )

    renta_variable_total = 0.0
    renta_fija_total = 0.0
    otros_total = 0.0

    activos_validados = []

    for asset in request.selected_assets:
        ticker = str(asset.get("ticker", "")).upper().strip()

        if not ticker:
            errors.append("Existe un activo seleccionado sin ticker.")
            continue

        catalog_asset = validar_activo_con_catalogo(ticker)

        if not catalog_asset:
            errors.append(
                f"El activo {ticker} no existe en el catálogo maestro académico."
            )
            continue

        asset_class = catalog_asset["asset_class"]

        if not clase_activo_permitida(asset_class, allowed_classes):
            errors.append(
                f"El activo {ticker} ({asset_class}) no está permitido para el perfil actual."
            )

        peso = pesos.get(ticker, 0)

        if peso < 0 and not short_allowed:
            errors.append(
                f"El activo {ticker} tiene peso negativo. Las posiciones cortas no están permitidas."
            )

        max_weight_asset = float(asset.get("max_weight_allowed", max_weight_per_asset))

        if asset_class == "Acciones individuales de alta volatilidad":
            max_weight_asset = min(max_weight_asset, 10)

        if peso > max_weight_asset + 0.0001:
            errors.append(
                f"El activo {ticker} tiene un peso de {peso:.2f}%, "
                f"pero su máximo permitido es {max_weight_asset:.2f}%."
            )

        clase_general = obtener_clase_general_activo(asset_class)

        if clase_general == "renta_variable":
            renta_variable_total += peso
        elif clase_general == "renta_fija":
            renta_fija_total += peso
        else:
            otros_total += peso

        activos_validados.append({
            **catalog_asset,
            "assigned_weight": round(peso, 4),
            "max_weight_allowed": round(max_weight_asset, 4),
            "general_class": clase_general,
        })

    if renta_variable_total > max_equity + 0.0001:
        errors.append(
            f"La exposición total a renta variable es {renta_variable_total:.2f}%, "
            f"pero el máximo permitido para este perfil es {max_equity:.2f}%."
        )

    if renta_fija_total < min_fixed_income - 0.0001:
        warnings.append(
            f"La exposición a renta fija es {renta_fija_total:.2f}%, "
            f"mientras que la mínima sugerida para este perfil es {min_fixed_income:.2f}%."
        )

    if len(request.selected_assets) < 2:
        warnings.append(
            "El portafolio tiene menos de dos activos. Esto limita la diversificación académica."
        )

    is_valid = len(errors) == 0

    if is_valid:
        mensaje = (
            "El portafolio propuesto cumple las restricciones principales del perfil académico. "
            "Puede avanzar al simulador para calcular métricas históricas de rentabilidad y riesgo. "
            "Este resultado no constituye una recomendación personalizada de inversión."
        )
    else:
        mensaje = (
            "El portafolio propuesto no cumple una o más restricciones académicas del perfil. "
            "Ajusta los pesos o cambia los activos seleccionados antes de continuar al simulador."
        )

    return {
        "status": "ok",
        "modulo": "portfolio-validate-selection",
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "normalized_weights": pesos,
        "portfolio_summary": {
            "numero_activos": len(request.selected_assets),
            "suma_pesos": round(suma_pesos, 4),
            "renta_variable_total": round(renta_variable_total, 4),
            "renta_fija_total": round(renta_fija_total, 4),
            "otros_total": round(otros_total, 4),
            "max_equity_allowed": max_equity,
            "min_fixed_income_suggested": min_fixed_income,
            "max_weight_per_asset": max_weight_per_asset,
            "fixed_income_status": "ok" if renta_fija_total >= min_fixed_income else "warning",
            "equity_status": "ok" if renta_variable_total <= max_equity else "error",
            "fixed_income_gap": round(renta_fija_total - min_fixed_income, 4),
            "equity_gap": round(max_equity - renta_variable_total, 4),
        },
        "validated_assets": activos_validados,
        "mensaje_academico": mensaje,
        "advertencia": (
            "La validación tiene finalidad académica y simulativa. "
            "No constituye asesoría financiera personalizada."
        ),
    }

#==========================================================================

# ============================================================
# MÓDULO SIMULADOR ACADÉMICO DE PORTAFOLIO
# ============================================================

class PortfolioSimulationRequest(BaseModel):
    resultado: Optional[Dict[str, Any]] = None
    selected_assets: List[Dict[str, Any]]
    weights: Dict[str, float]
    period: Optional[str] = "2y"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    timeframe: Optional[str] = "diario"
    risk_free_rate: Optional[float] = 0.0
    risk_free_rate_annual: Optional[float] = None


def obtener_factor_anualizacion(timeframe: str) -> int:
    tf = str(timeframe).strip().lower()

    factores = {
        "diario": 252,
        "semanal": 52,
        "quincenal": 24,
        "mensual": 12,
        "trimestral": 4,
        "semestral": 2,
        "anual": 1,
    }

    return factores.get(tf, 252)


def obtener_regla_resample(timeframe: str):
    tf = str(timeframe).strip().lower()

    reglas = {
        "diario": None,
        "semanal": "W-FRI",
        "quincenal": "15D",
        "mensual": "ME",
        "trimestral": "QE",
        "semestral": "6ME",
        "anual": "YE",
    }

    return reglas.get(tf, None)


def descargar_precios_yfinance_simulacion(
    tickers: List[str],
    period: str = "2y",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:

    kwargs = {
        "tickers": tickers,
        "interval": "1d",
        "auto_adjust": True,
        "progress": False,
        "threads": False,
    }

    if start_date and end_date:
        fecha_inicio = pd.to_datetime(start_date)
        fecha_fin = pd.to_datetime(end_date) + timedelta(days=1)

        kwargs["start"] = fecha_inicio.strftime("%Y-%m-%d")
        kwargs["end"] = fecha_fin.strftime("%Y-%m-%d")
    else:
        kwargs["period"] = period or "2y"

    data = yf.download(**kwargs)

    if data.empty:
        raise HTTPException(
            status_code=400,
            detail="yfinance no devolvió datos históricos para los activos seleccionados.",
        )

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            precios = data["Close"]
        elif "Adj Close" in data.columns.get_level_values(0):
            precios = data["Adj Close"]
        else:
            raise HTTPException(
                status_code=400,
                detail="No se encontró columna de cierre en los datos descargados.",
            )
    else:
        if "Close" in data.columns:
            precios = data[["Close"]]
            precios.columns = tickers
        elif "Adj Close" in data.columns:
            precios = data[["Adj Close"]]
            precios.columns = tickers
        else:
            raise HTTPException(
                status_code=400,
                detail="No se encontró columna Close o Adj Close en los datos descargados.",
            )

    precios = precios.copy()
    precios = precios.apply(pd.to_numeric, errors="coerce")
    precios = precios.dropna(how="all")
    precios = precios.ffill().dropna()

    columnas_disponibles = [col for col in precios.columns if str(col).upper() in tickers]
    precios = precios[columnas_disponibles]

    if precios.empty or len(precios) < 30:
        raise HTTPException(
            status_code=400,
            detail="No hay suficientes datos históricos comunes para ejecutar la simulación.",
        )

    precios.columns = [str(col).upper() for col in precios.columns]

    return precios


def aplicar_marco_temporal(precios: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    regla = obtener_regla_resample(timeframe)
    tf = str(timeframe).strip().lower()

    if regla is None:
        return precios

    try:
        precios_resampleados = precios.resample(regla).last().dropna()
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No fue posible convertir los precios al marco temporal '{timeframe}'. "
                f"Detalle técnico: {str(error)}"
            ),
        )

    min_observaciones = {
        "mensual": 6,
        "trimestral": 4,
        "semestral": 3,
        "anual": 3,
    }.get(tf, 3)

    if precios_resampleados.empty or len(precios_resampleados) < min_observaciones:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No hay suficientes observaciones para el marco temporal '{timeframe}'. "
                "Selecciona un periodo más amplio, por ejemplo 2 años, 5 años o máximo disponible."
            ),
        )

    return precios_resampleados


def generar_explicacion_simulacion(
    rentabilidad_anual: float,
    volatilidad_anual: float,
    sharpe_ratio: float,
    max_drawdown: float,
    var_95: float,
    cvar_95: float,
    timeframe: str,
    activos: List[str],
) -> str:

    rentabilidad_pct = rentabilidad_anual * 100
    volatilidad_pct = volatilidad_anual * 100
    drawdown_pct = max_drawdown * 100
    var_pct = var_95 * 100
    cvar_pct = cvar_95 * 100

    if volatilidad_anual < 0.10:
        nivel_riesgo = "bajo"
    elif volatilidad_anual < 0.20:
        nivel_riesgo = "medio"
    else:
        nivel_riesgo = "alto"

    return (
        f"La simulación histórica del portafolio compuesto por {', '.join(activos)} "
        f"muestra una rentabilidad anualizada aproximada de {rentabilidad_pct:.2f}% "
        f"y una volatilidad anualizada de {volatilidad_pct:.2f}%, usando un marco temporal "
        f"{timeframe}. En términos sencillos, la rentabilidad indica cuánto habría crecido "
        f"el portafolio en promedio anual según los datos históricos, mientras que la volatilidad "
        f"muestra qué tan fuertes fueron sus variaciones. Con base en la volatilidad observada, "
        f"el riesgo histórico se clasifica como {nivel_riesgo}. El máximo drawdown fue de "
        f"{drawdown_pct:.2f}%, lo que representa la mayor caída histórica desde un máximo hasta "
        f"un mínimo dentro del periodo analizado. El VaR histórico al 95% fue de {var_pct:.2f}% "
        f"y el CVaR fue de {cvar_pct:.2f}%, indicadores que permiten observar pérdidas históricas "
        f"en escenarios desfavorables. El ratio de Sharpe fue de {sharpe_ratio:.2f}, lo que permite "
        f"relacionar la rentabilidad obtenida con el riesgo asumido. Estos resultados no predicen "
        f"el futuro y tienen finalidad exclusivamente académica."
    )


@app.post("/api/portfolio/simulate")
def simular_portafolio_academico(request: PortfolioSimulationRequest):

    validacion = validar_portafolio_academico(
        PortfolioValidationRequest(
            resultado=request.resultado,
            selected_assets=request.selected_assets,
            weights=request.weights,
        )
    )

    if not validacion.get("is_valid"):
        return {
            "status": "error",
            "modulo": "portfolio-simulate",
            "validation": validacion,
            "mensaje": (
                "El portafolio no puede simularse porque no cumple las restricciones "
                "académicas del perfil."
            ),
        }

    tickers = [
        str(asset.get("ticker", "")).upper().strip()
        for asset in request.selected_assets
        if str(asset.get("ticker", "")).strip()
    ]

    if not tickers:
        raise HTTPException(
            status_code=400,
            detail="No se encontraron tickers válidos para la simulación.",
        )

    pesos_pct = {
        str(ticker).upper(): float(peso)
        for ticker, peso in request.weights.items()
    }

    pesos = np.array([
        pesos_pct.get(ticker, 0) / 100
        for ticker in tickers
    ])

    if abs(pesos.sum() - 1.0) > 0.001:
        raise HTTPException(
            status_code=400,
            detail="Los pesos deben sumar 100% para ejecutar la simulación.",
        )

    precios = descargar_precios_yfinance_simulacion(
        tickers=tickers,
        period=request.period or "2y",
        start_date=request.start_date,
        end_date=request.end_date,
    )

    precios = precios[tickers].dropna()
    precios = aplicar_marco_temporal(precios, request.timeframe or "diario")

    retornos = precios.pct_change().dropna()

    if retornos.empty:
        raise HTTPException(
            status_code=400,
            detail="No fue posible calcular retornos para la simulación.",
        )

    factor = obtener_factor_anualizacion(request.timeframe or "diario")
    retornos_portafolio = retornos.dot(pesos)

    rentabilidad_anual = float(retornos_portafolio.mean() * factor)
    volatilidad_anual = float(retornos_portafolio.std() * np.sqrt(factor))

    risk_free_rate = float(
        request.risk_free_rate_annual
        if request.risk_free_rate_annual is not None
        else (request.risk_free_rate or 0.0)
    )
    
    sharpe_ratio = (
        (rentabilidad_anual - risk_free_rate) / volatilidad_anual
        if volatilidad_anual > 0
        else 0.0
    )

    acumulado = (1 + retornos_portafolio).cumprod()
    maximo_acumulado = acumulado.cummax()
    drawdown = acumulado / maximo_acumulado - 1
    max_drawdown = float(drawdown.min())

    percentil_5 = float(np.percentile(retornos_portafolio, 5))
    var_95 = max(0.0, -percentil_5)

    retornos_cola = retornos_portafolio[retornos_portafolio <= percentil_5]
    cvar_95 = max(0.0, -float(retornos_cola.mean())) if len(retornos_cola) > 0 else 0.0

    correlacion = retornos.corr().round(4).to_dict()

    rentabilidades_individuales = (retornos.mean() * factor).round(6).to_dict()
    volatilidades_individuales = (retornos.std() * np.sqrt(factor)).round(6).to_dict()

    performance_series = [
        {
            "date": str(index.date()),
            "portfolio_value": round(float(value), 6),
        }
        for index, value in acumulado.items()
    ]

    drawdown_series = [
        {
            "date": str(index.date()),
            "drawdown": round(float(value), 6),
        }
        for index, value in drawdown.items()
    ]

    pesos_utilizados = {
        ticker: round(float(pesos_pct.get(ticker, 0)), 4)
        for ticker in tickers
    }

    activos_detalle = []

    for asset in request.selected_assets:
        ticker = str(asset.get("ticker", "")).upper().strip()

        catalog_asset = validar_activo_con_catalogo(ticker)

        activos_detalle.append({
            "ticker": ticker,
            "name": asset.get("name") or (catalog_asset or {}).get("name"),
            "asset_class": asset.get("asset_class") or (catalog_asset or {}).get("asset_class"),
            "asset_type": asset.get("asset_type") or (catalog_asset or {}).get("asset_type"),
            "risk_level": asset.get("risk_level") or (catalog_asset or {}).get("risk_level"),
            "complexity": asset.get("complexity") or (catalog_asset or {}).get("complexity"),
            "weight": pesos_utilizados.get(ticker, 0),
            "rentabilidad_anualizada": rentabilidades_individuales.get(ticker),
            "volatilidad_anualizada": volatilidades_individuales.get(ticker),
        })

    explicacion = generar_explicacion_simulacion(
        rentabilidad_anual=rentabilidad_anual,
        volatilidad_anual=volatilidad_anual,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        var_95=var_95,
        cvar_95=cvar_95,
        timeframe=request.timeframe or "diario",
        activos=tickers,
    )

    respuesta = {
    "status": "ok",
    "modulo": "portfolio-simulate",
    "validation": validacion,
    "activos": activos_detalle,
    "pesos_utilizados": pesos_utilizados,
    "parametros": {
        "period": request.period,
        "start_date": request.start_date,
        "end_date": request.end_date,
        "timeframe": request.timeframe,
        "annualization_factor": factor,
        "risk_free_rate": risk_free_rate,
        "risk_free_rate_annual_used": risk_free_rate,
    },
    "fecha_inicio": str(precios.index.min().date()),
    "fecha_fin": str(precios.index.max().date()),
    "numero_observaciones": int(len(precios)),
    "metricas_portafolio": {
        "rentabilidad_anualizada": round(rentabilidad_anual, 6),
        "volatilidad_anualizada": round(volatilidad_anual, 6),
        "sharpe_ratio": round(float(sharpe_ratio), 6),
        "max_drawdown": round(max_drawdown, 6),
        "var_95_historico": round(var_95, 6),
        "cvar_95_historico": round(cvar_95, 6),
    },
    "metricas_individuales": {
        "rentabilidad_anualizada": rentabilidades_individuales,
        "volatilidad_anualizada": volatilidades_individuales,
    },
    "matriz_correlacion": correlacion,
    "performance_series": performance_series,
    "drawdown_series": drawdown_series,
    "explicacion_lenguaje_natural": explicacion,
    "advertencia": (
        "Los cálculos se basan en datos históricos y tienen finalidad académica. "
        "No constituyen asesoría financiera personalizada ni garantizan resultados futuros."
    ),
    }

    return limpiar_valores_json(respuesta)

    #=============================================================

def limpiar_valores_json(obj):
    if isinstance(obj, dict):
        return {
            key: limpiar_valores_json(value)
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [
            limpiar_valores_json(item)
            for item in obj
        ]

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    return obj

#=============================================================

# ============================================================
# MÓDULO COMPARADOR ACADÉMICO DE CARTERAS
# ============================================================

class PortfolioCompareRequest(BaseModel):
    resultado: Optional[Dict[str, Any]] = None
    selected_assets: List[Dict[str, Any]]
    user_weights: Dict[str, float]
    period: Optional[str] = "2y"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    timeframe: Optional[str] = "mensual"
    risk_free_rate: Optional[float] = 0.0
    monte_carlo_portfolios: Optional[int] = 3000


def obtener_tickers_activos(selected_assets: List[Dict[str, Any]]) -> List[str]:
    return [
        str(asset.get("ticker", "")).upper().strip()
        for asset in selected_assets
        if str(asset.get("ticker", "")).strip()
    ]


def obtener_composicion_por_clase(
    weights_pct: Dict[str, float],
    selected_assets: List[Dict[str, Any]],
) -> Dict[str, float]:

    renta_fija = 0.0
    renta_variable = 0.0
    otros = 0.0

    for asset in selected_assets:
        ticker = str(asset.get("ticker", "")).upper().strip()
        peso = float(weights_pct.get(ticker, 0.0))
        clase_general = obtener_clase_general_activo(asset.get("asset_class", ""))

        if clase_general == "renta_fija":
            renta_fija += peso
        elif clase_general == "renta_variable":
            renta_variable += peso
        else:
            otros += peso

    return {
        "renta_fija": round(renta_fija, 4),
        "renta_variable": round(renta_variable, 4),
        "otros": round(otros, 4),
        "total": round(renta_fija + renta_variable + otros, 4),
    }


def obtener_restricciones_comparador(resultado: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    restricciones = extraer_restricciones_desde_request(
        AllowedAssetsRequest(resultado=resultado or {})
    )

    return {
        "max_equity_percentage": float(restricciones.get("max_equity_percentage", 100)),
        "min_fixed_income_percentage": float(restricciones.get("min_fixed_income_percentage", 0)),
        "max_weight_per_asset": float(restricciones.get("max_weight_per_asset", 100)),
        "short_positions_allowed": bool(restricciones.get("short_positions_allowed", False)),
        "leverage_allowed": bool(restricciones.get("leverage_allowed", False)),
    }


def validar_pesos_con_restricciones(
    weights_pct: Dict[str, float],
    selected_assets: List[Dict[str, Any]],
    restricciones: Dict[str, Any],
) -> bool:

    total = sum(float(v) for v in weights_pct.values())

    if abs(total - 100.0) > 0.01:
        return False

    max_weight = float(restricciones.get("max_weight_per_asset", 100))
    max_equity = float(restricciones.get("max_equity_percentage", 100))
    min_fixed = float(restricciones.get("min_fixed_income_percentage", 0))

    for peso in weights_pct.values():
        peso_float = float(peso)

        if peso_float < -0.0001:
            return False

        if peso_float > max_weight + 0.0001:
            return False

    composicion = obtener_composicion_por_clase(weights_pct, selected_assets)

    if composicion["renta_variable"] > max_equity + 0.01:
        return False

    if composicion["renta_fija"] + 0.01 < min_fixed:
        return False

    return True


def pesos_dict_a_vector(
    weights_pct: Dict[str, float],
    tickers: List[str],
) -> np.ndarray:

    return np.array([
        float(weights_pct.get(ticker, 0.0)) / 100.0
        for ticker in tickers
    ])


def pesos_vector_a_dict(
    weights_vector: np.ndarray,
    tickers: List[str],
) -> Dict[str, float]:

    return {
        ticker: round(float(weight) * 100.0, 4)
        for ticker, weight in zip(tickers, weights_vector)
    }


def construir_cartera_equiponderada_restringida(
    selected_assets: List[Dict[str, Any]],
    restricciones: Dict[str, Any],
) -> Dict[str, float]:

    tickers = obtener_tickers_activos(selected_assets)
    n = len(tickers)

    if n == 0:
        return {}

    peso_base = 100.0 / n
    max_weight = float(restricciones.get("max_weight_per_asset", 100))

    if peso_base <= max_weight:
        weights = {ticker: round(peso_base, 4) for ticker in tickers}

        if validar_pesos_con_restricciones(weights, selected_assets, restricciones):
            return weights

    return {}


def distribuir_peso_en_grupo(
    tickers_grupo: List[str],
    peso_total: float,
    max_weight: float,
) -> Dict[str, float]:

    if not tickers_grupo or peso_total <= 0:
        return {}

    capacidad_total = len(tickers_grupo) * max_weight

    if peso_total > capacidad_total + 0.0001:
        return {}

    peso_inicial = peso_total / len(tickers_grupo)

    if peso_inicial <= max_weight:
        return {
            ticker: round(peso_inicial, 4)
            for ticker in tickers_grupo
        }

    pesos = {ticker: 0.0 for ticker in tickers_grupo}
    restante = peso_total
    tickers_disponibles = tickers_grupo.copy()

    while restante > 0.0001 and tickers_disponibles:
        peso_por_ticker = restante / len(tickers_disponibles)
        nuevos_disponibles = []

        for ticker in tickers_disponibles:
            asignacion = min(peso_por_ticker, max_weight - pesos[ticker])
            pesos[ticker] += asignacion
            restante -= asignacion

            if pesos[ticker] < max_weight - 0.0001:
                nuevos_disponibles.append(ticker)

        if len(nuevos_disponibles) == len(tickers_disponibles):
            break

        tickers_disponibles = nuevos_disponibles

    return {
        ticker: round(peso, 4)
        for ticker, peso in pesos.items()
        if peso > 0.0001
    }

def construir_cartera_equiponderada_por_bloques(
    selected_assets: List[Dict[str, Any]],
    restricciones: Dict[str, Any],
) -> Dict[str, float]:

    max_equity = float(restricciones.get("max_equity_percentage", 100))
    min_fixed = float(restricciones.get("min_fixed_income_percentage", 0))
    max_weight = float(restricciones.get("max_weight_per_asset", 100))

    tickers_renta_fija = []
    tickers_renta_variable = []
    tickers_otros = []

    for asset in selected_assets:
        ticker = str(asset.get("ticker", "")).upper().strip()
        clase_general = obtener_clase_general_activo(asset.get("asset_class", ""))

        if clase_general == "renta_fija":
            tickers_renta_fija.append(ticker)
        elif clase_general == "renta_variable":
            tickers_renta_variable.append(ticker)
        else:
            tickers_otros.append(ticker)

    capacidad_renta_fija = len(tickers_renta_fija) * max_weight
    capacidad_renta_variable = len(tickers_renta_variable) * max_weight

    if min_fixed > 0 and not tickers_renta_fija:
        return {}

    peso_renta_fija = min_fixed
    peso_renta_variable = 100.0 - peso_renta_fija

    if peso_renta_variable > max_equity:
        peso_renta_variable = max_equity
        peso_renta_fija = 100.0 - peso_renta_variable

    if peso_renta_fija > capacidad_renta_fija + 0.0001:
        return {}

    if peso_renta_variable > capacidad_renta_variable + 0.0001:
        return {}

    if peso_renta_variable > 0 and not tickers_renta_variable:
        return {}

    pesos = {}

    pesos_rf = distribuir_peso_en_grupo(
        tickers_grupo=tickers_renta_fija,
        peso_total=peso_renta_fija,
        max_weight=max_weight,
    )

    pesos_rv = distribuir_peso_en_grupo(
        tickers_grupo=tickers_renta_variable,
        peso_total=peso_renta_variable,
        max_weight=max_weight,
    )

    if peso_renta_fija > 0 and not pesos_rf:
        return {}

    if peso_renta_variable > 0 and not pesos_rv:
        return {}

    pesos.update(pesos_rf)
    pesos.update(pesos_rv)

    total = sum(pesos.values())
    diferencia = round(100.0 - total, 6)

    if abs(diferencia) > 0.0001:
        tickers_ajustables = [
            ticker
            for ticker, peso in pesos.items()
            if peso + diferencia <= max_weight + 0.0001 and peso + diferencia >= -0.0001
        ]

        if tickers_ajustables:
            ticker_ajuste = tickers_ajustables[0]
            pesos[ticker_ajuste] = round(pesos[ticker_ajuste] + diferencia, 4)
        else:
            return {}

    pesos = {
        ticker: round(float(peso), 4)
        for ticker, peso in pesos.items()
        if float(peso) > 0.0001
    }

    if validar_pesos_con_restricciones(pesos, selected_assets, restricciones):
        return pesos

    return {}

def construir_cartera_modelo_perfil(
    selected_assets: List[Dict[str, Any]],
    restricciones: Dict[str, Any],
) -> Dict[str, float]:

    max_equity = float(restricciones.get("max_equity_percentage", 100))
    min_fixed = float(restricciones.get("min_fixed_income_percentage", 0))
    max_weight = float(restricciones.get("max_weight_per_asset", 100))

    tickers_renta_fija = []
    tickers_renta_variable = []
    tickers_otros = []

    for asset in selected_assets:
        ticker = str(asset.get("ticker", "")).upper().strip()
        clase_general = obtener_clase_general_activo(asset.get("asset_class", ""))

        if clase_general == "renta_fija":
            tickers_renta_fija.append(ticker)
        elif clase_general == "renta_variable":
            tickers_renta_variable.append(ticker)
        else:
            tickers_otros.append(ticker)

    peso_renta_fija = min_fixed
    peso_renta_variable = min(100.0 - peso_renta_fija, max_equity)

    if peso_renta_fija + peso_renta_variable < 100.0:
        faltante = 100.0 - peso_renta_fija - peso_renta_variable

        if tickers_renta_fija:
            peso_renta_fija += faltante
        elif tickers_renta_variable:
            peso_renta_variable += faltante

    pesos = {}

    pesos.update(
        distribuir_peso_en_grupo(
            tickers_grupo=tickers_renta_fija,
            peso_total=peso_renta_fija,
            max_weight=max_weight,
        )
    )

    pesos.update(
        distribuir_peso_en_grupo(
            tickers_grupo=tickers_renta_variable,
            peso_total=peso_renta_variable,
            max_weight=max_weight,
        )
    )

    total = sum(pesos.values())

    if abs(total - 100.0) > 0.01:
        return {}

    if validar_pesos_con_restricciones(pesos, selected_assets, restricciones):
        return pesos

    return {}


def generar_distribucion_aleatoria_con_limite(
    tickers_grupo: List[str],
    peso_total: float,
    max_weight: float,
) -> Optional[Dict[str, float]]:

    if not tickers_grupo or peso_total <= 0:
        return {}

    capacidad_total = len(tickers_grupo) * max_weight

    if peso_total > capacidad_total + 0.0001:
        return None

    if abs(peso_total - capacidad_total) <= 0.0001:
        return {
            ticker: round(max_weight, 4)
            for ticker in tickers_grupo
        }

    n = len(tickers_grupo)

    for _ in range(500):
        vector = np.random.dirichlet(np.ones(n))
        pesos = vector * peso_total

        if np.all(pesos <= max_weight + 0.0001):
            return {
                ticker: round(float(peso), 4)
                for ticker, peso in zip(tickers_grupo, pesos)
            }

    return distribuir_peso_en_grupo(
        tickers_grupo=tickers_grupo,
        peso_total=peso_total,
        max_weight=max_weight,
    )


def generar_pesos_aleatorios_restringidos(
    selected_assets: List[Dict[str, Any]],
    restricciones: Dict[str, Any],
    cantidad: int = 3000,
    seed: int = 42,
) -> List[Dict[str, float]]:

    np.random.seed(seed)

    max_weight = float(restricciones.get("max_weight_per_asset", 100))
    max_equity = float(restricciones.get("max_equity_percentage", 100))
    min_fixed = float(restricciones.get("min_fixed_income_percentage", 0))

    tickers_renta_fija = []
    tickers_renta_variable = []
    tickers_otros = []

    for asset in selected_assets:
        ticker = str(asset.get("ticker", "")).upper().strip()
        clase_general = obtener_clase_general_activo(asset.get("asset_class", ""))

        if clase_general == "renta_fija":
            tickers_renta_fija.append(ticker)
        elif clase_general == "renta_variable":
            tickers_renta_variable.append(ticker)
        else:
            tickers_otros.append(ticker)

    candidatos = []

    capacidad_renta_fija = len(tickers_renta_fija) * max_weight
    capacidad_renta_variable = len(tickers_renta_variable) * max_weight

    if capacidad_renta_fija < min_fixed - 0.0001:
        return []

    if capacidad_renta_variable < (100 - min_fixed) - 0.0001:
        return []

    intentos_maximos = cantidad * 20
    intentos = 0

    while len(candidatos) < cantidad and intentos < intentos_maximos:
        intentos += 1

        min_fixed_possible = min_fixed
        max_fixed_possible = min(
            100.0,
            capacidad_renta_fija,
            100.0
        )

        min_fixed_required_by_equity_capacity = max(
            min_fixed_possible,
            100.0 - min(max_equity, capacidad_renta_variable)
        )

        max_fixed_allowed_by_equity_minimum = min(
            max_fixed_possible,
            100.0
        )

        if min_fixed_required_by_equity_capacity > max_fixed_allowed_by_equity_minimum + 0.0001:
            return []

        if abs(min_fixed_required_by_equity_capacity - max_fixed_allowed_by_equity_minimum) <= 0.0001:
            peso_renta_fija = min_fixed_required_by_equity_capacity
        else:
            peso_renta_fija = np.random.uniform(
                min_fixed_required_by_equity_capacity,
                max_fixed_allowed_by_equity_minimum
            )

        peso_renta_variable = 100.0 - peso_renta_fija

        if peso_renta_variable > max_equity + 0.0001:
            continue

        if peso_renta_variable > capacidad_renta_variable + 0.0001:
            continue

        pesos_rf = generar_distribucion_aleatoria_con_limite(
            tickers_grupo=tickers_renta_fija,
            peso_total=peso_renta_fija,
            max_weight=max_weight,
        )

        pesos_rv = generar_distribucion_aleatoria_con_limite(
            tickers_grupo=tickers_renta_variable,
            peso_total=peso_renta_variable,
            max_weight=max_weight,
        )

        if pesos_rf is None or pesos_rv is None:
            continue

        pesos = {}
        pesos.update(pesos_rf)
        pesos.update(pesos_rv)

        for ticker in tickers_otros:
            pesos[ticker] = 0.0

        total = sum(pesos.values())

        if abs(total - 100.0) > 0.05:
            continue

        diferencia = 100.0 - total

        if abs(diferencia) > 0.0001:
            tickers_ajustables = [
                ticker for ticker, peso in pesos.items()
                if peso + diferencia <= max_weight + 0.0001 and peso + diferencia >= -0.0001
            ]

            if tickers_ajustables:
                ticker_ajuste = tickers_ajustables[0]
                pesos[ticker_ajuste] = round(pesos[ticker_ajuste] + diferencia, 4)

        pesos = {
            ticker: round(float(peso), 4)
            for ticker, peso in pesos.items()
            if float(peso) > 0.0001
        }

        if validar_pesos_con_restricciones(pesos, selected_assets, restricciones):
            candidatos.append(pesos)

    return candidatos


def calcular_metricas_de_pesos(
    retornos: pd.DataFrame,
    weights_pct: Dict[str, float],
    tickers: List[str],
    factor: int,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:

    vector = pesos_dict_a_vector(weights_pct, tickers)
    retornos_portafolio = retornos.dot(vector)

    rentabilidad_anual = float(retornos_portafolio.mean() * factor)
    volatilidad_anual = float(retornos_portafolio.std() * np.sqrt(factor))

    sharpe = (
        (rentabilidad_anual - risk_free_rate) / volatilidad_anual
        if volatilidad_anual > 0
        else 0.0
    )

    acumulado = (1 + retornos_portafolio).cumprod()
    drawdown = acumulado / acumulado.cummax() - 1
    max_drawdown = float(drawdown.min())

    percentil_5 = float(np.percentile(retornos_portafolio, 5))
    var_95 = max(0.0, -percentil_5)

    cola = retornos_portafolio[retornos_portafolio <= percentil_5]
    cvar_95 = max(0.0, -float(cola.mean())) if len(cola) > 0 else 0.0

    return {
        "rentabilidad_anualizada": round(rentabilidad_anual, 6),
        "volatilidad_anualizada": round(volatilidad_anual, 6),
        "sharpe_ratio": round(float(sharpe), 6),
        "max_drawdown": round(max_drawdown, 6),
        "var_95_historico": round(var_95, 6),
        "cvar_95_historico": round(cvar_95, 6),
    }


def construir_series_portafolio(
    retornos: pd.DataFrame,
    weights_pct: Dict[str, float],
    tickers: List[str],
) -> Dict[str, List[Dict[str, Any]]]:

    vector = pesos_dict_a_vector(weights_pct, tickers)
    retornos_portafolio = retornos.dot(vector)

    acumulado = (1 + retornos_portafolio).cumprod()
    drawdown = acumulado / acumulado.cummax() - 1

    performance_series = [
        {
            "date": str(index.date()),
            "portfolio_value": round(float(value), 6),
        }
        for index, value in acumulado.items()
    ]

    drawdown_series = [
        {
            "date": str(index.date()),
            "drawdown": round(float(value), 6),
        }
        for index, value in drawdown.items()
    ]

    return {
        "performance_series": performance_series,
        "drawdown_series": drawdown_series,
    }


def seleccionar_mejores_carteras_montecarlo(
    candidatos: List[Dict[str, float]],
    retornos: pd.DataFrame,
    tickers: List[str],
    factor: int,
    risk_free_rate: float,
) -> Dict[str, Dict[str, Any]]:

    evaluadas = []

    for pesos in candidatos:
        metricas = calcular_metricas_de_pesos(
            retornos=retornos,
            weights_pct=pesos,
            tickers=tickers,
            factor=factor,
            risk_free_rate=risk_free_rate,
        )

        evaluadas.append({
            "weights": pesos,
            "metrics": metricas,
        })

    if not evaluadas:
        return {}

    minima_varianza = min(
        evaluadas,
        key=lambda x: x["metrics"]["volatilidad_anualizada"],
    )

    maximo_sharpe = max(
        evaluadas,
        key=lambda x: x["metrics"]["sharpe_ratio"],
    )

    return {
        "minima_varianza": minima_varianza,
        "maximo_sharpe": maximo_sharpe,
        "evaluadas": evaluadas,
    }


def generar_interpretacion_comparador(
    portfolios: List[Dict[str, Any]],
) -> str:

    if not portfolios:
        return (
            "No fue posible generar una comparación académica de carteras con la información disponible."
        )

    usuario = next((p for p in portfolios if p.get("id") == "usuario"), None)

    menor_volatilidad = min(
        portfolios,
        key=lambda x: x["metrics"]["volatilidad_anualizada"],
    )

    mayor_sharpe = max(
        portfolios,
        key=lambda x: x["metrics"]["sharpe_ratio"],
    )

    if usuario:
        texto_usuario = (
            f"La cartera construida por el usuario presenta una rentabilidad anualizada histórica "
            f"de {usuario['metrics']['rentabilidad_anualizada'] * 100:.2f}%, una volatilidad "
            f"de {usuario['metrics']['volatilidad_anualizada'] * 100:.2f}% y un ratio de Sharpe "
            f"de {usuario['metrics']['sharpe_ratio']:.2f}. "
        )
    else:
        texto_usuario = ""

    return (
        texto_usuario
        + f"Al comparar las carteras académicas de referencia, la alternativa con menor volatilidad "
        f"histórica es '{menor_volatilidad['name']}', mientras que la alternativa con mejor relación "
        f"rentabilidad-riesgo histórica, medida por el ratio de Sharpe, es '{mayor_sharpe['name']}'. "
        f"Estos resultados no deben entenderse como una recomendación personalizada de inversión, "
        f"sino como una comparación académica que permite observar cómo cambian las métricas cuando "
        f"se modifican los pesos de los activos bajo las restricciones del perfil."
    )


@app.post("/api/portfolio/compare")
def comparar_carteras_academicas(request: PortfolioCompareRequest):

    if not request.selected_assets:
        raise HTTPException(
            status_code=400,
            detail="No se recibieron activos seleccionados para comparar carteras.",
        )

    tickers = obtener_tickers_activos(request.selected_assets)

    if not tickers:
        raise HTTPException(
            status_code=400,
            detail="No se encontraron tickers válidos.",
        )

    restricciones = obtener_restricciones_comparador(request.resultado)

    validacion_usuario = validar_portafolio_academico(
        PortfolioValidationRequest(
            resultado=request.resultado,
            selected_assets=request.selected_assets,
            weights=request.user_weights,
        )
    )

    if not validacion_usuario.get("is_valid"):
        return {
            "status": "error",
            "modulo": "portfolio-compare",
            "validation": validacion_usuario,
            "mensaje": (
                "La cartera del usuario no puede compararse porque no cumple las restricciones "
                "académicas del perfil."
            ),
        }

    precios = descargar_precios_yfinance_simulacion(
        tickers=tickers,
        period=request.period or "2y",
        start_date=request.start_date,
        end_date=request.end_date,
    )

    precios = precios[tickers].dropna()
    precios = aplicar_marco_temporal(precios, request.timeframe or "mensual")

    retornos = precios.pct_change().dropna()

    if retornos.empty:
        raise HTTPException(
            status_code=400,
            detail="No fue posible calcular retornos para comparar las carteras.",
        )

    factor = obtener_factor_anualizacion(request.timeframe or "mensual")
    risk_free_rate = float(request.risk_free_rate or 0.0)

    portfolios = []

    # 1. Cartera del usuario
    user_metrics = calcular_metricas_de_pesos(
        retornos=retornos,
        weights_pct=request.user_weights,
        tickers=tickers,
        factor=factor,
        risk_free_rate=risk_free_rate,
    )

    user_series = construir_series_portafolio(
        retornos=retornos,
        weights_pct=request.user_weights,
        tickers=tickers,
    )

    portfolios.append({
        "id": "usuario",
        "name": "Cartera del usuario",
        "type": "manual",
        "description": (
            "Cartera construida manualmente por el usuario y validada contra las restricciones "
            "académicas de su perfil."
        ),
        "weights": {
            ticker: round(float(request.user_weights.get(ticker, 0)), 4)
            for ticker in tickers
        },
        "composition": obtener_composicion_por_clase(request.user_weights, request.selected_assets),
        "metrics": user_metrics,
        "series": user_series,
    })

    # 2. Cartera equiponderada restringida
    pesos_equal = construir_cartera_equiponderada_restringida(
        selected_assets=request.selected_assets,
        restricciones=restricciones,
    )

    if pesos_equal:
        portfolios.append({
            "id": "equiponderada_restringida",
            "name": "Equiponderada restringida",
            "type": "academic_reference",
            "description": (
                "Distribuye el capital en partes iguales entre los activos seleccionados, siempre que "
                "cumpla los límites máximos por activo y las restricciones del perfil."
            ),
            "weights": pesos_equal,
            "composition": obtener_composicion_por_clase(pesos_equal, request.selected_assets),
            "metrics": calcular_metricas_de_pesos(
                retornos=retornos,
                weights_pct=pesos_equal,
                tickers=tickers,
                factor=factor,
                risk_free_rate=risk_free_rate,
            ),
            "series": construir_series_portafolio(retornos, pesos_equal, tickers),
        })
        
    # 2B. Cartera equiponderada por bloques
    pesos_equal_blocks = construir_cartera_equiponderada_por_bloques(
        selected_assets=request.selected_assets,
        restricciones=restricciones,
    )

    if pesos_equal_blocks:
        portfolios.append({
            "id": "equiponderada_por_bloques",
            "name": "Equiponderada por bloques",
            "type": "academic_reference",
            "description": (
                "Distribuye el capital de forma equilibrada dentro de cada bloque de activo. "
                "Primero respeta el mínimo de renta fija y el máximo de renta variable definidos "
                "por el perfil, y luego reparte el peso de manera uniforme entre los activos de "
                "cada bloque. No constituye una recomendación personalizada de inversión."
            ),
            "weights": pesos_equal_blocks,
            "composition": obtener_composicion_por_clase(
                pesos_equal_blocks,
                request.selected_assets,
            ),
            "metrics": calcular_metricas_de_pesos(
                retornos=retornos,
                weights_pct=pesos_equal_blocks,
                tickers=tickers,
                factor=factor,
                risk_free_rate=risk_free_rate,
            ),
            "series": construir_series_portafolio(
                retornos,
                pesos_equal_blocks,
                tickers,
            ),
        })
    
    # 3. Cartera modelo académica según perfil
    pesos_modelo = construir_cartera_modelo_perfil(
        selected_assets=request.selected_assets,
        restricciones=restricciones,
    )

    if pesos_modelo:
        portfolios.append({
            "id": "modelo_perfil",
            "name": "Modelo académico por perfil",
            "type": "academic_reference",
            "description": (
                "Cartera de referencia construida a partir de la matriz riesgo-conocimiento y las "
                "restricciones académicas del perfil. No representa una recomendación personalizada."
            ),
            "weights": pesos_modelo,
            "composition": obtener_composicion_por_clase(pesos_modelo, request.selected_assets),
            "metrics": calcular_metricas_de_pesos(
                retornos=retornos,
                weights_pct=pesos_modelo,
                tickers=tickers,
                factor=factor,
                risk_free_rate=risk_free_rate,
            ),
            "series": construir_series_portafolio(retornos, pesos_modelo, tickers),
        })

    # 4. Monte Carlo restringido para mínima varianza y máximo Sharpe
    cantidad_mc = int(request.monte_carlo_portfolios or 3000)
    cantidad_mc = max(500, min(cantidad_mc, 10000))

    candidatos = generar_pesos_aleatorios_restringidos(
        selected_assets=request.selected_assets,
        restricciones=restricciones,
        cantidad=cantidad_mc,
        seed=42,
    )

    seleccion_mc = seleccionar_mejores_carteras_montecarlo(
        candidatos=candidatos,
        retornos=retornos,
        tickers=tickers,
        factor=factor,
        risk_free_rate=risk_free_rate,
    )

    if seleccion_mc.get("minima_varianza"):
        pesos_min_var = seleccion_mc["minima_varianza"]["weights"]

        portfolios.append({
            "id": "minima_varianza",
            "name": "Mínima volatilidad histórica",
            "type": "optimization_reference",
            "description": (
                "Cartera académica que busca la menor volatilidad histórica dentro de las restricciones "
                "del perfil y los activos seleccionados."
            ),
            "weights": pesos_min_var,
            "composition": obtener_composicion_por_clase(pesos_min_var, request.selected_assets),
            "metrics": seleccion_mc["minima_varianza"]["metrics"],
            "series": construir_series_portafolio(retornos, pesos_min_var, tickers),
        })

    if seleccion_mc.get("maximo_sharpe"):
        pesos_max_sharpe = seleccion_mc["maximo_sharpe"]["weights"]

        portfolios.append({
            "id": "maximo_sharpe",
            "name": "Máximo Sharpe histórico",
            "type": "optimization_reference",
            "description": (
                "Cartera académica que busca la mejor relación histórica entre rentabilidad y riesgo, "
                "medida mediante el ratio de Sharpe, respetando las restricciones del perfil."
            ),
            "weights": pesos_max_sharpe,
            "composition": obtener_composicion_por_clase(pesos_max_sharpe, request.selected_assets),
            "metrics": seleccion_mc["maximo_sharpe"]["metrics"],
            "series": construir_series_portafolio(retornos, pesos_max_sharpe, tickers),
        })

    comparison_table = []

    for portfolio in portfolios:
        comparison_table.append({
            "id": portfolio["id"],
            "name": portfolio["name"],
            "rentabilidad_anualizada": portfolio["metrics"]["rentabilidad_anualizada"],
            "volatilidad_anualizada": portfolio["metrics"]["volatilidad_anualizada"],
            "sharpe_ratio": portfolio["metrics"]["sharpe_ratio"],
            "max_drawdown": portfolio["metrics"]["max_drawdown"],
            "var_95_historico": portfolio["metrics"]["var_95_historico"],
            "cvar_95_historico": portfolio["metrics"]["cvar_95_historico"],
            "renta_fija": portfolio["composition"]["renta_fija"],
            "renta_variable": portfolio["composition"]["renta_variable"],
            "otros": portfolio["composition"]["otros"],
        })

    risk_return_points = [
        {
            "id": portfolio["id"],
            "name": portfolio["name"],
            "risk": portfolio["metrics"]["volatilidad_anualizada"],
            "return": portfolio["metrics"]["rentabilidad_anualizada"],
            "sharpe": portfolio["metrics"]["sharpe_ratio"],
        }
        for portfolio in portfolios
    ]

    best_by_criteria = {
        "menor_volatilidad": min(
            portfolios,
            key=lambda x: x["metrics"]["volatilidad_anualizada"],
        )["name"],
        "mayor_sharpe": max(
            portfolios,
            key=lambda x: x["metrics"]["sharpe_ratio"],
        )["name"],
        "mayor_rentabilidad": max(
            portfolios,
            key=lambda x: x["metrics"]["rentabilidad_anualizada"],
        )["name"],
        "menor_drawdown": max(
            portfolios,
            key=lambda x: x["metrics"]["max_drawdown"],
        )["name"],
    }

    respuesta = {
        "status": "ok",
        "modulo": "portfolio-compare",
        "validation": validacion_usuario,
        "restricciones": restricciones,
        "parametros": {
            "period": request.period,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "timeframe": request.timeframe,
            "annualization_factor": factor,
            "risk_free_rate": risk_free_rate,
            "monte_carlo_portfolios": len(candidatos),
        },
        "fecha_inicio": str(precios.index.min().date()),
        "fecha_fin": str(precios.index.max().date()),
        "numero_observaciones": int(len(precios)),
        "portfolios": portfolios,
        "comparison_table": comparison_table,
        "risk_return_points": risk_return_points,
        "best_by_criteria": best_by_criteria,
        "academic_interpretation": generar_interpretacion_comparador(portfolios),
        "advertencia": (
            "Las carteras comparadas tienen finalidad académica y se construyen con base en datos "
            "históricos y restricciones del perfil. No constituyen asesoría financiera personalizada, "
            "recomendación de inversión ni garantía de resultados futuros."
        ),
    }

    return limpiar_valores_json(respuesta)
    
