from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from io import StringIO

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI(
    title="CLARUM Invest API",
    description="Backend académico en Python para cálculos financieros de CLARUM Invest.",
    version="0.2.5",
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
        "version": "0.2.5",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "servicio": "CLARUM Invest API",
        "version": "0.2.5",
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
        },
        "validated_assets": activos_validados,
        "mensaje_academico": mensaje,
        "advertencia": (
            "La validación tiene finalidad académica y simulativa. "
            "No constituye asesoría financiera personalizada."
        ),
    }

#==========================================================================
