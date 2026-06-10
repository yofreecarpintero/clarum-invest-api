from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="CLARUM Invest API",
    description="Backend académico en Python para cálculos financieros de CLARUM Invest.",
    version="0.2.1",
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


def limpiar_tickers(activos: List[str]) -> List[str]:
    tickers = []

    for activo in activos:
        ticker = str(activo).strip().upper()
        if ticker:
            tickers.append(ticker)

    tickers_unicos = list(dict.fromkeys(tickers))

    if len(tickers_unicos) < 1:
        raise HTTPException(
            status_code=400,
            detail="Debes enviar al menos un activo válido."
        )

    if len(tickers_unicos) > 10:
        raise HTTPException(
            status_code=400,
            detail="Por ahora el análisis permite máximo 10 activos."
        )

    return tickers_unicos


def extraer_serie_cierre(data: pd.DataFrame, ticker: str) -> pd.Series:
    if data.empty:
        raise HTTPException(
            status_code=400,
            detail=f"No se encontraron datos para el activo {ticker}."
        )

    if isinstance(data.columns, pd.MultiIndex):
        if ("Close", ticker) in data.columns:
            serie = data[("Close", ticker)]
        elif ("Adj Close", ticker) in data.columns:
            serie = data[("Adj Close", ticker)]
        else:
            columnas_disponibles = [str(col) for col in data.columns.tolist()]
            raise HTTPException(
                status_code=400,
                detail={
                    "mensaje": f"No se encontró columna de cierre para {ticker}.",
                    "columnas_disponibles": columnas_disponibles,
                },
            )
    else:
        if "Close" in data.columns:
            serie = data["Close"]
        elif "Adj Close" in data.columns:
            serie = data["Adj Close"]
        else:
            columnas_disponibles = [str(col) for col in data.columns.tolist()]
            raise HTTPException(
                status_code=400,
                detail={
                    "mensaje": f"No se encontró columna de cierre para {ticker}.",
                    "columnas_disponibles": columnas_disponibles,
                },
            )

    serie = pd.to_numeric(serie, errors="coerce").dropna()

    if serie.empty:
        raise HTTPException(
            status_code=400,
            detail=f"La serie de precios para {ticker} está vacía después de limpiar los datos."
        )

    serie.name = ticker
    return serie


def obtener_precios(activos: List[str], periodo: str = "2y") -> pd.DataFrame:
    series = []

    for ticker in activos:
        try:
            data = yf.download(
                ticker,
                period=periodo,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )

            serie = extraer_serie_cierre(data, ticker)
            series.append(serie)

        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(
                status_code=500,
                detail=f"Error descargando datos de {ticker}: {str(error)}"
            )

    precios_df = pd.concat(series, axis=1).dropna()

    if precios_df.empty or len(precios_df) < 30:
        raise HTTPException(
            status_code=400,
            detail="No hay suficientes datos históricos para calcular el análisis."
        )

    return precios_df


def obtener_pesos(payload: Dict[str, Any], activos: List[str]) -> np.ndarray:
    pesos_recibidos = payload.get("pesos")

    if pesos_recibidos is None:
        return np.array([1 / len(activos)] * len(activos))

    if not isinstance(pesos_recibidos, list):
        raise HTTPException(
            status_code=400,
            detail="El campo 'pesos' debe ser una lista de números."
        )

    if len(pesos_recibidos) != len(activos):
        raise HTTPException(
            status_code=400,
            detail="La cantidad de pesos debe coincidir con la cantidad de activos."
        )

    pesos = np.array(pesos_recibidos, dtype=float)

    if np.any(pesos < 0):
        raise HTTPException(
            status_code=400,
            detail="Los pesos no pueden ser negativos."
        )

    suma_pesos = pesos.sum()

    if suma_pesos <= 0:
        raise HTTPException(
            status_code=400,
            detail="La suma de los pesos debe ser mayor que cero."
        )

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
        f"{nivel_riesgo}. El máximo drawdown fue de {drawdown_pct:.2f}%, que representa "
        f"la mayor caída histórica desde un punto alto hasta un punto bajo dentro del periodo "
        f"analizado. El VaR histórico diario al 95% fue de {var_pct:.2f}%, lo que significa que, "
        f"en condiciones normales de mercado, la pérdida diaria no debería superar ese porcentaje "
        f"en aproximadamente 95 de cada 100 días. El CVaR fue de {cvar_pct:.2f}%, y representa "
        f"una estimación de la pérdida promedio en los peores escenarios observados."
    )


def calcular_metricas_financieras(payload: Dict[str, Any]) -> Dict[str, Any]:
    activos = limpiar_tickers(payload.get("activos", []))
    periodo = str(payload.get("periodo", "2y"))

    precios = obtener_precios(activos, periodo)
    pesos = obtener_pesos(payload, activos)

    retornos_diarios = precios.pct_change().dropna()

    if retornos_diarios.empty:
        raise HTTPException(
            status_code=400,
            detail="No fue posible calcular retornos diarios con los datos disponibles."
        )

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

    rentabilidades_individuales = (
        retornos_diarios.mean() * 252
    ).round(6).to_dict()

    volatilidades_individuales = (
        retornos_diarios.std() * np.sqrt(252)
    ).round(6).to_dict()

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
        "version": "0.2.1",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "servicio": "CLARUM Invest API",
        "version": "0.2.1",
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
    }


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
