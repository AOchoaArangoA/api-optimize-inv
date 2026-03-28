"""
Visualization utilities for portfolio optimization.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple

def plot_price_evolution(data: pd.DataFrame, title: str = "Normalized Price Evolution (Base 100)") -> go.Figure:
    """
    Crea un gráfico de línea de la evolución de precios normalizados.
    """
    normalized_data = data.divide(data.iloc[0]) * 100
    fig = px.line(normalized_data, title=title)
    fig.update_layout(template="plotly_dark", xaxis_title="Date", yaxis_title="Price (Base 100)")
    return fig

def plot_asset_weights(
    weights: Dict[str, float], 
    asset_classes: Optional[Dict[str, str]] = None,
    title: str = "Portfolio Allocation by Asset"
) -> go.Figure:
    """
    Crea un gráfico de barras de los pesos por activo.
    """
    df = pd.DataFrame(list(weights.items()), columns=['Asset', 'Weight'])
    
    color_col = None
    if asset_classes:
        df['Class'] = [asset_classes.get(a, 'Unclassified') for a in df['Asset']]
        color_col = 'Class'
        
    fig = px.bar(
        df, 
        x='Asset', 
        y='Weight', 
        color=color_col,
        title=title,
        text_auto='.2%'
    )
    fig.update_layout(template="plotly_dark", yaxis_tickformat=".0%")
    return fig

def plot_class_allocation(
    class_weights: Dict[str, float], 
    title: str = "Allocation by Asset Class"
) -> go.Figure:
    """
    Crea un gráfico de torta (donut) de la asignación por clase de activo.
    """
    # Convertir keys (que pueden ser Enums) a strings si es necesario
    labels = [str(k.value) if hasattr(k, 'value') else str(k) for k in class_weights.keys()]
    values = list(class_weights.values())
    
    fig = px.pie(
        values=values, 
        names=labels, 
        hole=0.4,
        title=title
    )
    fig.update_layout(template="plotly_dark")
    fig.update_traces(textinfo='percent+label')
    return fig

def plot_efficient_frontier(
    frontier_returns: List[float],
    frontier_vols: List[float],
    asset_returns: List[float],
    asset_vols: List[float],
    asset_names: List[str],
    current_port_return: Optional[float] = None,
    current_port_vol: Optional[float] = None,
    title: str = "Efficient Frontier (Risk-Return Profile)"
) -> go.Figure:
    """
    Grafica la frontera eficiente, activos individuales y el portafolio actual.
    """
    fig = go.Figure()
    
    # Ordenar puntos de la frontera por volatilidad para trazar una línea suave
    if frontier_vols and frontier_returns:
        # Zip, sort by vol, unzip
        sorted_points = sorted(zip(frontier_vols, frontier_returns))
        sorted_vols, sorted_rets = zip(*sorted_points)
        
        fig.add_trace(go.Scatter(
            x=sorted_vols, 
            y=sorted_rets,
            mode='lines+markers',
            name='Efficient Frontier',
            line=dict(color='royalblue', width=3),
            marker=dict(size=6)
        ))
    
    # 2. Activos Individuales
    fig.add_trace(go.Scatter(
        x=asset_vols, 
        y=asset_returns,
        mode='markers+text',
        name='Individual Assets',
        text=asset_names,
        textposition="top center",
        marker=dict(size=10, color='red', symbol='circle')
    ))
    
    # 3. Portafolio Actual (si se provee)
    if current_port_return is not None and current_port_vol is not None:
        fig.add_trace(go.Scatter(
            x=[current_port_vol], 
            y=[current_port_return],
            mode='markers',
            name='Selected Portfolio',
            marker=dict(size=15, color='gold', symbol='star', line=dict(width=2, color='black'))
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Annualized Volatility (Risk)",
        yaxis_title="Expected Annual Return",
        template="plotly_dark",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig

def create_rebalancing_table(trades: List[Dict]) -> pd.DataFrame:
    """
    Formatea la lista de trades para mostrar en la UI.
    Esperando lista de dicts con keys: 'Asset', 'Current', 'Target', 'Action', 'Amount'
    """
    return pd.DataFrame(trades)


def plot_correlation_heatmap(
    cov_matrix: np.ndarray,
    asset_names: List[str],
    title: str = "Matriz de Correlación"
) -> go.Figure:
    """
    Grafica la matriz de correlación a partir de la matriz de covarianza.
    """
    n = len(cov_matrix)
    vol = np.sqrt(np.diag(cov_matrix))
    vol_outer = np.outer(vol, vol)
    corr = np.where(vol_outer > 0, cov_matrix / vol_outer, 0)
    np.fill_diagonal(corr, 1.0)

    fig = go.Figure(data=go.Heatmap(
        z=corr,
        x=asset_names,
        y=asset_names,
        colorscale="RdBu",
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False,
    ))
    fig.update_layout(
        title=title,
        template="plotly_dark",
        xaxis=dict(tickangle=-45),
        yaxis=dict(autorange="reversed"),
        width=600,
        height=600,
    )
    return fig


def plot_risk_return_scatter(
    expected_returns: np.ndarray,
    volatilities: np.ndarray,
    asset_names: List[str],
    port_return: Optional[float] = None,
    port_vol: Optional[float] = None,
    title: str = "Riesgo vs Retorno por Activo",
) -> go.Figure:
    """
    Gráfico de dispersión: volatilidad (eje X) vs retorno esperado (eje Y) por activo.
    Opcionalmente marca el portafolio óptimo.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=volatilities,
        y=expected_returns,
        mode="markers+text",
        name="Activos",
        text=asset_names,
        textposition="top center",
        marker=dict(size=12, color="steelblue", line=dict(width=1, color="white")),
    ))
    if port_return is not None and port_vol is not None:
        fig.add_trace(go.Scatter(
            x=[port_vol],
            y=[port_return],
            mode="markers+text",
            name="Portafolio Óptimo",
            text=["Portafolio"],
            textposition="top center",
            marker=dict(size=18, color="gold", symbol="star", line=dict(width=2, color="black")),
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Volatilidad (Riesgo)",
        yaxis_title="Retorno Esperado",
        template="plotly_dark",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    return fig
