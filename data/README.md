# Data Directory

Este directorio contiene datos para el optimizador de portafolios.

## Estructura Recomendada

```
data/
├── raw/                  # Datos sin procesar
│   ├── prices.csv       # Precios históricos
│   └── fundamentals/    # Datos fundamentales
└── processed/           # Datos procesados
    ├── returns.csv      # Retornos calculados
    └── features/        # Features para ML
```

## Formato de Datos

### Precios Históricos (`prices.csv`)

```
Date,ASSET1,ASSET2,ASSET3,...
2020-01-01,100.0,50.0,75.0,...
2020-01-02,101.5,49.8,76.2,...
...
```

- Index: Fechas (formato ISO)
- Columns: Símbolos de activos
- Values: Precios de cierre ajustados
