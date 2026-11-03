"""
================================================================================
SCRIPT: analise_econometrica.py
OBJETIVO: Pipeline integrado de Econometria e Engenharia de Dados:
          1. Ingestão e tratamento dos dados do Ipeadata (2014)
          2. Estatística descritiva e matriz de dispersão/correlação
          3. Modelagem MQO (OLS) e comprovação analítica do Viés de Wooldridge
          4. Exportação de gráficos estáticos em alta definição (300 DPI)
          5. Geração de Simulador Visual Interativo com reação total a todas as variáveis
          6. Elaboração do Plano de Apresentação do Trabalho
================================================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração de encoding para terminal Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Diretórios estruturados
DIR_DADOS = "dados"
DIR_IMAGENS = "imagens"
DIR_RELATORIOS = "relatorios"

for d in [DIR_DADOS, DIR_IMAGENS, DIR_RELATORIOS]:
    os.makedirs(d, exist_ok=True)


# ==============================================================================
# 1. INGESTÃO E TRATAMENTO DOS DADOS
# ==============================================================================
def ler_csv_ipea(nome_arquivo: str) -> pd.DataFrame:
    """Lê CSV do Ipeadata identificando encoding, separador e cabeçalho."""
    caminho = os.path.join(DIR_DADOS, nome_arquivo) if os.path.exists(os.path.join(DIR_DADOS, nome_arquivo)) else nome_arquivo
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo não localizado: '{caminho}'")

    linhas, enc_usado = None, None
    for enc in ["utf-8-sig", "utf-8", "latin1", "cp1252"]:
        try:
            with open(caminho, "r", encoding=enc) as f:
                linhas = f.readlines()
                enc_usado = enc
                break
        except Exception:
            continue

    # Heurística para linha do cabeçalho
    idx_header = 0
    for idx, l in enumerate(linhas):
        l_low = l.lower()
        if ("sigla" in l_low or "código" in l_low or "codigo" in l_low or "uf" in l_low) and ("2014" in l_low or "estado" in l_low):
            idx_header = idx
            break

    sep = ";" if linhas[idx_header].count(";") >= linhas[idx_header].count(",") else ","
    df = pd.read_csv(caminho, skiprows=idx_header, sep=sep, encoding=enc_usado, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False)]
    return df


def parse_float_br(val) -> float:
    """Converte números em formato string brasileiro para float64."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if "." in s and "," in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    return float(s)


def carregar_e_tratar_dados() -> pd.DataFrame:
    """Processa e consolida as 4 bases estaduais para 2014."""
    print("=" * 75)
    print("ETAPA 1: INGESTÃO E CONSOLIDAÇÃO DOS DADOS")
    print("=" * 75)

    df_hom = ler_csv_ipea("homicidios.csv")
    df_pib = ler_csv_ipea("pib.csv")
    df_gini = ler_csv_ipea("gini.csv")
    df_des = ler_csv_ipea("desemprego.csv")

    for d in [df_hom, df_pib, df_gini, df_des]:
        col_uf = [c for c in d.columns if c.lower() in ["sigla", "uf", "sg_uf"]][0]
        d.rename(columns={col_uf: "uf"}, inplace=True)
        d["uf"] = d["uf"].astype(str).str.strip().str.upper()

    col_2014_hom = [c for c in df_hom.columns if "2014" in c][0]
    col_2014_pib = [c for c in df_pib.columns if "2014" in c][0]
    col_2014_gini = [c for c in df_gini.columns if "2014" in c][0]
    cols_2014_des = [c for c in df_des.columns if "2014" in c]

    df_hom["taxa_homicidios"] = df_hom[col_2014_hom].apply(parse_float_br)
    df_pib["pib_per_capita"] = df_pib[col_2014_pib].apply(parse_float_br)
    df_gini["gini"] = df_gini[col_2014_gini].apply(parse_float_br)

    for c in cols_2014_des:
        df_des[c] = df_des[c].apply(parse_float_br)
    df_des["desemprego"] = df_des[cols_2014_des].mean(axis=1)

    df = (
        df_hom[["uf", "taxa_homicidios"]]
        .merge(df_pib[["uf", "pib_per_capita"]], on="uf")
        .merge(df_gini[["uf", "gini"]], on="uf")
        .merge(df_des[["uf", "desemprego"]], on="uf")
    )

    df["ln_pib_pc"] = np.log(df["pib_per_capita"])
    df.sort_values(by="uf", inplace=True)
    df.reset_index(drop=True, inplace=True)

    caminho_csv = os.path.join(DIR_DADOS, "base_consolidada_2014.csv")
    df.to_csv(caminho_csv, index=False, sep=";", decimal=",", encoding="utf-8-sig")
    print(f"-> Base consolidada salva em '{caminho_csv}' ({len(df)} UFs).")
    return df


# ==============================================================================
# 2. ESTATÍSTICA DESCRITIVA E VISUALIZAÇÃO ESTÁTICA
# ==============================================================================
def calcular_descritivas(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula estatísticas descritivas amostrais e exporta tabela."""
    print("\n" + "=" * 75)
    print("ETAPA 2: ESTATÍSTICA DESCRITIVA (27 UFs - 2014)")
    print("=" * 75)
    variaveis = ["taxa_homicidios", "pib_per_capita", "gini", "desemprego"]
    linhas = []

    for v in variaveis:
        s = df[v]
        media = s.mean()
        dp = s.std(ddof=1)
        linhas.append({
            "Variável": v,
            "N": int(s.count()),
            "Média": media,
            "Mediana": s.median(),
            "Desvio-Padrão": dp,
            "CV (%)": (dp / media) * 100 if media != 0 else np.nan,
            "Mínimo": s.min(),
            "Máximo": s.max(),
            "Amplitude": s.max() - s.min()
        })

    tabela = pd.DataFrame(linhas)
    caminho_tabela = os.path.join(DIR_DADOS, "tabela1_descritiva.csv")
    tabela.to_csv(caminho_tabela, index=False, sep=";", decimal=",", encoding="utf-8-sig")

    try:
        print(tabela.to_markdown(index=False))
    except Exception:
        print(tabela.to_string(index=False))

    print(f"-> Tabela descritiva exportada para '{caminho_tabela}'.")
    return tabela


def gerar_graficos_estaticos(df: pd.DataFrame, modelos: dict):
    """Gera gráficos de dispersão, mapa de calor e comparação OVB em 300 DPI."""
    print("\n" + "=" * 75)
    print("ETAPA 3: EXPORTAÇÃO DE GRÁFICOS EM ALTA DEFINIÇÃO (300 DPI)")
    print("=" * 75)

    sns.set_theme(style="whitegrid")
    labels = {
        "taxa_homicidios": "Homicídios (Y)",
        "ln_pib_pc": "ln(PIB pc) (X1)",
        "gini": "Gini (X2)",
        "desemprego": "Desemprego % (X3)"
    }
    df_plot = df[["taxa_homicidios", "ln_pib_pc", "gini", "desemprego"]].rename(columns=labels)

    # 1. Pairplot
    g = sns.pairplot(
        df_plot,
        kind="reg",
        plot_kws={"line_kws": {"color": "#c0392b", "linewidth": 1.6}, "scatter_kws": {"alpha": 0.7, "color": "#2c3e50", "s": 40}},
        diag_kws={"kde": True, "color": "#2980b9", "fill": True}
    )
    g.figure.suptitle("Matriz de Dispersão e Tendências Lineares (2014)", y=1.02, fontsize=13, fontweight="bold")
    caminho_pair = os.path.join(DIR_IMAGENS, "grafico_dispersao_pares.png")
    g.savefig(caminho_pair, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"-> Gráfico salvo: '{caminho_pair}'")

    # 2. Heatmap
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    corr = df_plot.corr()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".3f",
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=1.2,
        annot_kws={"size": 11, "fontweight": "bold"},
        ax=ax
    )
    # Posiciona a legenda/rótulos das variáveis na parte superior (abaixo do título e acima do gráfico)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.setp(ax.get_xticklabels(), fontsize=10, fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontsize=10, fontweight="bold")
    plt.title("Matriz de Correlação de Pearson (2014)", fontsize=13, fontweight="bold", pad=20)
    caminho_heat = os.path.join(DIR_IMAGENS, "grafico_correlacao_heatmap.png")
    plt.savefig(caminho_heat, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"-> Gráfico salvo: '{caminho_heat}'")

    # 3. Gráfico 1: Regressão Simples (MQO Bivariado)
    m3 = modelos["m3"]
    b0_s = m3.params["const"]
    b1_s = m3.params["ln_pib_pc"]

    plt.figure(figsize=(8.5, 6))
    x_grid = np.linspace(df["ln_pib_pc"].min() - 0.05, df["ln_pib_pc"].max() + 0.05, 100)
    y_simples = b0_s + b1_s * x_grid

    plt.scatter(df["ln_pib_pc"], df["taxa_homicidios"], color="#2c3e50", s=65, alpha=0.85, edgecolors="white", linewidth=1.2, label="Estados Brasileiros (27 UFs)", zorder=4)
    for _, r in df[df["uf"].isin(["SP", "SC", "AL", "SE", "DF", "RJ"])].iterrows():
        plt.annotate(r["uf"], (r["ln_pib_pc"], r["taxa_homicidios"]), textcoords="offset points", xytext=(6, 5), fontsize=9, fontweight="bold", color="#1a252f")

    plt.plot(x_grid, y_simples, color="#c0392b", linewidth=2.5, linestyle="-", label=f"Reta MQO Simples: $\\hat{{Y}} = {b0_s:.2f} {b1_s:+.2f} \\cdot \\ln(\\text{{PIB}})$", zorder=3)
    plt.title("Regressão Simples: Taxa de Homicídios vs. Log do PIB per capita (2014)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Logaritmo Natural do PIB Estadual per capita [ln_pib_pc]", fontsize=11, fontweight="bold")
    plt.ylabel("Taxa de Homicídios por 100 mil habitantes", fontsize=11, fontweight="bold")
    plt.legend(frameon=True, facecolor="white", edgecolor="#bdc3c7", loc="upper right", fontsize=10)
    plt.tight_layout()
    caminho_simples = os.path.join(DIR_IMAGENS, "grafico_1_regressao_simples.png")
    plt.savefig(caminho_simples, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"-> Gráfico salvo: '{caminho_simples}'")

    # 4. Gráfico 2: Regressão Múltipla com 3 Variáveis Explicativas e 1 Dependente
    # Y = taxa_homicidios | X1 = ln_pib_pc | X2 = gini | X3 = desemprego
    m1 = modelos["m1"]
    b0_mult = m1.params["const"]
    b1_mult = m1.params["ln_pib_pc"]
    b2_mult = m1.params["gini"]
    b3_mult = m1.params["desemprego"]

    fig, ax = plt.subplots(figsize=(9, 6.5))

    gini_medio = df["gini"].mean()
    desemp_medio = df["desemprego"].mean()
    intercepto_ajustado = b0_mult + b2_mult * gini_medio + b3_mult * desemp_medio
    y_mult_grid = intercepto_ajustado + b1_mult * x_grid

    ax.scatter(df["ln_pib_pc"], df["taxa_homicidios"], color="#2c3e50", s=65, alpha=0.85, edgecolors="white", linewidth=1.2, label="Observações Reais (27 UFs)", zorder=4)
    for _, r in df[df["uf"].isin(["SP", "SC", "AL", "SE", "DF", "RJ"])].iterrows():
        ax.annotate(r["uf"], (r["ln_pib_pc"], r["taxa_homicidios"]), textcoords="offset points", xytext=(5, 4), fontsize=9, fontweight="bold", color="#34495e")

    # Simples: Contínua Vermelha
    ax.plot(x_grid, y_simples, color="#c0392b", linestyle="-", linewidth=2.5, label=f"Modelo Simples: $\\tilde{{\\beta}}_1$ = {b1_s:.2f} (Enviesado)", zorder=3)
    # Múltiplo: Tracejada Verde
    ax.plot(x_grid, y_mult_grid, color="#27ae60", linestyle="--", linewidth=2.5, label=f"Modelo Múltiplo (médias fixas): $\\hat{{\\beta}}_1$ = {b1_mult:.2f}", zorder=3)
    ax.set_title("Regressão Múltipla com 3 Variáveis Explicativas vs. Modelo Simples\nEfeito do PIB per capita Controlando Simultaneamente por Gini e Desemprego", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Logaritmo do PIB per capita [ln_pib_pc]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Taxa de Homicídios por 100k hab. (Y)", fontsize=11, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#bdc3c7", loc="upper right", fontsize=10)

    plt.tight_layout()
    caminho_multiplo = os.path.join(DIR_IMAGENS, "grafico_2_regressao_multipla.png")
    plt.savefig(caminho_multiplo, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"-> Gráfico salvo: '{caminho_multiplo}'")


# ==============================================================================
# 3. MODELAGEM ECONOMÉTRICA VIA MQO (OLS)
# ==============================================================================
def estimar_modelos(df: pd.DataFrame) -> dict:
    """Estima Modelo 1 (Completo), Modelo 2 (Benchmark), Modelo 3 (Simples) e Auxiliar."""
    y = df["taxa_homicidios"]
    m1 = sm.OLS(y, sm.add_constant(df[["ln_pib_pc", "gini", "desemprego"]])).fit()
    m2 = sm.OLS(y, sm.add_constant(df[["ln_pib_pc", "gini"]])).fit()
    m3 = sm.OLS(y, sm.add_constant(df[["ln_pib_pc"]])).fit()
    m_aux = sm.OLS(df["gini"], sm.add_constant(df[["ln_pib_pc"]])).fit()

    return {"m1": m1, "m2": m2, "m3": m3, "aux": m_aux}


def demonstrar_ovb(modelos: dict) -> dict:
    """Verifica e exibe a identidade analítica de Wooldridge."""
    m2, m3, m_aux = modelos["m2"], modelos["m3"], modelos["aux"]
    b1_til = m3.params["ln_pib_pc"]
    b1_chapeu = m2.params["ln_pib_pc"]
    b2_chapeu = m2.params["gini"]
    delta_1 = m_aux.params["ln_pib_pc"]
    vies = b2_chapeu * delta_1
    diferenca = abs(b1_til - (b1_chapeu + vies))

    print("\n" + "=" * 75)
    print("ETAPA 4: DEMONSTRAÇÃO FORMAL DO VIÉS DE VARIÁVEL OMITIDA (WOOLDRIDGE)")
    print("=" * 75)
    print(f"Beta1 Simples (Enviesado)  : {b1_til:12.6f}")
    print(f"Beta1 Múltiplo (Controlado): {b1_chapeu:12.6f}")
    print(f"Beta2 Gini (Múltiplo)      : {b2_chapeu:12.6f}")
    print(f"delta_1 (Regressão Auxiliar): {delta_1:12.6f}")
    print(f"Viés Calculado (Beta2*delta): {vies:12.6f}")
    print(f"Recomposição (Beta1_M + Viés): {b1_chapeu + vies:12.6f}")
    print(f"Discrepância Numérica       : {diferenca:12.4e}")
    assert diferenca < 1e-10, "Erro na comprovação de Wooldridge!"
    print("-> SUCESSO: Identidade de Wooldridge satisfeita com precisão de máquina!")

    return {
        "b1_til": b1_til, "b1_chapeu": b1_chapeu, "b2_chapeu": b2_chapeu,
        "delta_1": delta_1, "vies": vies, "diferenca": diferenca
    }


# ==============================================================================
# 4. GERAÇÃO DO SIMULADOR VISUAL INTERATIVO (HTML / JS / CANVAS)
# ==============================================================================
def gerar_dashboard_interativo(df: pd.DataFrame, modelos: dict, ovb_info: dict):
    """
    Gera um dashboard visual interativo completo em HTML standalone
    com sliders dinâmicos para simulação em tempo real de Gini, Desemprego e PIB.
    A taxa de desemprego desloca em tempo real o Modelo Completo e atualiza métricas.
    """
    print("\n" + "=" * 75)
    print("ETAPA 5: GERAÇÃO DO SIMULADOR VISUAL INTERATIVO")
    print("=" * 75)

    m1, m2, m3 = modelos["m1"], modelos["m2"], modelos["m3"]

    estados_json = json.dumps([
        {
            "uf": row["uf"],
            "pib_pc": round(row["pib_per_capita"], 2),
            "ln_pib_pc": round(row["ln_pib_pc"], 4),
            "taxa_homicidios": round(row["taxa_homicidios"], 2),
            "gini": round(row["gini"], 3),
            "desemprego": round(row["desemprego"], 2)
        }
        for _, row in df.iterrows()
    ], ensure_ascii=False)

    html_code = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simulador Econométrico Interativo: Viés de Variável Omitida</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.75);
            --border: rgba(255, 255, 255, 0.1);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #38bdf8;
            --red: #ef4444;
            --green: #10b981;
            --blue: #3b82f6;
            --yellow: #f59e0b;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', sans-serif;
            background: radial-gradient(circle at top, #1e293b 0%, #0f172a 100%);
            color: var(--text-main);
            min-height: 100vh;
            padding: 24px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        header {{
            margin-bottom: 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 16px;
        }}
        h1 {{
            font-size: 1.55rem;
            font-weight: 700;
            background: linear-gradient(90deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .badge {{
            background: rgba(56, 189, 248, 0.15);
            color: #38bdf8;
            padding: 5px 14px;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            border: 1px solid rgba(56, 189, 248, 0.3);
        }}
        .grid {{
            display: grid;
            grid-template-columns: 400px 1fr;
            gap: 24px;
        }}
        @media (max-width: 1080px) {{ .grid {{ grid-template-columns: 1fr; }} }}
        .card {{
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        }}
        .section-title {{
            font-size: 0.95rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--accent);
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .control-group {{ margin-bottom: 18px; }}
        .control-label {{
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            font-weight: 500;
            margin-bottom: 6px;
        }}
        .control-val {{
            font-family: 'Fira Code', monospace;
            font-weight: 600;
            color: var(--accent);
        }}
        input[type="range"] {{
            width: 100%;
            height: 6px;
            border-radius: 3px;
            background: #334155;
            outline: none;
            cursor: pointer;
            accent-color: var(--accent);
        }}
        .toggles {{
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-top: 14px;
        }}
        .toggle-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 0.85rem;
            cursor: pointer;
            user-select: none;
        }}
        .toggle-item input {{ cursor: pointer; }}
        .metric-cards {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-bottom: 20px;
        }}
        @media (max-width: 900px) {{ .metric-cards {{ grid-template-columns: repeat(2, 1fr); }} }}
        .metric-card {{
            background: rgba(15, 23, 42, 0.65);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 12px 14px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }}
        .metric-title {{
            font-size: 0.72rem;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 4px;
            font-weight: 600;
        }}
        .metric-value {{
            font-family: 'Fira Code', monospace;
            font-size: 1.35rem;
            font-weight: 700;
        }}
        .metric-sub {{
            font-size: 0.68rem;
            color: var(--text-muted);
            margin-top: 4px;
        }}
        .canvas-container {{
            position: relative;
            width: 100%;
            height: 480px;
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
        }}
        canvas {{ width: 100%; height: 100%; display: block; }}
        .legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            margin-top: 14px;
            font-size: 0.8rem;
            padding: 10px 14px;
            background: rgba(15, 23, 42, 0.5);
            border-radius: 8px;
            border: 1px solid var(--border);
        }}
        .legend-item {{ display: flex; align-items: center; gap: 7px; }}
        .legend-color {{ width: 20px; height: 4px; border-radius: 2px; }}
        .wooldridge-box {{
            background: rgba(15, 23, 42, 0.75);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 14px;
            margin-top: 18px;
            font-family: 'Fira Code', monospace;
            font-size: 0.8rem;
            line-height: 1.5;
        }}
        .notice-banner {{
            margin-top: 14px;
            padding: 10px 12px;
            background: rgba(59, 130, 246, 0.12);
            border-left: 3px solid var(--blue);
            border-radius: 4px;
            font-size: 0.78rem;
            line-height: 1.4;
            color: #93c5fd;
        }}
        .tooltip {{
            position: absolute;
            background: rgba(15, 23, 42, 0.95);
            border: 1px solid var(--accent);
            color: #fff;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.75rem;
            pointer-events: none;
            display: none;
            z-index: 100;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>Laboratório Econométrico Interativo: Viés de Variável Omitida</h1>
                <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 4px;">
                    Simulação dinâmica com dados estaduais do Brasil (Ipeadata 2014)
                </p>
            </div>
            <div class="badge">N = 27 UFs | MQO / OLS</div>
        </header>

        <div class="grid">
            <!-- PAINEL DE CONTROLES -->
            <div class="card">
                <div class="section-title">Controles das Variáveis Explicativas</div>

                <!-- 1. GINI -->
                <div class="control-group">
                    <div class="control-label">
                        <span>Índice de Gini (Desigualdade)</span>
                        <span class="control-val" id="val-gini">0.505</span>
                    </div>
                    <input type="range" id="slider-gini" min="0.380" max="0.620" step="0.005" value="0.5045">
                    <div style="display:flex; justify-content:space-between; font-size:0.7rem; color:var(--text-muted); margin-top:2px;">
                        <span>0.38 (Menos desigual)</span>
                        <span>0.50 (Média)</span>
                        <span>0.62 (Mais desigual)</span>
                    </div>
                </div>

                <!-- 2. DESEMPREGO (COM IMPACTO DIRETO VISÍVEL) -->
                <div class="control-group">
                    <div class="control-label">
                        <span>Taxa de Desemprego (%)</span>
                        <span class="control-val" id="val-desemp" style="color: var(--blue);">6.68%</span>
                    </div>
                    <input type="range" id="slider-desemp" min="2.0" max="14.0" step="0.1" value="6.68">
                    <div style="display:flex; justify-content:space-between; font-size:0.7rem; color:var(--text-muted); margin-top:2px;">
                        <span>2.0% (Mín: SC)</span>
                        <span>6.7% (Média)</span>
                        <span>14.0% (Crise)</span>
                    </div>
                </div>

                <!-- 3. PIB PER CAPITA -->
                <div class="control-group">
                    <div class="control-label">
                        <span>PIB Estadual per capita (R$ mil)</span>
                        <span class="control-val" id="val-pib">R$ 17.84k</span>
                    </div>
                    <input type="range" id="slider-pib" min="8.0" max="52.0" step="0.5" value="17.84">
                    <div style="display:flex; justify-content:space-between; font-size:0.7rem; color:var(--text-muted); margin-top:2px;">
                        <span>R$ 8.2k (MA)</span>
                        <span>R$ 17.8k (Média)</span>
                        <span>R$ 51.1k (DF)</span>
                    </div>
                </div>

                <div class="section-title" style="margin-top: 22px;">Camadas Visuais do Gráfico</div>
                <div class="toggles">
                    <label class="toggle-item">
                        <input type="checkbox" id="chk-simples" checked>
                        <span>Modelo 3: Simples (Linha Vermelha Contínua)</span>
                    </label>
                    <label class="toggle-item">
                        <input type="checkbox" id="chk-multiplo" checked>
                        <span>Modelo 2: Benchmark c/ Gini (Linha Verde Tracejada)</span>
                    </label>
                    <label class="toggle-item">
                        <input type="checkbox" id="chk-completo" checked>
                        <span>Modelo 1: Completo c/ Desemprego (Linha Azul Pontilhada)</span>
                    </label>
                </div>

                <div class="notice-banner">
                    <strong>Por que o Desemprego desloca o Modelo 1 (Linha Azul)?</strong><br>
                    O coeficiente estimado é <strong>β̂₃ = +2.6029</strong>. Cada ponto percentual a mais de desemprego eleva a previsão em 2,6 homicídios por 100k hab.
                    Contudo, sua inclusão gera multicolinearidade com o Gini (r = +0.767) e tem p = 0.120 (não significante).
                </div>

                <div class="wooldridge-box">
                    <div style="color:var(--accent); font-weight:600; margin-bottom:4px;">IDENTIDADE DE WOOLDRIDGE:</div>
                    β̃₁ = β̂₁ + (β̂₂ × δ̃₁)<br>
                    {ovb_info['b1_til']:.4f} = {ovb_info['b1_chapeu']:.4f} + ({ovb_info['b2_chapeu']:.4f} × {ovb_info['delta_1']:.4f})<br>
                    Viés = <span style="color:var(--red); font-weight:bold;">{ovb_info['vies']:.4f}</span> (Superestimação do PIB)
                </div>
            </div>

            <!-- PAINEL DE VISUALIZAÇÃO E PREVISÕES -->
            <div class="card">
                <div class="metric-cards">
                    <!-- CARD 1 -->
                    <div class="metric-card" style="border-top: 3px solid var(--red);">
                        <div class="metric-title">Mod. 3: Simples</div>
                        <div class="metric-value" style="color: var(--red);" id="card-simples">--</div>
                        <div class="metric-sub">Ỹ = 69.37 - 12.25 · ln(PIB)</div>
                    </div>
                    <!-- CARD 2 -->
                    <div class="metric-card" style="border-top: 3px solid var(--green);">
                        <div class="metric-title">Mod. 2: Benchmark</div>
                        <div class="metric-value" style="color: var(--green);" id="card-multiplo">--</div>
                        <div class="metric-sub">Controlado pelo Gini</div>
                    </div>
                    <!-- CARD 3 -->
                    <div class="metric-card" style="border-top: 3px solid var(--blue);">
                        <div class="metric-title">Mod. 1: Completo</div>
                        <div class="metric-value" style="color: var(--blue);" id="card-completo">--</div>
                        <div class="metric-sub">Gini + Desemprego</div>
                    </div>
                    <!-- CARD 4 -->
                    <div class="metric-card" style="border-top: 3px solid var(--accent);">
                        <div class="metric-title">Impacto Desemprego</div>
                        <div class="metric-value" style="color: var(--accent);" id="card-impacto-des">--</div>
                        <div class="metric-sub">vs. Média Amostral (6.7%)</div>
                    </div>
                </div>

                <div class="canvas-container" id="container-plot">
                    <canvas id="plot"></canvas>
                    <div class="tooltip" id="tooltip"></div>
                </div>

                <div class="legend">
                    <div class="legend-item">
                        <div style="width:10px; height:10px; border-radius:50%; background:#2c3e50; border:1px solid #fff;"></div>
                        <span>27 UFs (Reais)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background:var(--red);"></div>
                        <span>Mod. 3 Simples (Linha Contínua)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background:var(--green); border-top: 2px dashed var(--green); height:0;"></div>
                        <span>Mod. 2 Benchmark (Tracejada Verde - Gini)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background:var(--blue); border-top: 2px dotted var(--blue); height:0;"></div>
                        <span>Mod. 1 Completo (Pontilhada Azul - Desemprego)</span>
                    </div>
                    <div class="legend-item">
                        <div style="width:12px; height:12px; border-radius:50%; background:var(--yellow); border:2px solid #fff;"></div>
                        <span>Previsão Mod. 2</span>
                    </div>
                    <div class="legend-item">
                        <div style="width:12px; height:12px; border-radius:50%; background:var(--blue); border:2px solid #fff;"></div>
                        <span>Previsão Mod. 1</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Coeficientes dos Modelos MQO
        const b0_simp = {m3.params['const']:.6f};
        const b1_simp = {m3.params['ln_pib_pc']:.6f};

        const b0_mult = {m2.params['const']:.6f};
        const b1_mult = {m2.params['ln_pib_pc']:.6f};
        const b2_gini = {m2.params['gini']:.6f};

        const b0_comp = {m1.params['const']:.6f};
        const b1_comp = {m1.params['ln_pib_pc']:.6f};
        const b2_comp_gini = {m1.params['gini']:.6f};
        const b3_comp_des = {m1.params['desemprego']:.6f};
        const des_medio = 6.68148;

        const dadosEstados = {estados_json};

        // Elementos DOM
        const sliderGini = document.getElementById('slider-gini');
        const sliderDesemp = document.getElementById('slider-desemp');
        const sliderPib = document.getElementById('slider-pib');
        const valGini = document.getElementById('val-gini');
        const valDesemp = document.getElementById('val-desemp');
        const valPib = document.getElementById('val-pib');

        const cardSimp = document.getElementById('card-simples');
        const cardMult = document.getElementById('card-multiplo');
        const cardComp = document.getElementById('card-completo');
        const cardImpactoDes = document.getElementById('card-impacto-des');

        const chkSimples = document.getElementById('chk-simples');
        const chkMultiplo = document.getElementById('chk-multiplo');
        const chkCompleto = document.getElementById('chk-completo');

        const canvas = document.getElementById('plot');
        const ctx = canvas.getContext('2d');
        const tooltip = document.getElementById('tooltip');

        function resizeCanvas() {{
            const rect = canvas.parentElement.getBoundingClientRect();
            canvas.width = rect.width * window.devicePixelRatio;
            canvas.height = rect.height * window.devicePixelRatio;
            ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
            draw();
        }}
        window.addEventListener('resize', resizeCanvas);

        function calc() {{
            const gini = parseFloat(sliderGini.value);
            const des = parseFloat(sliderDesemp.value);
            const pib = parseFloat(sliderPib.value);
            const lnPib = Math.log(pib);

            valGini.innerText = gini.toFixed(3);
            valDesemp.innerText = des.toFixed(1) + '%';
            valPib.innerText = 'R$ ' + pib.toFixed(2) + 'k';

            // 1. Modelo Simples
            const predSimp = b0_simp + b1_simp * lnPib;
            // 2. Modelo Benchmark (Gini)
            const predMult = b0_mult + b1_mult * lnPib + b2_gini * gini;
            // 3. Modelo Completo (Gini + Desemprego)
            const predComp = b0_comp + b1_comp * lnPib + b2_comp_gini * gini + b3_comp_des * des;
            // 4. Impacto do desemprego em relação à média
            const impactoDes = b3_comp_des * (des - des_medio);

            cardSimp.innerText = predSimp.toFixed(1);
            cardMult.innerText = predMult.toFixed(1);
            cardComp.innerText = predComp.toFixed(1);
            cardImpactoDes.innerText = (impactoDes >= 0 ? '+' : '') + impactoDes.toFixed(2);

            draw();
        }}

        function draw() {{
            const rect = canvas.getBoundingClientRect();
            const w = rect.width;
            const h = rect.height;

            ctx.clearRect(0, 0, w, h);

            const padLeft = 60, padRight = 30, padTop = 30, padBottom = 50;
            const plotW = w - padLeft - padRight;
            const plotH = h - padTop - padBottom;

            const minX = 2.0, maxX = 4.1;
            const minY = 10, maxY = 70;

            const toX = x => padLeft + ((x - minX) / (maxX - minX)) * plotW;
            const toY = y => padTop + plotH - ((y - minY) / (maxY - minY)) * plotH;

            // Grade e Eixos
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
            ctx.lineWidth = 1;
            ctx.fillStyle = '#64748b';
            ctx.font = '11px Inter, sans-serif';

            for (let yVal = 10; yVal <= 70; yVal += 10) {{
                const py = toY(yVal);
                ctx.beginPath();
                ctx.moveTo(padLeft, py);
                ctx.lineTo(w - padRight, py);
                ctx.stroke();
                ctx.fillText(yVal, padLeft - 28, py + 4);
            }}

            for (let xVal = 2.0; xVal <= 4.0; xVal += 0.5) {{
                const px = toX(xVal);
                ctx.beginPath();
                ctx.moveTo(px, padTop);
                ctx.lineTo(px, h - padBottom);
                ctx.stroke();
                ctx.fillText(xVal.toFixed(1), px - 10, h - padBottom + 20);
            }}

            ctx.fillStyle = '#94a3b8';
            ctx.font = 'bold 11px Inter, sans-serif';
            ctx.fillText('Logaritmo do PIB per capita [ln_pib_pc]', padLeft + plotW / 2 - 100, h - 15);
            ctx.save();
            ctx.translate(16, padTop + plotH / 2 + 60);
            ctx.rotate(-Math.PI / 2);
            ctx.fillText('Taxa de Homicídios por 100k hab.', 0, 0);
            ctx.restore();

            const curGini = parseFloat(sliderGini.value);
            const curDes = parseFloat(sliderDesemp.value);
            const curPib = parseFloat(sliderPib.value);
            const curLnPib = Math.log(curPib);

            // 1. RETA SIMPLES (Contínua Vermelha)
            if (chkSimples.checked) {{
                ctx.strokeStyle = '#ef4444';
                ctx.lineWidth = 2.5;
                ctx.setLineDash([]);
                ctx.beginPath();
                for (let px = minX; px <= maxX; px += 0.05) {{
                    const py = b0_simp + b1_simp * px;
                    const cx = toX(px), cy = toY(py);
                    if (px === minX) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
                }}
                ctx.stroke();
            }}

            // 2. RETA MÚLTIPLA BENCHMARK (Tracejada Verde - reage ao Gini)
            if (chkMultiplo.checked) {{
                ctx.strokeStyle = '#10b981';
                ctx.lineWidth = 2.5;
                ctx.setLineDash([8, 6]);
                ctx.beginPath();
                for (let px = minX; px <= maxX; px += 0.05) {{
                    const py = (b0_mult + b2_gini * curGini) + b1_mult * px;
                    const cx = toX(px), cy = toY(py);
                    if (px === minX) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
                }}
                ctx.stroke();
            }}

            // 3. RETA COMPLETA (Pontilhada Azul - reage ao Gini E AO DESEMPREGO EM TEMPO REAL)
            if (chkCompleto.checked) {{
                ctx.strokeStyle = '#3b82f6';
                ctx.lineWidth = 2.5;
                ctx.setLineDash([3, 4]);
                ctx.beginPath();
                for (let px = minX; px <= maxX; px += 0.05) {{
                    const py = (b0_comp + b2_comp_gini * curGini + b3_comp_des * curDes) + b1_comp * px;
                    const cx = toX(px), cy = toY(py);
                    if (px === minX) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
                }}
                ctx.stroke();
            }}
            ctx.setLineDash([]);

            // PONTOS DOS ESTADOS (Observados)
            dadosEstados.forEach(est => {{
                const cx = toX(est.ln_pib_pc);
                const cy = toY(est.taxa_homicidios);

                ctx.fillStyle = '#334155';
                ctx.beginPath();
                ctx.arc(cx, cy, 6, 0, Math.PI * 2);
                ctx.fill();
                ctx.strokeStyle = '#94a3b8';
                ctx.lineWidth = 1.2;
                ctx.stroke();

                if (['SP', 'SC', 'AL', 'SE', 'DF', 'RJ'].includes(est.uf)) {{
                    ctx.fillStyle = '#cbd5e1';
                    ctx.font = 'bold 10px Inter, sans-serif';
                    ctx.fillText(est.uf, cx + 8, cy + 3);
                }}
            }});

            // PONTOS SIMULADOS
            const simX = toX(curLnPib);

            // Ponto Modelo 2 (Gini) - Amarelo
            const predY2 = (b0_mult + b2_gini * curGini) + b1_mult * curLnPib;
            const simY2 = toY(predY2);

            // Ponto Modelo 1 (Completo c/ Desemprego) - Azul
            const predY1 = (b0_comp + b2_comp_gini * curGini + b3_comp_des * curDes) + b1_comp * curLnPib;
            const simY1 = toY(predY1);

            // Linha conectando as duas previsões
            if (chkMultiplo.checked && chkCompleto.checked) {{
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
                ctx.lineWidth = 1.5;
                ctx.setLineDash([2, 3]);
                ctx.beginPath();
                ctx.moveTo(simX, simY2);
                ctx.lineTo(simX, simY1);
                ctx.stroke();
                ctx.setLineDash([]);
            }}

            if (chkMultiplo.checked) {{
                ctx.fillStyle = '#f59e0b';
                ctx.beginPath();
                ctx.arc(simX, simY2, 7, 0, Math.PI * 2);
                ctx.fill();
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 2;
                ctx.stroke();
            }}

            if (chkCompleto.checked) {{
                ctx.fillStyle = '#3b82f6';
                ctx.beginPath();
                ctx.arc(simX, simY1, 6, 0, Math.PI * 2);
                ctx.fill();
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 2;
                ctx.stroke();
            }}
        }}

        // Interação Hover
        canvas.addEventListener('mousemove', e => {{
            const rect = canvas.getBoundingClientRect();
            const mx = e.clientX - rect.left;
            const my = e.clientY - rect.top;

            const padLeft = 60, padRight = 30, padTop = 30, padBottom = 50;
            const plotW = rect.width - padLeft - padRight;
            const plotH = rect.height - padTop - padBottom;
            const minX = 2.0, maxX = 4.1, minY = 10, maxY = 70;
            const toX = x => padLeft + ((x - minX) / (maxX - minX)) * plotW;
            const toY = y => padTop + plotH - ((y - minY) / (maxY - minY)) * plotH;

            let hovered = null;
            dadosEstados.forEach(est => {{
                const cx = toX(est.ln_pib_pc);
                const cy = toY(est.taxa_homicidios);
                const dist = Math.hypot(mx - cx, my - cy);
                if (dist < 8) hovered = est;
            }});

            if (hovered) {{
                tooltip.style.display = 'block';
                tooltip.style.left = (mx + 15) + 'px';
                tooltip.style.top = (my - 15) + 'px';
                tooltip.innerHTML = `<strong>${{hovered.uf}}</strong><br>` +
                    `Homicídios: ${{hovered.taxa_homicidios}}<br>` +
                    `PIB pc: R$ ${{hovered.pib_pc}}k (ln: ${{hovered.ln_pib_pc}})<br>` +
                    `Gini: ${{hovered.gini}}<br>` +
                    `Desemprego: ${{hovered.desemprego}}%`;
            }} else {{
                tooltip.style.display = 'none';
            }}
        }});

        [sliderGini, sliderDesemp, sliderPib].forEach(s => s.addEventListener('input', calc));
        [chkSimples, chkMultiplo, chkCompleto].forEach(c => c.addEventListener('change', draw));

        setTimeout(resizeCanvas, 50);
        calc();
    </script>
</body>
</html>
"""

    caminho_html = "dashboard_interativo.html"
    with open(caminho_html, "w", encoding="utf-8") as f:
        f.write(html_code)

    print(f"-> Simulador visual interativo gerado com sucesso: '{caminho_html}'")
    return caminho_html


# ==============================================================================
# 5. EXPORTAÇÃO DE RELATÓRIOS E PLANO DE APRESENTAÇÃO
# ==============================================================================
def exportar_relatorios(modelos: dict, ovb_info: dict):
    """Exporta sumários em formato TXT e plano de apresentação completo em Markdown."""
    print("\n" + "=" * 75)
    print("ETAPA 6: EXPORTAÇÃO DOS RELATÓRIOS E PLANO DE APRESENTAÇÃO")
    print("=" * 75)

    m1, m2, m3, m_aux = modelos["m1"], modelos["m2"], modelos["m3"], modelos["aux"]

    # 1. TXT com sumários
    caminho_txt = os.path.join(DIR_RELATORIOS, "resultados_tabelas.txt")
    with open(caminho_txt, "w", encoding="utf-8") as f:
        f.write("RELATÓRIO ECONOMÉTRICO COMPLETO - REGRESSÕES MQO (OLS)\n")
        f.write("Dados: 27 UFs Brasil (2014) - Ipeadata / PNAD Contínua\n")
        f.write("=" * 80 + "\n\n")
        f.write(">>> MODELO 1: MÚLTIPLO COMPLETO\n" + str(m1.summary()) + "\n\n")
        f.write(">>> MODELO 2: MÚLTIPLO BENCHMARK\n" + str(m2.summary()) + "\n\n")
        f.write(">>> MODELO 3: SIMPLES ENVIESADO\n" + str(m3.summary()) + "\n\n")
        f.write(">>> REGRESSÃO AUXILIAR: GINI ~ LN_PIB\n" + str(m_aux.summary()) + "\n\n")
        f.write("CONFERÊNCIA DA IDENTIDADE DE WOOLDRIDGE:\n")
        f.write(f"Beta1_Simples : {ovb_info['b1_til']:.6f}\n")
        f.write(f"Beta1_Múltiplo: {ovb_info['b1_chapeu']:.6f}\n")
        f.write(f"Beta2_Gini    : {ovb_info['b2_chapeu']:.6f}\n")
        f.write(f"delta_1       : {ovb_info['delta_1']:.6f}\n")
        f.write(f"Viés          : {ovb_info['vies']:.6f}\n")
        f.write(f"Discrepância  : {ovb_info['diferenca']:.4e}\n")

    print(f"-> Sumários exportados em '{caminho_txt}'.")

    # 2. Resumo Executivo e Plano de Apresentação
    caminho_md = os.path.join(DIR_RELATORIOS, "resumo_apresentacao.md")
    template_md = r"""# Plano de Apresentação do Trabalho & Guia de Arguição

**Tema:** Modelagem Econométrica da Taxa de Homicídios e Demonstração do Viés de Variável Omitida (OVB)  
**Amostra:** 27 Unidades da Federação do Brasil (2014) | Fonte: Ipeadata / IBGE (PNAD Contínua)  

---

## 1. Tabela Comparativa Geral dos Modelos (MQO / OLS)

| Métrica / Variável | Modelo 1 (Completo) | Modelo 2 (Benchmark) | Modelo 3 (Simples Enviesado) |
| :--- | :---: | :---: | :---: |
| **Constante ($\beta_0$)** | +56.2025 (34.9089) | +30.7680 (32.1712) | +69.3655*** (12.8158) |
| **ln(PIB per capita) ($X_1$)** | -8.0283 (4.8401) | **-11.1903\*\*** (4.5711) | **-12.2465\*\*** (4.5616) |
| **Índice de Gini ($X_2$)** | -31.5743 (82.2824) | **+70.7016** (54.1915) | — |
| **Taxa de Desemprego ($X_3$)** | +2.6029 (1.6131) | — | — |
| **Observações ($N$)** | 27 | 27 | 27 |
| **$R^2$** | __R2_1__ | __R2_2__ | __R2_3__ |
| **$R^2$ Ajustado** | __R2_ADJ_1__ | __R2_ADJ_2__ | __R2_ADJ_3__ |
| **Estatística F** | __F_1__ | __F_2__ | __F_3__ |

*Nota: Erros-padrão entre parênteses. Níveis de significância: \* p < 0.10, \*\* p < 0.05, \*\*\* p < 0.01.*

---

## 2. Roteiro Passo a Passo para Apresentação em Aula (10 Minutos)

### Bloco 1: Introdução e Motivação Empírica (2 min)
- **Problema:** Estados mais ricos são automaticamente mais seguros? Ou o crescimento econômico desacompanhado de igualdade social falha em conter a violência?
- **Base de Dados:** 27 UFs no ano de 2014. 
- **Estatística Descritiva Chave (Tabela 1):**
  - Média de homicídios no Brasil em 2014: **35.4 por 100 mil hab.** (extremos: SC com 13.4 e AL com 65.9).
  - PIB per capita varia mais de 600%: de R$ 8.274 (MA) a R$ 51.060 (DF).
  - Correlação bivariada inicial entre ln(PIB pc) e Homicídios: **r = -0.473**.

### Bloco 2: A Armadilha do Modelo Simples (2 min)
- **Especificação:** $\text{Homicídios} = \beta_0 + \beta_1 \ln(\text{PIB\_pc}) + u$.
- **Resultado:** $\widetilde{\beta}_1 = -12.25$ ($p = 0.013^{**}$).
- **Interpretação Econômica (Semi-elasticidade):** Um aumento de 10% no PIB per capita parece reduzir a taxa em 1,22 homicídios por 100k hab.
- **Pergunta provocativa para a banca:** *Esse efeito pertence de fato à renda, ou a renda está levando o crédito de outra variável omitida correlacionada?*

### Bloco 3: Demonstração Formal do Viés de Wooldridge (3 min)
- **Equação Analítica:**
  $$\widetilde{\beta}_1 = \widehat{\beta}_1 + \widehat{\beta}_2 \cdot \widetilde{\delta}_1$$
- **Regressão Auxiliar ($\text{Gini} \sim \ln(\text{PIB})$):** $\widetilde{\delta}_1 = -0.0149$ (estados mais ricos têm menor desigualdade).
- **Modelo Múltiplo:** $\widehat{\beta}_1 = -11.1903$, $\widehat{\beta}_2 = +70.7016$ (maior Gini amplia o crime).
- **Cálculo do Viés:**
  $$\text{Viés} = (+70.7016) \times (-0.0149) = \mathbf{-1.0563}$$
  $$\widetilde{\beta}_1 = -11.1903 + (-1.0563) = \mathbf{-12.2465}$$
- **Conclusão:** O Modelo Simples **superestima** a eficácia do PIB em módulo ($-12.25$ vs $-11.19$). Parte da redução de homicídios atribuída ao PIB decorre, na verdade, da menor desigualdade dos estados ricos.

### Bloco 4: O Dilema do Desemprego e Multicolinearidade (1.5 min)
- Ao incluir o desemprego (Modelo 1): $\widehat{\beta}_3 = +2.6029$, porém **não atinge significância estatística** ($p = 0.1202$).
- **Por que não usar o Modelo 1 como benchmark?**
  1. Alta colinearidade entre Desemprego e Gini ($r = +0.767$).
  2. O erro-padrão do Gini inflaciona em 51,8% (de 54.19 para 82.28).
  3. $\ln(\text{PIB})$ perde significância individual ($p = 0.111$).
  4. O Modelo 2 é o modelo econométrico parcimonioso ideal.

### Bloco 5: Demonstração Prática no Simulador Visual (1.5 min)
- Abrir `dashboard_interativo.html` no projetor.
- **Mexer no Gini:** Mostrar a linha verde tracejada subindo com a desigualdade.
- **Mexer no Desemprego:** Mostrar a linha azul pontilhada reagindo à desocupação em tempo real.
- **Ajustar o PIB:** Mostrar o ponto simulado amarelo e a leitura dos 4 cards de previsão.

---

## 3. Respostas Prontas para Perguntas Difíceis da Banca

1. **"Por que utilizar ln(PIB per capita) em vez do PIB em nível?"**
   - *Resposta:* A especificação nível-log modela retornos decrescentes (o impacto de R$ 1.000 a mais em um estado pobre como o Maranhão é muito maior do que no Distrito Federal) e suaviza a assimetria da distribuição de renda, garantindo resíduos bem comportados.

2. **"O que comprova que o Modelo Simples é enviesado?"**
   - *Resposta:* A identidade de Wooldridge fecha numericamente com precisão de máquina ($10^{-15}$). O viés analítico é exatamente $\widehat{\beta}_2 \cdot \widetilde{\delta}_1 = -1.0563$.

3. **"Por que manter o Modelo 2 se o Modelo 1 tem R² maior (0.3489 vs 0.2752)?"**
   - *Resposta:* O $R^2$ ordinário sempre sobe com a inclusão de regressores. Contudo, o desemprego é estatisticamente insignificante ($p = 0.120$) e gera severa multicolinearidade ($r = +0.767$ com Gini), inflando a variância dos estimadores. Pelo princípio da parcimônia e pelo critério de identificação causal, o Modelo 2 é superior.

---

## 4. Estrutura Oficial dos Gráficos Gerados (Sem Duplicações)

Para a apresentação de slides e relatório impresso, o pipeline exporta exatamente **4 figuras de alta resolução (300 DPI)** na pasta `imagens/`:

1. **`grafico_1_regressao_simples.png`** (Regressão Simples):
   - Diagrama de dispersão das 27 UFs com a reta de MQO bivariada $\widetilde{Y} = 69.37 - 12.25 \ln(\text{PIB})$.
2. **`grafico_2_regressao_multipla.png`** (Regressão Múltipla com 3 Explicativas e 1 Dependente):
   - Comparação da reta ajustada do Modelo Múltiplo de 3 variáveis (avaliado nas médias amostrais de Gini e Desemprego) versus a reta do Modelo Simples (verde tracejada vs. vermelha contínua), isolando o efeito parcial ceteris paribus do PIB per capita.
3. **`grafico_dispersao_pares.png`** (Correlação das Variáveis - Dispersão):
   - Matriz completa de pairplot mostrando as relações bivariadas, dispersões e distribuições marginais (KDE).
4. **`grafico_correlacao_heatmap.png`** (Correlação das Variáveis - Matriz de Correlação):
   - Mapa de calor com os coeficientes de Pearson ($r$) entre as 4 variáveis.
"""
    md = template_md
    md = md.replace("__R2_1__", f"{m1.rsquared:.4f}")
    md = md.replace("__R2_2__", f"{m2.rsquared:.4f}")
    md = md.replace("__R2_3__", f"{m3.rsquared:.4f}")
    md = md.replace("__R2_ADJ_1__", f"{m1.rsquared_adj:.4f}")
    md = md.replace("__R2_ADJ_2__", f"{m2.rsquared_adj:.4f}")
    md = md.replace("__R2_ADJ_3__", f"{m3.rsquared_adj:.4f}")
    md = md.replace("__F_1__", f"{m1.fvalue:.4f} (p = {m1.f_pvalue:.4f})")
    md = md.replace("__F_2__", f"{m2.fvalue:.4f} (p = {m2.f_pvalue:.4f})")
    md = md.replace("__F_3__", f"{m3.fvalue:.4f} (p = {m3.f_pvalue:.4f})")
    with open(caminho_md, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"-> Plano de apresentação exportado em '{caminho_md}'.")


# ==============================================================================
# EXECUÇÃO DO PIPELINE
# ==============================================================================
def main():
    df = carregar_e_tratar_dados()
    calcular_descritivas(df)
    modelos = estimar_modelos(df)
    ovb_info = demonstrar_ovb(modelos)
    gerar_graficos_estaticos(df, modelos)
    gerar_dashboard_interativo(df, modelos, ovb_info)
    exportar_relatorios(modelos, ovb_info)

    print("\n" + "=" * 75)
    print("PIPELINE COMPLETO EXECUTADO COM SUCESSO!")
    print("Para interagir com o gráfico e os sliders, abra no navegador:")
    print("-> file://" + os.path.abspath("dashboard_interativo.html").replace("\\", "/"))
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
