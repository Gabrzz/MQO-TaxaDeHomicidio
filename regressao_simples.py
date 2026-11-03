"""
================================================================================
SCRIPT 1: REGRESSÃO SIMPLES (MODELO BIVARIADO INICIAL)
Tema: Taxa de Homicídios vs. Desenvolvimento Econômico (PIB per capita)
Disciplina: Econometria I
Amostra: 27 Unidades da Federação (Brasil, 2014)
================================================================================
"""


import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# 1. CARREGAMENTO DOS DADOS (PASTA DADOS)
# ------------------------------------------------------------------------------
caminho_dados = os.path.join("dados", "base_consolidada_2014.csv")
df = pd.read_csv(caminho_dados, sep=";", decimal=",")

print("=" * 70)
print(f"BASE DE DADOS CARREGADA COM SUCESSO: '{caminho_dados}' (27 UFs)")
print("=" * 70)
print(df[["uf", "taxa_homicidios", "pib_per_capita", "ln_pib_pc"]].head())

# ------------------------------------------------------------------------------
# 2. MODELAGEM ECONOMÉTRICA VIA MQO (OLS)
# Especificação: taxa_homicidios = beta_0 + beta_1 * ln_pib_pc + u
# ------------------------------------------------------------------------------
y = df["taxa_homicidios"]
X = sm.add_constant(df["ln_pib_pc"])  # Inclui o intercepto (beta_0)

modelo_simples = sm.OLS(y, X).fit()

# Exibição do sumário estatístico completo
print("\n" + "=" * 70)
print("RESULTADOS DO MODELO 3: REGRESSÃO SIMPLES")
print("=" * 70)
print(modelo_simples.summary())

# Coeficientes estimados
b0 = modelo_simples.params["const"]
b1 = modelo_simples.params["ln_pib_pc"]
r2 = modelo_simples.rsquared
p_val = modelo_simples.pvalues["ln_pib_pc"]

print("-" * 70)
print("INTERPRETAÇÃO ECONÔMICA (SEMI-ELASTICIDADE NÍVEL-LOG):")
print(f"-> beta_0 (Intercepto) : {b0:.4f}")
print(f"-> beta_1 (ln_pib_pc)  : {b1:.4f} (p-valor = {p_val:.4f})")
print(f"-> R²                  : {r2:.4f} ({r2*100:.1f}% da variação de Y explicada)")
print(
    f"Interpretação: Um aumento de 1% no PIB per capita estadual está associado a uma\n"
    f"redução média de aproximadamente {abs(b1)/100:.3f} homicídios por 100 mil habitantes."
)
print("-" * 70)

# ------------------------------------------------------------------------------
# 3. GERAÇÃO DO GRÁFICO DIDÁTICO DE DISPERSÃO E RETA MQO
# ------------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, ax = plt.subplots(figsize=(8.5, 6))

# Pontos reais dos estados
ax.scatter(
    df["ln_pib_pc"],
    df["taxa_homicidios"],
    color="#2c3e50",
    s=65,
    alpha=0.85,
    edgecolors="white",
    linewidth=1.2,
    label="Estados Brasileiros (27 UFs)",
    zorder=4
)

# Rótulos para estados extremos destacados
destaques = ["SP", "SC", "AL", "SE", "DF", "RJ"]
for _, row in df[df["uf"].isin(destaques)].iterrows():
    ax.annotate(
        row["uf"],
        (row["ln_pib_pc"], row["taxa_homicidios"]),
        textcoords="offset points",
        xytext=(6, 5),
        fontsize=9,
        fontweight="bold",
        color="#1a252f"
    )

# Reta de regressão ajustada
x_grid = np.linspace(df["ln_pib_pc"].min() - 0.05, df["ln_pib_pc"].max() + 0.05, 100)
y_grid = b0 + b1 * x_grid

ax.plot(
    x_grid,
    y_grid,
    color="#c0392b",
    linewidth=2.5,
    linestyle="-",
    label=f"Reta MQO Simples: $\\hat{{Y}} = {b0:.2f} {b1:+.2f} \\cdot \\ln(\\text{{PIB}})$",
    zorder=3
)

# Configurações visuais
ax.set_title("Regressão Simples: Taxa de Homicídios vs. Log do PIB per capita (2014)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Logaritmo Natural do PIB Estadual per capita [ln_pib_pc]", fontsize=11, fontweight="bold")
ax.set_ylabel("Taxa de Homicídios por 100 mil habitantes", fontsize=11, fontweight="bold")
ax.legend(frameon=True, facecolor="white", edgecolor="#bdc3c7", loc="upper right", fontsize=10)
plt.tight_layout()

# Salvamento em alta resolução na pasta 'imagens'
os.makedirs("imagens", exist_ok=True)
caminho_figura = os.path.join("imagens", "grafico_1_regressao_simples.png")
plt.savefig(caminho_figura, dpi=300, bbox_inches="tight")
plt.close()
print(f"\n[OK] Gráfico salvo com sucesso: '{caminho_figura}' (300 DPI)")
