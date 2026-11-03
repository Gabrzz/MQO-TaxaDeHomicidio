"""
================================================================================
SCRIPT 2: REGRESSÃO MÚLTIPLA (3 VARIÁVEIS EXPLICATIVAS E 1 DEPENDENTE)
Tema: Modelagem da Criminalidade com ln(PIB), Gini e Taxa de Desemprego
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
# 1. CARREGAMENTO DA BASE CONSOLIDADA (PASTA DADOS)
# ------------------------------------------------------------------------------
caminho_dados = os.path.join("dados", "base_consolidada_2014.csv")
if not os.path.exists(caminho_dados):
    raise FileNotFoundError(f"Arquivo de dados não encontrado: '{caminho_dados}'")

df = pd.read_csv(caminho_dados, sep=";", decimal=",")

print("=" * 80)
print(f"BASE DE DADOS CARREGADA COM SUCESSO: '{caminho_dados}' (27 UFs)")
print("=" * 80)
print(df[["uf", "taxa_homicidios", "ln_pib_pc", "gini", "desemprego"]].head())

# ------------------------------------------------------------------------------
# 2. ESTIMAÇÃO DA REGRESSÃO MÚLTIPLA COMPLETA
# Variável Dependente (1): taxa_homicidios
# Variáveis Explicativas (3): ln_pib_pc (X1), gini (X2), desemprego (X3)
# Equação Populacional: Homicídios = beta_0 + beta_1*ln_pib + beta_2*gini + beta_3*desemp + u
# ------------------------------------------------------------------------------
y = df["taxa_homicidios"]
X_multiplo = sm.add_constant(df[["ln_pib_pc", "gini", "desemprego"]])

modelo_multiplo = sm.OLS(y, X_multiplo).fit()

print("\n" + "=" * 80)
print("RESULTADOS DA REGRESSÃO MÚLTIPLA COMPLETA (3 EXPLICATIVAS E 1 DEPENDENTE)")
print("=" * 80)
print(modelo_multiplo.summary())

# Interpretação dos coeficientes
beta0 = modelo_multiplo.params["const"]
beta1 = modelo_multiplo.params["ln_pib_pc"]
beta2 = modelo_multiplo.params["gini"]
beta3 = modelo_multiplo.params["desemprego"]
r2 = modelo_multiplo.rsquared
r2_adj = modelo_multiplo.rsquared_adj
f_val = modelo_multiplo.fvalue
f_pval = modelo_multiplo.f_pvalue

print("-" * 80)
print("INTERPRETAÇÃO ECONÔMICA DOS COEFICIENTES (CETERIS PARIBUS):")
print(f"• Intercepto (beta_0)      : {beta0:+.4f} (p = {modelo_multiplo.pvalues['const']:.4f})")
print(f"• ln(PIB per capita) (beta_1): {beta1:+.4f} (p = {modelo_multiplo.pvalues['ln_pib_pc']:.4f})")
print(f"  -> Aumento de 1% no PIB pc reduz em média ~{abs(beta1)/100:.3f} homicídios/100k hab., ceteris paribus.")
print(f"• Índice de Gini (beta_2)  : {beta2:+.4f} (p = {modelo_multiplo.pvalues['gini']:.4f})")
print(f"• Desemprego % (beta_3)    : {beta3:+.4f} (p = {modelo_multiplo.pvalues['desemprego']:.4f})")
print(f"  -> Cada 1 p.p. a mais na taxa de desocupação eleva em média ~{beta3:.3f} homicídios/100k hab.")
print(f"• R² = {r2:.4f} | R² Ajustado = {r2_adj:.4f} | F = {f_val:.4f} (p = {f_pval:.4f})")
print("-" * 80)

# ------------------------------------------------------------------------------
# 3. COMPARAÇÃO PEDAGÓGICA: VIÉS DE VARIÁVEL OMITIDA (WOOLDRIDGE)
# Compara com o Modelo Simples (Y ~ ln_pib) e o Benchmark (Y ~ ln_pib + gini)
# ------------------------------------------------------------------------------
X_simp = sm.add_constant(df["ln_pib_pc"])
mod_simp = sm.OLS(y, X_simp).fit()

X_bench = sm.add_constant(df[["ln_pib_pc", "gini"]])
mod_bench = sm.OLS(y, X_bench).fit()

mod_aux = sm.OLS(df["gini"], X_simp).fit()

b1_til = mod_simp.params["ln_pib_pc"]
b1_bench = mod_bench.params["ln_pib_pc"]
b2_bench = mod_bench.params["gini"]
delta_1 = mod_aux.params["ln_pib_pc"]
vies = b2_bench * delta_1

print("\n" + "=" * 80)
print("DECOMPOSIÇÃO FORMAL DO VIÉS DE VARIÁVEL OMITIDA (IDENTIDADE DE WOOLDRIDGE):")
print("=" * 80)
print(f"1. Coeficiente Simples (Beta1_til)        : {b1_til:12.6f}")
print(f"2. Coeficiente Controlado (Beta1_chapeu) : {b1_bench:12.6f}")
print(f"3. Efeito da Desigualdade (Beta2_chapeu) : {b2_bench:12.6f}")
print(f"4. Regressão Auxiliar Gini~PIB (delta_1) : {delta_1:12.6f}")
print(f"-> Viés Analítico (Beta2 * delta_1)      : {vies:12.6f}")
print(f"-> Recomposição (Beta1 + Viés)           : {b1_bench + vies:12.6f}")
print(f"-> Discrepância Numérica                 : {abs(b1_til - (b1_bench + vies)):12.4e}")
print("=" * 80)

# ------------------------------------------------------------------------------
# 4. GERAÇÃO DO GRÁFICO DA REGRESSÃO MÚLTIPLA
# Efeito Parcial de ln(PIB) com Gini e Desemprego fixados na média amostral vs. Simples
# ------------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, ax = plt.subplots(figsize=(9, 6.5))

gini_medio = df["gini"].mean()
desemp_medio = df["desemprego"].mean()

# Reta da regressão múltipla completa avaliada nas médias das outras 2 variáveis
intercepto_ajustado = beta0 + beta2 * gini_medio + beta3 * desemp_medio
x_grid = np.linspace(df["ln_pib_pc"].min() - 0.05, df["ln_pib_pc"].max() + 0.05, 100)
y_mult_grid = intercepto_ajustado + beta1 * x_grid

# Reta do modelo simples
y_simp_grid = mod_simp.params["const"] + b1_til * x_grid

# Dispersão real das 27 UFs
ax.scatter(
    df["ln_pib_pc"], df["taxa_homicidios"],
    color="#2c3e50", s=65, alpha=0.85, edgecolors="white", linewidth=1.2,
    label="Observações Reais (27 UFs)", zorder=4
)

# Rótulos para UFs emblemáticas
ufs_destaque = ["SP", "SC", "AL", "SE", "DF", "RJ"]
for _, r in df[df["uf"].isin(ufs_destaque)].iterrows():
    ax.annotate(r["uf"], (r["ln_pib_pc"], r["taxa_homicidios"]), textcoords="offset points", xytext=(5, 4), fontsize=9, fontweight="bold", color="#34495e")

# Modelo Simples: Linha Sólida/Contínua Vermelha
ax.plot(
    x_grid, y_simp_grid,
    color="#c0392b", linestyle="-", linewidth=2.5,
    label=f"Modelo Simples: $\\tilde{{\\beta}}_1$ = {b1_til:.2f} (Enviesado)", zorder=3
)

# Modelo Múltiplo: Linha Tracejada Verde
ax.plot(
    x_grid, y_mult_grid,
    color="#27ae60", linestyle="--", linewidth=2.5,
    label=f"Modelo Múltiplo (médias fixas): $\\hat{{\\beta}}_1$ = {beta1:.2f}", zorder=3
)

ax.set_title("Regressão Múltipla vs. Modelo Simples\nEfeito do PIB per capita Controlando Simultaneamente por Gini e Desemprego", fontsize=12, fontweight="bold", pad=12)
ax.set_xlabel("Logaritmo do PIB per capita [ln_pib_pc]", fontsize=11, fontweight="bold")
ax.set_ylabel("Taxa de Homicídios por 100 mil habitantes (Y)", fontsize=11, fontweight="bold")
ax.legend(frameon=True, facecolor="white", edgecolor="#bdc3c7", loc="upper right", fontsize=10)

plt.tight_layout()

# Salvamento do gráfico da regressão múltipla na pasta 'imagens'
os.makedirs("imagens", exist_ok=True)
caminho_figura_multipla = os.path.join("imagens", "grafico_2_regressao_multipla.png")
plt.savefig(caminho_figura_multipla, dpi=300, bbox_inches="tight")
plt.close()

print(f"\n[OK] Gráfico da Regressão Múltipla salvo com sucesso: '{caminho_figura_multipla}' (300 DPI)")
print("=" * 80 + "\n")
print("=" * 80 + "\n")
