# Plano de Apresentação do Trabalho & Guia de Arguição

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
| **$R^2$** | 0.3489 | 0.2752 | 0.2238 |
| **$R^2$ Ajustado** | 0.2640 | 0.2148 | 0.1927 |
| **Estatística F** | 4.1082 (p = 0.0180) | 4.5561 (p = 0.0210) | 7.2076 (p = 0.0127) |

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
