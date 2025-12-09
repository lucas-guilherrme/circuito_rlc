# 📖 Circuitos RLC - Análise do Regime Transitório
---

O regime transitório em circuitos RLC descreve o comportamento dinâmico do circuito durante o período entre dois estados estáveis. Diferente do regime permanente senoidal, o transitório analisa como as variáveis do circuito (tensão e corrente) evoluem no tempo após uma perturbação, como o fechamento ou abertura de uma chave.

# ⚡ Circuitos de Segunda Ordem
---
## Definição e Características
Circuitos RLC são circuitos de segunda ordem porque são descritos por equações diferenciais de segunda ordem. A ordem é determinada pelo número de elementos armazenadores de energia independentes (indutores e capacitores).

# 🔄 Circuito RLC Série - Análise do Transitório
---

Para o circuito RLC série sem fonte (resposta natural):\
![Homogênea RLC](https://latex.codecogs.com/svg.latex?L\frac{d^2i}{dt^2}+R\frac{di}{dt}+\frac{1}{C}i=0#gh-light-mode-only)

![Homogênea RLC](https://latex.codecogs.com/svg.latex?\color{white}L\frac{d^2i}{dt^2}+R\frac{di}{dt}+\frac{1}{C}i=0#gh-dark-mode-only)
Para o circuito com fonte (resposta completa):\
![EDO com Fonte](https://latex.codecogs.com/svg.latex?L\frac{d^2i}{dt^2}+R\frac{di}{dt}+\frac{1}{C}i=\frac{dv_s}{dt}#gh-light-mode-only)

![EDO com Fonte](https://latex.codecogs.com/svg.latex?\color{white}L\frac{d^2i}{dt^2}+R\frac{di}{dt}+\frac{1}{C}i=\frac{dv_s}{dt}#gh-dark-mode-only)
onde v_s é a tensão da fonte.

## Forma Padrão da Equação
A equação pode ser escrita na forma:\

onde:
 * α = R/(2L) → coeficiente de amortecimento (Np/s)
 * ω₀ = 1/√(LC) → frequência natural não-amortecida (rad/s)
 * x → variável de interesse (i ou v)
## Resposta Natural (Sem Fonte)
