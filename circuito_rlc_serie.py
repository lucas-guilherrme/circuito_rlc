#!/usr/bin/env python3
"""
ANÁLISE DO TRANSIENTE EM CIRCUITOS RLC SÉRIE
Baseado no Capítulo 8 - Fundamentos de Circuitos Elétricos (Sadiku)
Autor: Python para Engenharia Elétrica
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
import sympy as sp
from sympy import symbols, Function, Eq, dsolve, diff, simplify, lambdify
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Configuração do matplotlib para gráficos mais bonitos
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16

@dataclass
class CircuitoRLC_Serie:
    """Classe para análise de circuitos RLC série"""
    
    def __init__(self, R: float, L: float, C: float, V: float = 0):
        """
        Inicializa o circuito RLC série
        
        Parâmetros:
        -----------
        R : float
            Resistência (Ω)
        L : float
            Indutância (H)
        C : float
            Capacitância (F)
        V : float, opcional
            Tensão da fonte DC (V)
        """
        self.R = R
        self.L = L
        self.C = C
        self.V = V
        
        # Parâmetros característicos
        self.alpha = R / (2 * L)  # Coeficiente de amortecimento (Np/s)
        self.omega0 = 1 / np.sqrt(L * C)  # Frequência natural (rad/s)
        self.zeta = self.alpha / self.omega0  # Fator de amortecimento
        
        # Determinar tipo de resposta
        self.tipo_resposta = self._classificar_resposta()
        
    def _classificar_resposta(self) -> str:
        """Classifica o tipo de resposta do circuito"""
        if self.zeta > 1:
            return "superamortecido"
        elif abs(self.zeta - 1) < 1e-9:
            return "criticamente amortecido"
        else:
            return "subamortecido"
    
    def calcular_parametros(self) -> Dict[str, float]:
        """Calcula e retorna os parâmetros característicos"""
        if self.tipo_resposta == "subamortecido":
            omega_d = np.sqrt(self.omega0**2 - self.alpha**2)
            Q = self.omega0 / (2 * self.alpha)  # Fator de qualidade
            t_settle_2 = 4 / self.alpha
            t_settle_5 = 3 / self.alpha
        else:
            omega_d = 0
            Q = 0
            t_settle_2 = 4 / self.alpha
            t_settle_5 = 3 / self.alpha
            
        return {
            'alpha': self.alpha,
            'omega0': self.omega0,
            'zeta': self.zeta,
            'omega_d': omega_d if self.tipo_resposta == "subamortecido" else 0,
            'Q': Q,
            't_settle_2%': t_settle_2,
            't_settle_5%': t_settle_5,
            'tipo': self.tipo_resposta
        }
    
    def resolver_sistema_edos(self, t_span: Tuple[float, float], 
                             cond_iniciais: Tuple[float, float],
                             n_points: int = 1000) -> Dict[str, np.ndarray]:
        """
        Resolve o sistema de EDOs numericamente
        
        Parâmetros:
        -----------
        t_span : tuple
            Intervalo de tempo (t_inicial, t_final)
        cond_iniciais : tuple
            Condições iniciais (i0, vC0)
        n_points : int
            Número de pontos na solução
            
        Retorna:
        --------
        dict : Dicionário com as soluções
        """
        # Sistema de EDOs: di/dt = (V - R*i - vC)/L, dvC/dt = i/C
        def sistema_edos(t, y, V, R, L, C):
            i, vC = y
            di_dt = (V - R * i - vC) / L
            dvC_dt = i / C
            return [di_dt, dvC_dt]
        
        # Vetor de tempo
        t = np.linspace(t_span[0], t_span[1], n_points)
        
        # Resolver usando solve_ivp
        sol = solve_ivp(
            lambda t, y: sistema_edos(t, y, self.V, self.R, self.L, self.C),
            t_span,
            cond_iniciais,
            t_eval=t,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        # Extrair resultados
        t_sol = sol.t
        i_sol = sol.y[0]
        vC_sol = sol.y[1]
        
        # Calcular outras grandezas
        vR_sol = self.R * i_sol
        vL_sol = self.V - vR_sol - vC_sol
        pR_sol = i_sol**2 * self.R
        
        # Energias
        W_L = 0.5 * self.L * i_sol**2
        W_C = 0.5 * self.C * vC_sol**2
        W_total = W_L + W_C
        
        return {
            't': t_sol,
            'i': i_sol,
            'vC': vC_sol,
            'vR': vR_sol,
            'vL': vL_sol,
            'pR': pR_sol,
            'W_L': W_L,
            'W_C': W_C,
            'W_total': W_total
        }
    
    def resolver_analiticamente(self, cond_iniciais: Tuple[float, float]) -> sp.Expr:
        """
        Resolve analiticamente usando sympy
        
        Parâmetros:
        -----------
        cond_iniciais : tuple
            Condições iniciais (i0, vC0)
            
        Retorna:
        --------
        sympy.Expr : Expressão simbólica para i(t)
        """
        # Definir símbolos
        t = symbols('t', real=True, positive=True)
        i = Function('i')(t)
        
        # Equação diferencial: L*i'' + R*i' + (1/C)*i = 0
        # (considerando resposta natural após degrau)
        eq = Eq(self.L * diff(i, t, 2) + self.R * diff(i, t) + (1/self.C) * i, 0)
        
        # Condições iniciais
        i0, vC0 = cond_iniciais
        di0 = (self.V - self.R * i0 - vC0) / self.L
        
        # Resolver
        sol = dsolve(eq, i, ics={i.subs(t, 0): i0, diff(i, t).subs(t, 0): di0})
        
        return simplify(sol.rhs)
    
    def plotar_resposta(self, solucao: Dict[str, np.ndarray], 
                       titulo: str = "Resposta Transitória - Circuito RLC Série"):
        """
        Plota os resultados da simulação
        
        Parâmetros:
        -----------
        solucao : dict
            Dicionário com as soluções do sistema
        titulo : str
            Título do gráfico
        """
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        # 1. Corrente no circuito
        ax = axes[0, 0]
        ax.plot(solucao['t'], solucao['i'], 'b-', linewidth=2)
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Corrente i(t) (A)')
        ax.set_title('Corrente no Circuito')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        # 2. Tensões nos componentes
        ax = axes[0, 1]
        ax.plot(solucao['t'], solucao['vR'], 'r-', label='V_R', linewidth=2)
        ax.plot(solucao['t'], solucao['vL'], 'g-', label='V_L', linewidth=2)
        ax.plot(solucao['t'], solucao['vC'], 'b-', label='V_C', linewidth=2)
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Tensão (V)')
        ax.set_title('Tensões nos Componentes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Potência no resistor
        ax = axes[0, 2]
        ax.plot(solucao['t'], solucao['pR'], 'm-', linewidth=2)
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Potência (W)')
        ax.set_title('Potência Dissipada no Resistor')
        ax.grid(True, alpha=0.3)
        
        # 4. Energias armazenadas
        ax = axes[1, 0]
        ax.plot(solucao['t'], solucao['W_L'], 'g-', label='Indutor', linewidth=2)
        ax.plot(solucao['t'], solucao['W_C'], 'b-', label='Capacitor', linewidth=2)
        ax.plot(solucao['t'], solucao['W_total'], 'r-', label='Total', linewidth=2)
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Energia (J)')
        ax.set_title('Energia Armazenada')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Diagrama de fases (vC vs i)
        ax = axes[1, 1]
        ax.plot(solucao['vC'], solucao['i'], 'b-', linewidth=1.5)
        ax.scatter(solucao['vC'][0], solucao['i'][0], color='r', s=100, 
                  zorder=5, label='Início')
        ax.scatter(solucao['vC'][-1], solucao['i'][-1], color='g', s=100,
                  zorder=5, label='Fim')
        ax.set_xlabel('Tensão no Capacitor v_C (V)')
        ax.set_ylabel('Corrente i (A)')
        ax.set_title('Diagrama de Fases (v_C vs i)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        
        # 6. Corrente e derivada
        ax = axes[1, 2]
        ax.plot(solucao['t'], solucao['i'], 'b-', label='i(t)', linewidth=2)
        di_dt = np.gradient(solucao['i'], solucao['t'])
        ax.plot(solucao['t'], di_dt / np.max(np.abs(di_dt)) * np.max(np.abs(solucao['i'])), 
                'r--', label='di/dt (normalizado)', linewidth=1.5)
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Corrente (A)')
        ax.set_title('Corrente e Derivada')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 7. Comparação V_R, V_L, V_C em subplot
        ax = axes[2, 0]
        time_limited = solucao['t'] < min(0.1, np.max(solucao['t']))
        if np.any(time_limited):
            t_lim = solucao['t'][time_limited]
            ax.plot(t_lim, solucao['vR'][time_limited], 'r-', linewidth=2)
            ax.plot(t_lim, solucao['vL'][time_limited], 'g-', linewidth=2)
            ax.plot(t_lim, solucao['vC'][time_limited], 'b-', linewidth=2)
            ax.set_xlabel('Tempo (s)')
            ax.set_ylabel('Tensão (V)')
            ax.set_title('Tensões (zoom inicial)')
            ax.grid(True, alpha=0.3)
        
        # 8. Análise logarítmica do decaimento
        ax = axes[2, 1]
        envelope = np.exp(-self.alpha * solucao['t'])
        ax.semilogy(solucao['t'], np.abs(solucao['i']), 'b-', label='|i(t)|', linewidth=2)
        ax.semilogy(solucao['t'], envelope * np.max(np.abs(solucao['i'])), 
                   'r--', label='Envelope', linewidth=2)
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('|Corrente| (A)')
        ax.set_title('Decaimento Exponencial')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
        
        # 9. Informações do circuito
        ax = axes[2, 2]
        ax.axis('off')
        params = self.calcular_parametros()
        info_text = (
            f"R = {self.R:.2f} Ω\n"
            f"L = {self.L:.4f} H\n"
            f"C = {self.C:.4e} F\n"
            f"V = {self.V:.2f} V\n\n"
            f"α = {params['alpha']:.4f} Np/s\n"
            f"ω₀ = {params['omega0']:.4f} rad/s\n"
            f"ζ = {params['zeta']:.4f}\n"
            f"Tipo: {params['tipo']}\n"
        )
        if params['tipo'] == 'subamortecido':
            info_text += f"ω_d = {params['omega_d']:.4f} rad/s\n"
            info_text += f"Q = {params['Q']:.4f}\n"
            info_text += f"t_settle(2%) = {params['t_settle_2%']:.4f} s"
        
        ax.text(0.1, 0.5, info_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(titulo, fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def analisar_sensibilidade(self, t_span: Tuple[float, float],
                              cond_iniciais: Tuple[float, float],
                              variacao_percentual: float = 20):
        """
        Analisa a sensibilidade da resposta a variações nos componentes
        
        Parâmetros:
        -----------
        t_span : tuple
            Intervalo de tempo
        cond_iniciais : tuple
            Condições iniciais
        variacao_percentual : float
            Variação percentual (±%)
        """
        # Valores base
        R_base, L_base, C_base = self.R, self.L, self.C
        
        # Variações a testar
        variacoes = np.array([-variacao_percentual/100, 0, variacao_percentual/100])
        cores = ['r', 'b', 'g']
        estilos = ['--', '-', ':']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Variação de R
        ax = axes[0]
        for i, var in enumerate(variacoes):
            R_test = R_base * (1 + var)
            circuito_test = CircuitoRLC_Serie(R_test, L_base, C_base, self.V)
            sol_test = circuito_test.resolver_sistema_edos(t_span, cond_iniciais, 500)
            
            label = f"R = {R_test:.2f} Ω"
            if var < 0:
                label += " (-20%)"
            elif var > 0:
                label += " (+20%)"
            else:
                label += " (base)"
                
            ax.plot(sol_test['t'], sol_test['i'], color=cores[i], 
                   linestyle=estilos[i], linewidth=2, label=label)
        
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Corrente i(t) (A)')
        ax.set_title('Sensibilidade à Resistência R')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Variação de L
        ax = axes[1]
        for i, var in enumerate(variacoes):
            L_test = L_base * (1 + var)
            circuito_test = CircuitoRLC_Serie(R_base, L_test, C_base, self.V)
            sol_test = circuito_test.resolver_sistema_edos(t_span, cond_iniciais, 500)
            
            label = f"L = {L_test:.4f} H"
            if var < 0:
                label += " (-20%)"
            elif var > 0:
                label += " (+20%)"
            else:
                label += " (base)"
                
            ax.plot(sol_test['t'], sol_test['i'], color=cores[i], 
                   linestyle=estilos[i], linewidth=2, label=label)
        
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Corrente i(t) (A)')
        ax.set_title('Sensibilidade à Indutância L')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Variação de C
        ax = axes[2]
        for i, var in enumerate(variacoes):
            C_test = C_base * (1 + var)
            circuito_test = CircuitoRLC_Serie(R_base, L_base, C_test, self.V)
            sol_test = circuito_test.resolver_sistema_edos(t_span, cond_iniciais, 500)
            
            label = f"C = {C_test:.4e} F"
            if var < 0:
                label += " (-20%)"
            elif var > 0:
                label += " (+20%)"
            else:
                label += " (base)"
                
            ax.plot(sol_test['t'], sol_test['i'], color=cores[i], 
                   linestyle=estilos[i], linewidth=2, label=label)
        
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Corrente i(t) (A)')
        ax.set_title('Sensibilidade à Capacitância C')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Análise de Sensibilidade (±{variacao_percentual}%)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def exportar_dados(self, solucao: Dict[str, np.ndarray], 
                      nome_arquivo: str = "rlc_serie_dados"):
        """
        Exporta os dados para arquivo CSV
        
        Parâmetros:
        -----------
        solucao : dict
            Dicionário com as soluções
        nome_arquivo : str
            Nome do arquivo (sem extensão)
        """
        # Criar DataFrame
        df = pd.DataFrame({
            'Tempo_s': solucao['t'],
            'Corrente_A': solucao['i'],
            'V_R_V': solucao['vR'],
            'V_L_V': solucao['vL'],
            'V_C_V': solucao['vC'],
            'Potencia_R_W': solucao['pR'],
            'Energia_L_J': solucao['W_L'],
            'Energia_C_J': solucao['W_C'],
            'Energia_Total_J': solucao['W_total']
        })
        
        # Exportar para CSV
        df.to_csv(f"{nome_arquivo}.csv", index=False)
        
        # Exportar parâmetros
        params = self.calcular_parametros()
        with open(f"{nome_arquivo}_parametros.txt", 'w') as f:
            f.write("PARÂMETROS DO CIRCUITO RLC SÉRIE\n")
            f.write("=" * 40 + "\n")
            f.write(f"Resistência R = {self.R} Ω\n")
            f.write(f"Indutância L = {self.L} H\n")
            f.write(f"Capacitância C = {self.C} F\n")
            f.write(f"Tensão da fonte V = {self.V} V\n\n")
            f.write(f"Coeficiente de amortecimento α = {params['alpha']:.6f} Np/s\n")
            f.write(f"Frequência natural ω₀ = {params['omega0']:.6f} rad/s\n")
            f.write(f"Fator de amortecimento ζ = {params['zeta']:.6f}\n")
            f.write(f"Tipo de resposta: {params['tipo']}\n")
            
            if params['tipo'] == 'subamortecido':
                f.write(f"Frequência amortecida ω_d = {params['omega_d']:.6f} rad/s\n")
                f.write(f"Fator de qualidade Q = {params['Q']:.6f}\n")
                f.write(f"Tempo de acomodação (2%) = {params['t_settle_2%']:.6f} s\n")
                f.write(f"Tempo de acomodação (5%) = {params['t_settle_5%']:.6f} s\n")
        
        print(f"Dados exportados para {nome_arquivo}.csv")
        print(f"Parâmetros exportados para {nome_arquivo}_parametros.txt")


# Função principal para análise de circuito RLC série
def analisar_rlc_serie():
    """Interface principal para análise de circuito RLC série"""
    
    print("=" * 60)
    print("ANÁLISE DO TRANSIENTE - CIRCUITO RLC SÉRIE")
    print("=" * 60)
    
    # Entrada de parâmetros
    print("\n--- PARÂMETROS DO CIRCUITO ---")
    
    # Valores padrão para teste rápido
    usar_exemplo = input("Usar exemplo rápido? (s/n): ").strip().lower()
    
    if usar_exemplo == 's':
        print("\nEscolha um exemplo:")
        print("1. Subamortecido (R=10Ω, L=0.1H, C=100μF, V=12V)")
        print("2. Criticamente amortecido (R=20Ω, L=0.1H, C=100μF, V=12V)")
        print("3. Superamortecido (R=100Ω, L=0.1H, C=100μF, V=12V)")
        
        escolha = input("Digite 1, 2 ou 3: ").strip()
        
        if escolha == '1':
            R, L, C, V = 10, 0.1, 100e-6, 12
            i0, vC0 = 0, 0
            t_final = 0.1
        elif escolha == '2':
            R, L, C, V = 20, 0.1, 100e-6, 12
            i0, vC0 = 0, 0
            t_final = 0.1
        elif escolha == '3':
            R, L, C, V = 100, 0.1, 100e-6, 12
            i0, vC0 = 0, 0
            t_final = 0.1
        else:
            print("Opção inválida. Usando exemplo subamortecido.")
            R, L, C, V = 10, 0.1, 100e-6, 12
            i0, vC0 = 0, 0
            t_final = 0.1
            
    else:
        try:
            R = float(input("Resistência R (Ω): "))
            L = float(input("Indutância L (H): "))
            C = float(input("Capacitância C (F): "))
            V = float(input("Tensão da fonte DC V (V): "))
            
            print("\n--- CONDIÇÕES INICIAIS ---")
            i0 = float(input("Corrente inicial no indutor i(0) (A): "))
            vC0 = float(input("Tensão inicial no capacitor vC(0) (V): "))
            
            print("\n--- TEMPO DE SIMULAÇÃO ---")
            t_final = float(input("Tempo final de simulação (s): "))
            
        except ValueError:
            print("Entrada inválida! Usando valores padrão.")
            R, L, C, V = 10, 0.1, 100e-6, 12
            i0, vC0 = 0, 0
            t_final = 0.1
    
    # Criar circuito
    circuito = CircuitoRLC_Serie(R, L, C, V)
    
    # Exibir parâmetros
    params = circuito.calcular_parametros()
    
    print("\n" + "=" * 60)
    print("PARÂMETROS CARACTERÍSTICOS")
    print("=" * 60)
    print(f"Resistência R = {R:.4f} Ω")
    print(f"Indutância L = {L:.4f} H")
    print(f"Capacitância C = {C:.4e} F")
    print(f"Tensão da fonte V = {V:.4f} V")
    print(f"\nCoeficiente de amortecimento α = {params['alpha']:.4f} Np/s")
    print(f"Frequência natural ω₀ = {params['omega0']:.4f} rad/s")
    print(f"Frequência natural f₀ = {params['omega0']/(2*np.pi):.4f} Hz")
    print(f"Fator de amortecimento ζ = α/ω₀ = {params['zeta']:.4f}")
    print(f"\nTIPO DE RESPOSTA: {params['tipo'].upper()}")
    
    if params['tipo'] == 'subamortecido':
        print(f"Frequência amortecida ω_d = {params['omega_d']:.4f} rad/s")
        print(f"Frequência amortecida f_d = {params['omega_d']/(2*np.pi):.4f} Hz")
        print(f"Fator de qualidade Q = {params['Q']:.4f}")
        print(f"Tempo de acomodação (2%) = {params['t_settle_2%']:.4f} s")
        print(f"Tempo de acomodação (5%) = {params['t_settle_5%']:.4f} s")
    
    # Resolver sistema
    print("\n" + "=" * 60)
    print("RESOLVENDO SISTEMA DE EDOs...")
    
    t_span = (0, t_final)
    cond_iniciais = (i0, vC0)
    
    solucao = circuito.resolver_sistema_edos(t_span, cond_iniciais, 1000)
    
    # Plotar resultados
    print("GERANDO GRÁFICOS...")
    circuito.plotar_resposta(solucao, f"Circuito RLC Série - {params['tipo']}")
    
    # Análise de sensibilidade
    print("\n" + "=" * 60)
    analisar_sensibilidade = input("Realizar análise de sensibilidade? (s/n): ").strip().lower()
    
    if analisar_sensibilidade == 's':
        variacao = float(input("Variação percentual (±%): "))
        circuito.analisar_sensibilidade(t_span, cond_iniciais, variacao)
    
    # Exportar dados
    print("\n" + "=" * 60)
    exportar_dados = input("Exportar dados para arquivo? (s/n): ").strip().lower()
    
    if exportar_dados == 's':
        nome_arquivo = input("Nome do arquivo (sem extensão): ").strip()
        if not nome_arquivo:
            nome_arquivo = "rlc_serie_dados"
        circuito.exportar_dados(solucao, nome_arquivo)
    
    print("\n" + "=" * 60)
    print("ANÁLISE CONCLUÍDA COM SUCESSO!")
    print("=" * 60)
    
    return circuito, solucao, params


# Exemplo de uso rápido
def exemplo_rapido_rlc_serie():
    """Executa exemplos rápidos de circuitos RLC série"""
    
    exemplos = [
        {
            "nome": "Subamortecido",
            "R": 10,
            "L": 0.1,
            "C": 100e-6,
            "V": 12,
            "i0": 0,
            "vC0": 0,
            "t_final": 0.1
        },
        {
            "nome": "Criticamente Amortecido",
            "R": 20,
            "L": 0.1,
            "C": 100e-6,
            "V": 12,
            "i0": 0,
            "vC0": 0,
            "t_final": 0.1
        },
        {
            "nome": "Superamortecido",
            "R": 100,
            "L": 0.1,
            "C": 100e-6,
            "V": 12,
            "i0": 0,
            "vC0": 0,
            "t_final": 0.2
        }
    ]
    
    for exemplo in exemplos:
        print(f"\n{'='*60}")
        print(f"EXEMPLO: {exemplo['nome']}")
        print(f"{'='*60}")
        
        circuito = CircuitoRLC_Serie(
            exemplo['R'], 
            exemplo['L'], 
            exemplo['C'], 
            exemplo['V']
        )
        
        params = circuito.calcular_parametros()
        print(f"α = {params['alpha']:.4f} Np/s")
        print(f"ω₀ = {params['omega0']:.4f} rad/s")
        print(f"ζ = {params['zeta']:.4f}")
        print(f"Tipo: {params['tipo']}")
        
        solucao = circuito.resolver_sistema_edos(
            (0, exemplo['t_final']),
            (exemplo['i0'], exemplo['vC0']),
            500
        )
        
        circuito.plotar_resposta(solucao, f"Exemplo: {exemplo['nome']}")


if __name__ == "__main__":
    # Executar análise interativan
    circuito, solucao, params = analisar_rlc_serie()
    
    # Para executar exemplos rápidos, descomente a linha abaixo:
    # exemplo_rapido_rlc_serie()