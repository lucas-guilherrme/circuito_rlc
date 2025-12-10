#!/usr/bin/env python3
"""
ANÁLISE DO TRANSIENTE EM CIRCUITOS RLC PARALELO
Baseado no Capítulo 8 - Fundamentos de Circuitos Elétricos (Sadiku)
Autor: Python para Engenharia Elétrica
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sp
from sympy import symbols, Function, Eq, dsolve, diff, simplify, integrate
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, Any

@dataclass
class CircuitoRLC_Paralelo:
    """Classe para análise de circuitos RLC paralelo"""
    
    def __init__(self, R: float, L: float, C: float, I: float = 0):
        """
        Inicializa o circuito RLC paralelo
        
        Parâmetros:
        -----------
        R : float
            Resistência (Ω)
        L : float
            Indutância (H)
        C : float
            Capacitância (F)
        I : float, opcional
            Corrente da fonte DC (A)
        """
        self.R = R
        self.L = L
        self.C = C
        self.I = I
        
        # Parâmetros característicos
        self.alpha = 1 / (2 * R * C)  # Coeficiente de amortecimento (paralelo)
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
            Q = self.R * np.sqrt(self.C / self.L)  # Fator de qualidade (paralelo)
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
            Condições iniciais (v0, iL0)
        n_points : int
            Número de pontos na solução
            
        Retorna:
        --------
        dict : Dicionário com as soluções
        """
        # Sistema de EDOs: dv/dt = (I - v/R - iL)/C, diL/dt = v/L
        def sistema_edos(t, y, I, R, L, C):
            v, iL = y
            dv_dt = (I - v/R - iL) / C
            diL_dt = v / L
            return [dv_dt, diL_dt]
        
        # Vetor de tempo
        t = np.linspace(t_span[0], t_span[1], n_points)
        
        # Resolver usando solve_ivp
        sol = solve_ivp(
            lambda t, y: sistema_edos(t, y, self.I, self.R, self.L, self.C),
            t_span,
            cond_iniciais,
            t_eval=t,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        # Extrair resultados
        t_sol = sol.t
        v_sol = sol.y[0]
        iL_sol = sol.y[1]
        
        # Calcular outras grandezas
        iR_sol = v_sol / self.R
        iC_sol = self.I - iR_sol - iL_sol
        pR_sol = v_sol**2 / self.R
        
        # Energias
        W_C = 0.5 * self.C * v_sol**2
        W_L = 0.5 * self.L * iL_sol**2
        W_total = W_C + W_L
        
        return {
            't': t_sol,
            'v': v_sol,
            'iL': iL_sol,
            'iR': iR_sol,
            'iC': iC_sol,
            'pR': pR_sol,
            'W_C': W_C,
            'W_L': W_L,
            'W_total': W_total
        }
    
    def plotar_resposta(self, solucao: Dict[str, np.ndarray], 
                       titulo: str = "Resposta Transitória - Circuito RLC Paralelo"):
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
        
        # 1. Tensão no circuito
        ax = axes[0, 0]
        ax.plot(solucao['t'], solucao['v'], 'b-', linewidth=2)
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Tensão v(t) (V)')
        ax.set_title('Tensão no Circuito')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        # 2. Correntes nos componentes
        ax = axes[0, 1]
        ax.plot(solucao['t'], solucao['iR'], 'r-', label='i_R', linewidth=2)
        ax.plot(solucao['t'], solucao['iL'], 'g-', label='i_L', linewidth=2)
        ax.plot(solucao['t'], solucao['iC'], 'b-', label='i_C', linewidth=2)
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Corrente (A)')
        ax.set_title('Correntes nos Componentes')
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
        ax.plot(solucao['t'], solucao['W_C'], 'b-', label='Capacitor', linewidth=2)
        ax.plot(solucao['t'], solucao['W_L'], 'g-', label='Indutor', linewidth=2)
        ax.plot(solucao['t'], solucao['W_total'], 'r-', label='Total', linewidth=2)
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Energia (J)')
        ax.set_title('Energia Armazenada')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Diagrama de fases (v vs iL)
        ax = axes[1, 1]
        ax.plot(solucao['v'], solucao['iL'], 'b-', linewidth=1.5)
        ax.scatter(solucao['v'][0], solucao['iL'][0], color='r', s=100, 
                  zorder=5, label='Início')
        ax.scatter(solucao['v'][-1], solucao['iL'][-1], color='g', s=100,
                  zorder=5, label='Fim')
        ax.set_xlabel('Tensão v (V)')
        ax.set_ylabel('Corrente no indutor i_L (A)')
        ax.set_title('Diagrama de Fases (v vs i_L)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        
        # 6. Tensão e derivada
        ax = axes[1, 2]
        ax.plot(solucao['t'], solucao['v'], 'b-', label='v(t)', linewidth=2)
        dv_dt = np.gradient(solucao['v'], solucao['t'])
        ax.plot(solucao['t'], dv_dt / np.max(np.abs(dv_dt)) * np.max(np.abs(solucao['v'])), 
                'r--', label='dv/dt (normalizado)', linewidth=1.5)
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Tensão (V)')
        ax.set_title('Tensão e Derivada')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 7. Comparação i_R, i_L, i_C em subplot
        ax = axes[2, 0]
        time_limited = solucao['t'] < min(0.1, np.max(solucao['t']))
        if np.any(time_limited):
            t_lim = solucao['t'][time_limited]
            ax.plot(t_lim, solucao['iR'][time_limited], 'r-', linewidth=2)
            ax.plot(t_lim, solucao['iL'][time_limited], 'g-', linewidth=2)
            ax.plot(t_lim, solucao['iC'][time_limited], 'b-', linewidth=2)
            ax.set_xlabel('Tempo (s)')
            ax.set_ylabel('Corrente (A)')
            ax.set_title('Correntes (zoom inicial)')
            ax.grid(True, alpha=0.3)
        
        # 8. Análise logarítmica do decaimento
        ax = axes[2, 1]
        envelope = np.exp(-self.alpha * solucao['t'])
        ax.semilogy(solucao['t'], np.abs(solucao['v']), 'b-', label='|v(t)|', linewidth=2)
        ax.semilogy(solucao['t'], envelope * np.max(np.abs(solucao['v'])), 
                   'r--', label='Envelope', linewidth=2)
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('|Tensão| (V)')
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
            f"I = {self.I:.2f} A\n\n"
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


# Função principal para análise de circuito RLC paralelo
def analisar_rlc_paralelo():
    """Interface principal para análise de circuito RLC paralelo"""
    
    print("=" * 60)
    print("ANÁLISE DO TRANSIENTE - CIRCUITO RLC PARALELO")
    print("=" * 60)
    
    # Entrada de parâmetros
    print("\n--- PARÂMETROS DO CIRCUITO ---")
    
    # Valores padrão para teste rápido
    usar_exemplo = input("Usar exemplo rápido? (s/n): ").strip().lower()
    
    if usar_exemplo == 's':
        print("\nEscolha um exemplo:")
        print("1. Subamortecido (R=100Ω, L=0.1H, C=10μF, I=1A)")
        print("2. Criticamente amortecido (R=50Ω, L=0.1H, C=10μF, I=1A)")
        print("3. Superamortecido (R=10Ω, L=0.1H, C=10μF, I=1A)")
        
        escolha = input("Digite 1, 2 ou 3: ").strip()
        
        if escolha == '1':
            R, L, C, I = 100, 0.1, 10e-6, 1
            v0, iL0 = 0, 0
            t_final = 0.02
        elif escolha == '2':
            R, L, C, I = 50, 0.1, 10e-6, 1
            v0, iL0 = 0, 0
            t_final = 0.02
        elif escolha == '3':
            R, L, C, I = 10, 0.1, 10e-6, 1
            v0, iL0 = 0, 0
            t_final = 0.02
        else:
            print("Opção inválida. Usando exemplo subamortecido.")
            R, L, C, I = 100, 0.1, 10e-6, 1
            v0, iL0 = 0, 0
            t_final = 0.02
            
    else:
        try:
            R = float(input("Resistência R (Ω): "))
            L = float(input("Indutância L (H): "))
            C = float(input("Capacitância C (F): "))
            I = float(input("Corrente da fonte DC I (A): "))
            
            print("\n--- CONDIÇÕES INICIAIS ---")
            v0 = float(input("Tensão inicial no capacitor v(0) (V): "))
            iL0 = float(input("Corrente inicial no indutor iL(0) (A): "))
            
            print("\n--- TEMPO DE SIMULAÇÃO ---")
            t_final = float(input("Tempo final de simulação (s): "))
            
        except ValueError:
            print("Entrada inválida! Usando valores padrão.")
            R, L, C, I = 100, 0.1, 10e-6, 1
            v0, iL0 = 0, 0
            t_final = 0.02
    
    # Criar circuito
    circuito = CircuitoRLC_Paralelo(R, L, C, I)
    
    # Exibir parâmetros
    params = circuito.calcular_parametros()
    
    print("\n" + "=" * 60)
    print("PARÂMETROS CARACTERÍSTICOS")
    print("=" * 60)
    print(f"Resistência R = {R:.4f} Ω")
    print(f"Indutância L = {L:.4f} H")
    print(f"Capacitância C = {C:.4e} F")
    print(f"Corrente da fonte I = {I:.4f} A")
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
    cond_iniciais = (v0, iL0)
    
    solucao = circuito.resolver_sistema_edos(t_span, cond_iniciais, 1000)
    
    # Plotar resultados
    print("GERANDO GRÁFICOS...")
    circuito.plotar_resposta(solucao, f"Circuito RLC Paralelo - {params['tipo']}")
    
    print("\n" + "=" * 60)
    print("ANÁLISE CONCLUÍDA COM SUCESSO!")
    print("=" * 60)
    
    return circuito, solucao, params


if __name__ == "__main__":
    # Executar análise interativa
    circuito, solucao, params = analisar_rlc_paralelo()