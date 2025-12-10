%% ============================================================
% ANÁLISE DO TRANSIENTE EM CIRCUITOS RLC PARALELO
% Baseado no Capítulo 8 - Fundamentos de Circuitos Elétricos (Sadiku)
% ============================================================

clear all; close all; clc;
fprintf('=== ANÁLISE DO TRANSIENTE - CIRCUITO RLC PARALELO ===\n\n');

%% 1. PARÂMETROS DO CIRCUITO
% ===========================
R = input('Resistência (Ω): ');
L = input('Indutância (H): ');
C = input('Capacitância (F): ');
I = input('Corrente da fonte (A): ');

% Condições iniciais
fprintf('\n--- Condições Iniciais ---\n');
v0 = input('Tensão inicial no capacitor v(0) (V): ');
iL0 = input('Corrente inicial no indutor i_L(0) (A): ');

% Tempo de simulação
t_final = input('\nTempo final de simulação (s): ');

%% 2. CÁLCULO DOS PARÂMETROS CARACTERÍSTICOS
% ===========================================
alpha = 1/(2*R*C);              % Coeficiente de amortecimento (paralelo)
omega0 = 1/sqrt(L*C);           % Frequência natural não amortecida
omega_d = sqrt(omega0^2 - alpha^2); % Frequência amortecida

% Classificação do circuito
fprintf('\n=== PARÂMETROS DO CIRCUITO ===\n');
fprintf('α = %.4f Np/s\n', alpha);
fprintf('ω₀ = %.4f rad/s\n', omega0);
fprintf('Frequência natural f₀ = %.4f Hz\n', omega0/(2*pi));

if alpha > omega0
    fprintf('\nSISTEMA SUPERAMORTECIDO (α > ω₀)\n');
    tipo = 1;
elseif abs(alpha - omega0) < 1e-6
    fprintf('\nSISTEMA CRITICAMENTE AMORTECIDO (α = ω₀)\n');
    tipo = 2;
else
    fprintf('\nSISTEMA SUBAMORTECIDO (α < ω₀)\n');
    fprintf('ω_d = %.4f rad/s\n', omega_d);
    fprintf('Frequência amortecida f_d = %.4f Hz\n', omega_d/(2*pi));
    tipo = 3;
end

% Fator de qualidade (paralelo)
if tipo == 3
    Q = R * sqrt(C/L);  % Fator de qualidade para paralelo
    t_settle_2 = 4/alpha;
    t_settle_5 = 3/alpha;
    fprintf('Fator de qualidade Q = %.4f\n', Q);
    fprintf('Tempo de acomodação (2%%) = %.4f s\n', t_settle_2);
    fprintf('Tempo de acomodação (5%%) = %.4f s\n', t_settle_5);
end

%% 3. SOLUÇÃO ANALÍTICA
% =====================
syms v(t)

% Equação diferencial: C*d²v/dt² + (1/R)*dv/dt + (1/L)*v = I'
% Para degrau de corrente: I' = 0 após t=0+
eqn = C*diff(v,t,2) + (1/R)*diff(v,t) + (1/L)*v == 0;

% Condições iniciais
cond1 = v(0) == v0;
% Para dv/dt(0): i_C(0) = C*dv/dt(0) = I - v0/R - iL0
dv0 = (I - v0/R - iL0)/C;
cond2 = diff(v) == dv0;

% Solução geral
if tipo == 1  % Superamortecido
    s1 = -alpha + sqrt(alpha^2 - omega0^2);
    s2 = -alpha - sqrt(alpha^2 - omega0^2);
    
    syms A1 A2
    v_sym = A1*exp(s1*t) + A2*exp(s2*t);
    
    eq1 = subs(v_sym, t, 0) == v0;
    eq2 = subs(diff(v_sym, t), t, 0) == dv0;
    
    [A1_sol, A2_sol] = solve([eq1, eq2], [A1, A2]);
    v_analitico = subs(v_sym, [A1, A2], [A1_sol, A2_sol]);
    
elseif tipo == 2  % Criticamente amortecido
    syms A1 A2
    v_sym = (A1 + A2*t)*exp(-alpha*t);
    
    eq1 = subs(v_sym, t, 0) == v0;
    eq2 = subs(diff(v_sym, t), t, 0) == dv0;
    
    [A1_sol, A2_sol] = solve([eq1, eq2], [A1, A2]);
    v_analitico = subs(v_sym, [A1, A2], [A1_sol, A2_sol]);
    
else  % Subamortecido
    syms B1 B2
    v_sym = exp(-alpha*t)*(B1*cos(omega_d*t) + B2*sin(omega_d*t));
    
    eq1 = subs(v_sym, t, 0) == v0;
    eq2 = subs(diff(v_sym, t), t, 0) == dv0;
    
    [B1_sol, B2_sol] = solve([eq1, eq2], [B1, B2]);
    v_analitico = subs(v_sym, [B1, B2], [B1_sol, B2_sol]);
end

v_analitico = simplify(v_analitico);
fprintf('\n=== SOLUÇÃO ANALÍTICA ===\n');
fprintf('v(t) = %s\n', char(v_analitico));

% Calcular correntes analiticamente
iR_analitico = v_analitico / R;
iC_analitico = C * diff(v_analitico, t);
iL_analitico = (1/L) * int(v_analitico, t) + iL0;

%% 4. SOLUÇÃO NUMÉRICA (ODE45)
% =============================
% Sistema: dv/dt = iC/C, diL/dt = v/L
% onde iC = I - v/R - iL

f = @(t, y) [
    (I - y(1)/R - y(2))/C;  % dv/dt
    y(1)/L                  % diL/dt
];

tspan = [0 t_final];
y0 = [v0; iL0];  % [v(0); iL(0)]

options = odeset('RelTol', 1e-8, 'AbsTol', 1e-10);
[t_num, y_num] = ode45(f, tspan, y0, options);

v_num = y_num(:, 1);
iL_num = y_num(:, 2);

% Calcular outras grandezas
iR_num = v_num / R;
iC_num = I - iR_num - iL_num;
pR_num = v_num.^2 / R;  % Potência dissipada

%% 5. VALIDAÇÃO
% =============
v_analitico_num = double(subs(v_analitico, t, t_num));
erro = max(abs(v_num - v_analitico_num));
fprintf('\n=== VALIDAÇÃO ===\n');
fprintf('Erro máximo entre soluções analítica e numérica: %.2e V\n', erro);

%% 6. VISUALIZAÇÃO DOS RESULTADOS
% ================================
figure('Position', [100, 100, 1200, 800]);

% Subplot 1: Tensão no circuito
subplot(3, 3, [1, 2]);
plot(t_num, v_num, 'b-', 'LineWidth', 2);
hold on;
plot(t_num, v_analitico_num, 'r--', 'LineWidth', 1.5);
xlabel('Tempo (s)', 'FontSize', 10);
ylabel('Tensão v(t) (V)', 'FontSize', 10);
title('Resposta Transitória - Tensão no Circuito Paralelo', 'FontSize', 12);
legend('Solução Numérica', 'Solução Analítica', 'Location', 'best');
grid on;

% Subplot 2: Correntes nos componentes
subplot(3, 3, [4, 5]);
plot(t_num, iR_num, 'r-', 'LineWidth', 2);
hold on;
plot(t_num, iL_num, 'g-', 'LineWidth', 2);
plot(t_num, iC_num, 'b-', 'LineWidth', 2);
xlabel('Tempo (s)', 'FontSize', 10);
ylabel('Corrente (A)', 'FontSize', 10);
title('Correntes nos Componentes', 'FontSize', 12);
legend('i_R(t)', 'i_L(t)', 'i_C(t)', 'Location', 'best');
grid on;

% Subplot 3: Potência dissipada
subplot(3, 3, [7, 8]);
plot(t_num, pR_num, 'm-', 'LineWidth', 2);
xlabel('Tempo (s)', 'FontSize', 10);
ylabel('Potência (W)', 'FontSize', 10);
title('Potência Dissipada no Resistor', 'FontSize', 12);
grid on;

% Subplot 4: Diagrama de fases
subplot(3, 3, [3, 6, 9]);
plot(v_num, iL_num, 'b-', 'LineWidth', 1.5);
hold on;
plot(v0, iL0, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
xlabel('Tensão v (V)', 'FontSize', 10);
ylabel('Corrente no indutor i_L (A)', 'FontSize', 10);
title('Diagrama de Fases (v vs i_L)', 'FontSize', 12);
grid on;

% Informações
annotation('textbox', [0.15, 0.85, 0.3, 0.1], 'String', ...
    sprintf('R = %.2f Ω\nL = %.4f H\nC = %.4f F\nI = %.2f A', R, L, C, I), ...
    'FitBoxToText', 'on', 'BackgroundColor', 'w');

if tipo == 1
    tipo_str = 'SUPERAMORTECIDO';
elseif tipo == 2
    tipo_str = 'CRITICAMENTE AMORTECIDO';
else
    tipo_str = 'SUBAMORTECIDO';
end

annotation('textbox', [0.65, 0.85, 0.3, 0.1], 'String', ...
    sprintf('TIPO: %s\nα = %.4f\nω₀ = %.4f\nQ = %.4f', ...
    tipo_str, alpha, omega0, Q), 'FitBoxToText', 'on', 'BackgroundColor', 'w');

%% 7. ANÁLISE DE ENERGIA
% ======================
W_C = 0.5 * C * v_num.^2;
W_L = 0.5 * L * iL_num.^2;
W_total = W_C + W_L;
W_R = cumtrapz(t_num, pR_num);

figure('Position', [100, 100, 1000, 600]);

subplot(2, 2, 1);
plot(t_num, W_C, 'b-', 'LineWidth', 2);
hold on;
plot(t_num, W_L, 'g-', 'LineWidth', 2);
plot(t_num, W_total, 'r-', 'LineWidth', 2);
xlabel('Tempo (s)');
ylabel('Energia (J)');
title('Energia Armazenada');
legend('W_C (capacitor)', 'W_L (indutor)', 'W_{total}', 'Location', 'best');
grid on;

subplot(2, 2, 2);
plot(t_num, W_R, 'm-', 'LineWidth', 2);
xlabel('Tempo (s)');
ylabel('Energia (J)');
title('Energia Dissipada no Resistor');
grid on;

% Conservação de energia
if abs(v0) < 1e-6 && abs(iL0) < 1e-6
    % Energia fornecida pela fonte: ∫I*v dt
    W_fonte = cumtrapz(t_num, I * v_num);
    W_total_sistema = W_total + W_R;
    
    subplot(2, 2, [3, 4]);
    plot(t_num, W_fonte, 'k-', 'LineWidth', 2);
    hold on;
    plot(t_num, W_total_sistema, 'r--', 'LineWidth', 2);
    xlabel('Tempo (s)');
    ylabel('Energia (J)');
    title('Conservação de Energia');
    legend('Energia fornecida pela fonte', 'Energia no sistema', 'Location', 'best');
    grid on;
    
    erro_energia = max(abs(W_fonte - W_total_sistema));
    fprintf('Erro na conservação de energia: %.2e J\n', erro_energia);
end

%% 8. COMPARAÇÃO SÉRIE vs PARALELO (se tiver dados)
% ===================================================
comparar = input('\nComparar com circuito série equivalente? (1=Sim, 0=Não): ');
if comparar == 1
    % Para comparação justa, usar componentes com mesmo valor
    fprintf('\n--- Circuito Série Equivalente ---\n');
    fprintf('Usando mesmos valores de R, L, C\n');
    fprintf('Tensão da fonte: V = I * R = %.2f V\n', I*R);
    
    % Chamar função série ou resolver diretamente
    V_serie = I * R;
    i0_serie = 0;  % Assumindo condições iniciais zero
    vC0_serie = 0;
    
    % Resolver série numericamente
    f_serie = @(t, y) [
        (V_serie - R*y(1) - y(2))/L;
        y(1)/C
    ];
    
    y0_serie = [i0_serie; vC0_serie];
    [t_serie, y_serie] = ode45(f_serie, tspan, y0_serie, options);
    i_serie = y_serie(:, 1);
    vC_serie = y_serie(:, 2);
    
    figure('Position', [100, 100, 1000, 600]);
    
    % Comparar tensão no capacitor (paralelo) vs corrente no indutor (série)
    % Normalizar para comparação
    v_par_norm = v_num / max(abs(v_num));
    i_ser_norm = i_serie / max(abs(i_serie));
    
    subplot(2, 2, 1);
    plot(t_num, v_par_norm, 'b-', 'LineWidth', 2);
    hold on;
    plot(t_serie, i_ser_norm, 'r--', 'LineWidth', 2);
    xlabel('Tempo (s)');
    ylabel('Resposta Normalizada');
    title('Comparação: Paralelo (v) vs Série (i)');
    legend('v(t) - Paralelo', 'i(t) - Série', 'Location', 'best');
    grid on;
    
    % Comparar energia
    W_C_par = 0.5 * C * v_num.^2;
    W_L_ser = 0.5 * L * i_serie.^2;
    
    subplot(2, 2, 2);
    plot(t_num, W_C_par/max(W_C_par), 'b-', 'LineWidth', 2);
    hold on;
    plot(t_serie, W_L_ser/max(W_L_ser), 'r--', 'LineWidth', 2);
    xlabel('Tempo (s)');
    ylabel('Energia Normalizada');
    title('Comparação de Energia Armazenada');
    legend('W_C - Paralelo', 'W_L - Série', 'Location', 'best');
    grid on;
    
    % Comparar taxa de amortecimento
    alpha_serie = R/(2*L);
    alpha_paralelo = 1/(2*R*C);
    
    subplot(2, 2, [3, 4]);
    bar([1, 2], [alpha_serie, alpha_paralelo]);
    set(gca, 'XTickLabel', {'Série', 'Paralelo'});
    ylabel('Coeficiente de Amortecimento α (Np/s)');
    title('Comparação dos Coeficientes de Amortecimento');
    grid on;
    
    fprintf('\n=== COMPARAÇÃO SÉRIE vs PARALELO ===\n');
    fprintf('α_serie = %.4f Np/s\n', alpha_serie);
    fprintf('α_paralelo = %.4f Np/s\n', alpha_paralelo);
    fprintf('Razão α_paralelo/α_serie = %.4f\n', alpha_paralelo/alpha_serie);
end

fprintf('\n=== ANÁLISE CONCLUÍDA ===\n');