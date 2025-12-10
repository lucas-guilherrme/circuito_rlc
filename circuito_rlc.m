%% ============================================================
% ANÁLISE DO TRANSIENTE EM CIRCUITOS RLC SÉRIE
% Baseado no Capítulo 8 - Fundamentos de Circuitos Elétricos (Sadiku)
% ============================================================

clear all; close all; clc;
fprintf('=== ANÁLISE DO TRANSIENTE - CIRCUITO RLC SÉRIE ===\n\n');

%% 1. PARÂMETROS DO CIRCUITO
% ===========================
R = input('Resistência (Ω): ');
L = input('Indutância (H): ');
C = input('Capacitância (F): ');
V = input('Tensão da fonte (V): ');

% Condições iniciais
fprintf('\n--- Condições Iniciais ---\n');
i0 = input('Corrente inicial no indutor i(0) (A): ');
vC0 = input('Tensão inicial no capacitor vC(0) (V): ');

% Tempo de simulação
t_final = input('\nTempo final de simulação (s): ');

%% 2. CÁLCULO DOS PARÂMETROS CARACTERÍSTICOS
% ===========================================
alpha = R/(2*L);                    % Coeficiente de amortecimento (Np/s)
omega0 = 1/sqrt(L*C);               % Frequência natural não amortecida (rad/s)
omega_d = sqrt(omega0^2 - alpha^2); % Frequência amortecida (rad/s)

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

% Fator de qualidade e tempo de acomodação
if tipo == 3
    Q = omega0/(2*alpha);  % Fator de qualidade
    t_settle_2 = 4/alpha;  % Tempo de acomodação (2%)
    t_settle_5 = 3/alpha;  % Tempo de acomodação (5%)
    fprintf('Fator de qualidade Q = %.4f\n', Q);
    fprintf('Tempo de acomodação (2%%) = %.4f s\n', t_settle_2);
    fprintf('Tempo de acomodação (5%%) = %.4f s\n', t_settle_5);
end

%% 3. SOLUÇÃO ANALÍTICA DA EQUAÇÃO DIFERENCIAL
% =============================================
syms i(t)

% Equação diferencial: L*di'' + R*di' + (1/C)*i = V'
% Para degrau em t=0: V' = 0 (após t=0+)
eqn = L*diff(i,t,2) + R*diff(i,t) + (1/C)*i == 0;

% Condições iniciais
cond1 = i(0) == i0;
% Para di/dt(0): vL(0) = L*di/dt(0) = V - R*i0 - vC0
di0 = (V - R*i0 - vC0)/L;
cond2 = diff(i) == di0;

% Solução geral
if tipo == 1  % Superamortecido
    % Raízes: s1, s2 = -α ± √(α² - ω₀²)
    s1 = -alpha + sqrt(alpha^2 - omega0^2);
    s2 = -alpha - sqrt(alpha^2 - omega0^2);
    
    % Forma geral: i(t) = A1*exp(s1*t) + A2*exp(s2*t)
    syms A1 A2
    i_sym = A1*exp(s1*t) + A2*exp(s2*t);
    
    % Aplicar condições iniciais
    eq1 = subs(i_sym, t, 0) == i0;
    eq2 = subs(diff(i_sym, t), t, 0) == di0;
    
    [A1_sol, A2_sol] = solve([eq1, eq2], [A1, A2]);
    i_analitico = subs(i_sym, [A1, A2], [A1_sol, A2_sol]);
    
elseif tipo == 2  % Criticamente amortecido
    % Forma geral: i(t) = (A1 + A2*t)*exp(-α*t)
    syms A1 A2
    i_sym = (A1 + A2*t)*exp(-alpha*t);
    
    % Aplicar condições iniciais
    eq1 = subs(i_sym, t, 0) == i0;
    eq2 = subs(diff(i_sym, t), t, 0) == di0;
    
    [A1_sol, A2_sol] = solve([eq1, eq2], [A1, A2]);
    i_analitico = subs(i_sym, [A1, A2], [A1_sol, A2_sol]);
    
else  % Subamortecido
    % Forma geral: i(t) = exp(-α*t)*[B1*cos(ω_d*t) + B2*sin(ω_d*t)]
    syms B1 B2
    i_sym = exp(-alpha*t)*(B1*cos(omega_d*t) + B2*sin(omega_d*t));
    
    % Aplicar condições iniciais
    eq1 = subs(i_sym, t, 0) == i0;
    eq2 = subs(diff(i_sym, t), t, 0) == di0;
    
    [B1_sol, B2_sol] = solve([eq1, eq2], [B1, B2]);
    i_analitico = subs(i_sym, [B1, B2], [B1_sol, B2_sol]);
end

% Simplificar a solução
i_analitico = simplify(i_analitico);
fprintf('\n=== SOLUÇÃO ANALÍTICA ===\n');
fprintf('i(t) = %s\n', char(i_analitico));

%% 4. SOLUÇÃO NUMÉRICA (ODE45)
% =============================
% Sistema de equações de primeira ordem:
% di/dt = vL/L
% dvC/dt = i/C
% onde vL = V - R*i - vC

% Definir o sistema de EDOs
f = @(t, y) [
    (V - R*y(1) - y(2))/L;  % di/dt
    y(1)/C                  % dvC/dt
];

% Vetor de tempo
tspan = [0 t_final];
y0 = [i0; vC0];  % Condições iniciais [i(0); vC(0)]

% Resolver numericamente
options = odeset('RelTol', 1e-8, 'AbsTol', 1e-10);
[t_num, y_num] = ode45(f, tspan, y0, options);

i_num = y_num(:, 1);
vC_num = y_num(:, 2);

% Calcular outras grandezas
vR_num = R * i_num;
vL_num = V - vR_num - vC_num;
pR_num = i_num.^2 * R;  % Potência dissipada no resistor

%% 5. VALIDAÇÃO: Comparação entre analítico e numérico
% ====================================================
% Avaliar solução analítica nos pontos numéricos
i_analitico_num = double(subs(i_analitico, t, t_num));

% Calcular erro
erro = max(abs(i_num - i_analitico_num));
fprintf('\n=== VALIDAÇÃO ===\n');
fprintf('Erro máximo entre soluções analítica e numérica: %.2e A\n', erro);

%% 6. VISUALIZAÇÃO DOS RESULTADOS
% ================================
figure();

% Subplot 1: Corrente no circuito
subplot(3, 3, [1, 2]);
plot(t_num, i_num, 'b-', 'LineWidth', 2);
hold on;
plot(t_num, i_analitico_num, 'r--', 'LineWidth', 1.5);
xlabel('Tempo (s)', 'FontSize', 10);
ylabel('Corrente i(t) (A)', 'FontSize', 10);
title('Resposta Transitória - Corrente no Circuito', 'FontSize', 12);
legend('Solução Numérica', 'Solução Analítica', 'Location', 'best');
grid on;

% Adicionar linhas de tempo característico se subamortecido
if tipo == 3
    % Tempo de pico
    t_peak = pi/omega_d;
    i_peak = double(subs(i_analitico, t, t_peak));
    line([t_peak t_peak], [0 i_peak], 'Color', 'g', 'LineStyle', ':', 'LineWidth', 1);
    text(t_peak, i_peak/2, sprintf('t_p = %.4f s', t_peak), 'Color', 'g');
end

% Subplot 2: Tensões nos componentes
subplot(3, 3, [4, 5]);
plot(t_num, vR_num, 'r-');
hold on;
plot(t_num, vL_num, 'g-');
plot(t_num, vC_num, 'b-');
xlabel('Tempo (s)');
ylabel('Tensão (V)');
title('Tensões nos Componentes');
legend('V_R(t)', 'V_L(t)', 'V_C(t)', 'Location', 'best');
grid on;

% Subplot 3: Potência dissipada no resistor
subplot(3, 3, [7, 8]);
plot(t_num, pR_num, 'm-');
xlabel('Tempo (s)');
ylabel('Potência (W)');
title('Potência Dissipada no Resistor');
grid on;

% Subplot 4: Diagrama de fases (espaço de estados)
subplot(3, 3, [3, 6, 9]);
plot(vC_num, i_num, 'b-');
hold on;
plot(vC0, i0, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
xlabel('Tensão no Capacitor v_C (V)', 'FontSize', 10);
ylabel('Corrente i (A)', 'FontSize', 10);
title('Diagrama de Fases (Espaço de Estados)', 'FontSize', 12);
grid on;

% Adicionar informações no gráfico
annotation('textbox', [0.15, 0.85, 0.3, 0.1], 'String', ...
    sprintf('R = %.2f Ω\nL = %.4f H\nC = %.4f F\nV = %.2f V', R, L, C, V), ...
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
% Calcular energias armazenadas
W_L = 0.5 * L * i_num.^2;      % Energia no indutor
W_C = 0.5 * C * vC_num.^2;     % Energia no capacitor
W_total = W_L + W_C;           % Energia total armazenada

% Energia dissipada no resistor (integral da potência)
W_R = cumtrapz(t_num, pR_num);

figure('Position', [100, 100, 1000, 600]);

% Gráfico de energias
subplot(2, 2, 1);
plot(t_num, W_L, 'g-', 'LineWidth', 2);
hold on;
plot(t_num, W_C, 'b-', 'LineWidth', 2);
plot(t_num, W_total, 'r-', 'LineWidth', 2);
xlabel('Tempo (s)');
ylabel('Energia (J)');
title('Energia Armazenada nos Componentes');
legend('W_L (indutor)', 'W_C (capacitor)', 'W_{total}', 'Location', 'best');
grid on;

% Energia dissipada
subplot(2, 2, 2);
plot(t_num, W_R, 'm-', 'LineWidth', 2);
xlabel('Tempo (s)');
ylabel('Energia (J)');
title('Energia Dissipada no Resistor');
grid on;

% Conservação de energia (para verificação)
if abs(i0) < 1e-6 && abs(vC0) < 1e-6
    % Se condições iniciais zero, energia fornecida pela fonte
    % Trabalho da fonte = ∫V*i dt
    W_fonte = cumtrapz(t_num, V * i_num);
    W_total_sistema = W_total + W_R;
    
    subplot(2, 2, [3, 4]);
    plot(t_num, W_fonte, 'k-', 'LineWidth', 2);
    hold on;
    plot(t_num, W_total_sistema, 'r--', 'LineWidth', 2);
    xlabel('Tempo (s)');
    ylabel('Energia (J)');
    title('Conservação de Energia');
    legend('Energia fornecida pela fonte', 'Energia no sistema (W+L+R)', 'Location', 'best');
    grid on;
    
    % Verificar conservação
    erro_energia = max(abs(W_fonte - W_total_sistema));
    fprintf('Erro na conservação de energia: %.2e J\n', erro_energia);
end

%% 8. ANÁLISE SENSIBILIDADE - Variação de R
% =========================================
fprintf('\n=== ANÁLISE DE SENSIBILIDADE ===\n');
fprintf('Variação da resposta com diferentes valores de R:\n');

figure('Position', [100, 100, 900, 600]);
R_values = [0.5*R, R, 2*R, 5*R];
colors = ['r', 'b', 'g', 'm'];
styles = {'-', '--', ':', '-.'};

for k = 1:length(R_values)
    R_test = R_values(k);
    alpha_test = R_test/(2*L);
    
    % Classificar
    if alpha_test > omega0
        tipo_test = 1;
    elseif abs(alpha_test - omega0) < 1e-6
        tipo_test = 2;
    else
        tipo_test = 3;
    end
    
    % Resolver numericamente
    f_test = @(t, y) [
        (V - R_test*y(1) - y(2))/L;
        y(1)/C
    ];
    
    [t_test, y_test] = ode45(f_test, tspan, y0, options);
    
    % Plot
    plot(t_test, y_test(:, 1), 'Color', colors(k), ...
        'LineStyle', styles{mod(k, length(styles))+1}, 'LineWidth', 2);
    hold on;
    
    % Legenda
    if k == 1
        leg_text = sprintf('R = %.1fΩ (Subamortecido)', R_test);
    elseif k == 2
        leg_text = sprintf('R = %.1fΩ (Original)', R_test);
    elseif tipo_test == 1
        leg_text = sprintf('R = %.1fΩ (Superamortecido)', R_test);
    else
        leg_text = sprintf('R = %.1fΩ', R_test);
    end
    
    legend_str{k} = leg_text;
end

xlabel('Tempo (s)');
ylabel('Corrente i(t) (A)');
title('Sensibilidade: Variação da Resposta com R');
legend(legend_str, 'Location', 'best');
grid on;

%% 9. EXPORTAÇÃO DE DADOS
% =======================
% Opção para exportar resultados
exportar = input('\nExportar dados para arquivo CSV? (1=Sim, 0=Não): ');
if exportar == 1
    nome_arquivo = input('Nome do arquivo (sem extensão): ', 's');
    dados = [t_num, i_num, vR_num, vL_num, vC_num];
    header = {'Tempo(s)', 'Corrente(A)', 'V_R(V)', 'V_L(V)', 'V_C(V)'};
    
    % Criar tabela e exportar
    T = array2table(dados, 'VariableNames', header);
    writetable(T, [nome_arquivo '.csv']);
    fprintf('Dados exportados para %s.csv\n', nome_arquivo);
    
    % Salvar também a solução analítica
    fileID = fopen([nome_arquivo '_sol_analitica.txt'], 'w');
    fprintf(fileID, 'Solução analítica para i(t):\n');
    fprintf(fileID, 'i(t) = %s\n', char(i_analitico));
    fprintf(fileID, '\nParâmetros:\n');
    fprintf(fileID, 'R = %.4f Ω\n', R);
    fprintf(fileID, 'L = %.4f H\n', L);
    fprintf(fileID, 'C = %.4f F\n', C);
    fprintf(fileID, 'V = %.4f V\n', V);
    fprintf(fileID, 'α = %.4f Np/s\n', alpha);
    fprintf(fileID, 'ω₀ = %.4f rad/s\n', omega0);
    fclose(fileID);
end

fprintf('\n=== ANÁLISE CONCLUÍDA ===\n');