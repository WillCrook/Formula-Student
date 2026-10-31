%% TBRe27 - Vehicle Dynamics Taster
% Rear-Heave Damper Modelling

%% Task 1 - Linear Damper Model

clear; clc; close all;

c = 1500;                  % Damping coefficient [Ns/m]
v = -0.3:0.01:0.3;        % Damper velocity [m/s]

F = c * v;                % Damping force [N]

figure;
plot(v * 1000, F, 'LineWidth', 2);
xlabel('Damper Velocity [mm/s]');
ylabel('Damper Force [N]');
title('Linear Damper Model');
grid on;