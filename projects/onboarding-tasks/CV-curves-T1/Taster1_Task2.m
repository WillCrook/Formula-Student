%% TBRe27 - Vehicle Dynamics Taster
% Rear-Heave Damper Modelling

%% Task 2 - Real Rear-Heave Damper

B = TBRe_BlueValve_Data();

click = 8;                 % Click setting (0 to 11)

bump = B.Bump{click + 1};
rebound = B.Rebound{click + 1};

figure;
plot(bump.v_mmps, bump.F_N, 'LineWidth', 2);
hold on;
plot(rebound.v_mmps, rebound.F_N, 'LineWidth', 2);

xlabel('Damper Velocity [mm/s]');
ylabel('Damper Force [N]');
title('TBRe Rear-Heave Damper');
legend('Bump', 'Rebound', 'Location', 'best');
grid on;


% Combine bump and rebound measurements
v_data = [rebound.v_mmps, bump.v_mmps(2:end)];
F_data = [rebound.F_N, bump.F_N(2:end)];

% Sort data by velocity
[v_data, order] = sort(v_data);
F_data = F_data(order);

% Prompt for velocity
v = input('Enter damper velocity (mm/s): ');

% Calculate damping force
F = interp1(v_data, F_data, v, 'linear');

% Display result
fprintf('Damping force = %.2f N\n', F);