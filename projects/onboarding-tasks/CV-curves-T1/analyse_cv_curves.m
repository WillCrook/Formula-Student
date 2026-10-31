%% TBRe26 rear-heave Blue Valve: force-velocity and incremental damping
% Run this script in MATLAB R2025b (base MATLAB only).
% Source files remain unchanged. All calculated outputs go to output/.
% Low-speed means damper SHAFT speed, not vehicle speed.
clear;
taskDir = fileparts(mfilename('fullpath'));
addpath(taskDir);
outDir = fullfile(taskDir, 'output');
if ~isfolder(outDir), mkdir(outDir); end
logFile = fullfile(outDir, 'verification.txt');
if isfile(logFile), delete(logFile); end
diary(logFile);
diaryCleanup = onCleanup(@() diary('off'));
fprintf('MATLAB %s\n', version);
fprintf('Analysis: TBRe26 rear-heave damper; all 12 clicks, both branches.\n');
test_incremental_damping();
B = TBRe_BlueValve_Data();
audit = verify_source(B, fullfile(taskDir, 'BLUE Valve Raw Data.xlsx'));

representativeClicks = [0 4 8 11]; % Presentation choice, not a team requirement.
querySpeeds_mmps = [25 50 100 200]; % Magnitudes; rebound queries stay negative.
branches = {'Bump','Rebound'};
curves = struct();
samples = table();
comparisons = table();
nearZero = table();
for b = 1:2
    branch = branches{b};
    curves.(branch) = cell(size(B.Clicks));
    direction = 3 - 2*b; % +1 bump, -1 rebound.
    for k = 1:numel(B.Clicks)
        click = B.Clicks(k);
        raw = B.(branch){k};
        curve = incremental_damping(raw.v_mmps, raw.F_N, branch);
        curves.(branch){k} = curve;
        n = numel(curve.v_mmps);
        samples = [samples; table(repmat(string(branch),n,1), ...
            repmat(click,n,1), curve.v_mmps, curve.v_mps, curve.F_N, ...
            curve.C_Nspm, 'VariableNames', ...
            {'Branch','Click','Velocity_mmps','Velocity_mps','Force_N','C_Nspm'})]; %#ok<AGROW>
        queries = direction * querySpeeds_mmps(:);
        assert(all(queries >= min(curve.v_mmps) & queries <= max(curve.v_mmps)), ...
            'CV:Range', 'Comparison would extrapolate beyond the supplied range.');
        % Interpolate the computed nodal slopes, not F/v or a fitted curve.
        force = interp1(curve.v_mmps, curve.F_N, queries, 'linear');
        rate = interp1(curve.v_mmps, curve.C_Nspm, queries, 'linear');
        assert(all(isfinite(force)) && all(isfinite(rate)));
        comparisons = [comparisons; table(repmat(string(branch),numel(queries),1), ...
            repmat(click,numel(queries),1), queries, force, rate, ...
            'VariableNames', {'Branch','Click','Velocity_mmps','Force_N','C_Nspm'})]; %#ok<AGROW>
        [~, order] = sort(abs(curve.v_mmps));
        assert(curve.v_mmps(order(1)) == 0 && curve.F_N(order(1)) == 0, ...
            'CV:Origin', 'Expected source origin; investigate before changing it.');
        j = order(2);
        firstSlope = curve.F_N(j) / curve.v_mps(j); % Exact zero-to-first secant.
        nearZero = [nearZero; table(string(branch),click,n,curve.v_mmps(j), ...
            curve.F_N(j),firstSlope,min(curve.C_Nspm),max(curve.C_Nspm), ...
            min(curve.v_mmps),max(curve.v_mmps), 'VariableNames', ...
            {'Branch','Click','Samples','FirstVelocity_mmps','FirstForce_N', ...
            'ZeroToFirst_C_Nspm','Min_C_Nspm','Max_C_Nspm', ...
            'MinVelocity_mmps','MaxVelocity_mmps'})]; %#ok<AGROW>
    end
end
writetable(audit, fullfile(outDir,'source_audit.csv'));
writetable(samples, fullfile(outDir,'curves.csv'));
writetable(comparisons, fullfile(outDir,'comparisons.csv'));
writetable(nearZero, fullfile(outDir,'near_zero.csv'));
save(fullfile(outDir,'analysis.mat'), 'curves','comparisons','nearZero','audit', ...
    'representativeClicks','querySpeeds_mmps');
fprintf('PASS: %d paired samples; finite outputs; no duplicate velocities.\n',height(samples));
fprintf('PASS: signed branches sorted separately; comparison queries within every curve.\n');
fprintf('No smoothing, extrapolation, point deletion or zero adjustment.\n');
disp(comparisons(ismember(comparisons.Click,representativeClicks) & ...
    abs(comparisons.Velocity_mmps)==50,:));
disp(nearZero(ismember(nearZero.Click,representativeClicks),:));

% Colours are consistent across all representative-setting figures.
colours = [0.00 0.32 0.56; 0.80 0.29 0.00; 0.00 0.49 0.35; 0.53 0.25 0.65];
figures = struct();
figures.fv = twoBranchFigure(curves, representativeClicks, colours, 'F_N', ...
    'Force [N]', 'Force-velocity characteristics');
exportFigure(figures.fv, outDir, '01_force_velocity');

figures.cvFull = twoBranchFigure(curves, representativeClicks, colours, 'C_Nspm', ...
    'Incremental damping [N s/m]', 'C-V curves: full range');
exportFigure(figures.cvFull, outDir, '02_cv_full');

figures.cvZoom = twoBranchFigure(curves, representativeClicks, colours, 'C_Nspm', ...
    'Incremental damping [N s/m]', 'C-V curves: vertical zoom (0-6000 N s/m)');
axs = findobj(figures.cvZoom, 'Type','axes');
for ax = axs'
    ylim(ax,[0 6000]);
end
exportFigure(figures.cvZoom, outDir, '03_cv_zoom');

figures.nearZero = twoBranchFigure(curves, representativeClicks, colours, 'F_N', ...
    'Force [N]', 'Near-zero source force-velocity points (markers = supplied samples)');
axs = findobj(figures.nearZero, 'Type','axes');
for ax = axs'
    if contains(ax.Title.String,'Bump'), xlim(ax,[0 8]); else, xlim(ax,[-8 0]); end
    lines = findobj(ax,'Type','line');
    set(lines,'Marker','.');
end
exportFigure(figures.nearZero, outDir, '04_near_zero_force');

figures.allClicks = figure('Visible','off','Theme','light','Color','w','Position',[50 50 1200 780]);
layout = tiledlayout(figures.allClicks,2,2,'TileSpacing','compact','Padding','compact');
allColours = turbo(12)*0.8; % Darken the light colours for white backgrounds.
for b = 1:2
    for row = 1:2
        ax = nexttile(layout,2*(row-1)+b);
        drawBranch(ax,curves.(branches{b}),B.Clicks,allColours, ...
            pickField(row),row==2);
        title(ax,branches{b});
        if row==1, ylabel(ax,'Force [N]'); else, ylabel(ax,'Incremental damping [N s/m] (log scale)'); end
        if row==1, legend(ax,'Location','northwest','NumColumns',3,'FontSize',10); end
    end
end
title(layout,'All 12 supplied click settings; full velocity range');
exportFigure(figures.allClicks,outDir,'05_all_clicks');

% Report text and figures are assembled in MATLAB; no report toolbox needed.
build_report(outDir,comparisons,nearZero,audit);
fprintf('PASS: exported five figure sets, data tables and three-page draft report.\n');
fprintf('Outputs: %s\n',outDir);
diary off;
clear diaryCleanup;
% Close only figures created by this script.
names = fieldnames(figures);
for k = 1:numel(names), close(figures.(names{k})); end

function f = twoBranchFigure(curves,clicks,colours,field,yLabel,heading)
f = figure('Visible','off','Theme','light','Color','w','Position',[50 50 1200 430]);
layout = tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');
for b = 1:2
    branches = {'Bump','Rebound'};
    ax = nexttile(layout);
    drawBranch(ax,curves.(branches{b}),clicks,colours,field,false);
    title(ax,[branches{b} ' (source signs)']);
    ylabel(ax,yLabel);
    legend(ax,'Location','northeast','FontSize',13);
    if strcmp(field,'C_Nspm'), ylim(ax,[0 inf]); end
end
title(layout,heading,'FontSize',14,'FontWeight','bold');
end

function drawBranch(ax,branchCurves,clicks,colours,field,logY)
hold(ax,'on');
for k = 1:numel(clicks)
    curve = branchCurves{clicks(k)+1};
    plot(ax,curve.v_mmps,curve.(field),'Color',colours(k,:), ...
        'LineWidth',1.6,'DisplayName',sprintf('Click %d',clicks(k)));
end
if logY, set(ax,'YScale','log'); end
xlabel(ax,'Signed shaft velocity [mm/s]');
grid(ax,'on');
ax.FontSize = 14;
ax.Box = 'on';
end

function field = pickField(row)
if row==1, field='F_N'; else, field='C_Nspm'; end
end

function exportFigure(f,outDir,name)
exportgraphics(f,fullfile(outDir,[name '.png']),'Resolution',220);
exportgraphics(f,fullfile(outDir,[name '.svg']),'ContentType','vector');
end
