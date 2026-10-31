function build_report(outDir, comparisons, nearZero, audit)
%BUILD_REPORT Editable report source. Base MATLAB, no report toolbox.
% Wording lives here; output/report_draft.md is a generated editable text copy.
% All numbers below are selected from the computed tables, not typed results.
clicks = [0 4 8 11];
intro = ['Aim: calculate incremental damping C(v) = dF/dv for the TBRe26 ' ...
    'rear-heave Blue Valve damper and compare the available click settings. ' ...
    'The analysis covers all 12 settings in bump and rebound; settings 0, 4, 8 ' ...
    'and 11 are shown here to keep the figures readable.'];
method = sprintf(['Data and method: all %d supplied MATLAB curves match the workbook ' ...
    '(velocity in mm/s, force in N). Each branch is sorted with its forces, ' ...
    'then velocity is divided by 1000 before gradient(F,v_mps). Interior values ' ...
    'use (F(i+1)-F(i-1))/(v(i+1)-v(i-1)); endpoints use one-sided secants. ' ...
    'Branches are never differentiated across zero. No smoothing or points are removed.'],height(audit));
comparisonMethod = ['Comparison values below use linear interpolation of the supplied forces ' ...
    'and calculated nodal damping rates at signed shaft velocities. Queries of ' ...
    '25, 50, 100 and 200 mm/s lie within every branch; no extrapolation is used. ' ...
    'Displayed forces retain their source signs. Low-speed damping refers to shaft speed, not vehicle speed.'];

tableHeader = {'Branch / click','F at 50','C at 50','F at 200','C at 200'};
tableRows = cell(8,5);
r = 0;
for branch = ["Bump","Rebound"]
    for click = clicks
        r = r+1;
        a = getComparison(comparisons,branch,click,50);
        b = getComparison(comparisons,branch,click,200);
        tableRows(r,:) = {sprintf('%s / %d',branch,click),sprintf('%.1f',a.Force_N), ...
            sprintf('%.0f',a.C_Nspm),sprintf('%.1f',b.Force_N),sprintf('%.0f',b.C_Nspm)};
    end
end
b0 = getComparison(comparisons,"Bump",0,50);
b8 = getComparison(comparisons,"Bump",8,50);
b11 = getComparison(comparisons,"Bump",11,50);
r11 = getComparison(comparisons,"Rebound",11,50);
bh = getComparison(comparisons,"Bump",11,200);
b8h = getComparison(comparisons,"Bump",8,200);
b4h = getComparison(comparisons,"Bump",4,200);
r8 = getComparison(comparisons,"Rebound",8,50);
r8h = getComparison(comparisons,"Rebound",8,200);
r0 = getComparison(comparisons,"Rebound",0,50);
r0h = getComparison(comparisons,"Rebound",0,200);
b0h = getComparison(comparisons,"Bump",0,200);
finding1 = sprintf(['Click setting changes force and slope differently. At +50 mm/s, ' ...
    'bump force increases from %.1f N at click 0 to %.1f N at click 11. ' ...
    'Yet C is %.0f N s/m at click 8 and %.0f N s/m at click 11: the higher ' ...
    'force curve does not always have the larger local slope. At +200 mm/s, ' ...
    'bump C is %.0f, %.0f, %.0f and %.0f N s/m for clicks 0, 4, 8 and 11, ' ...
    'so the rates are much closer despite the force offsets.'], ...
    b0.Force_N,b11.Force_N,b8.C_Nspm,b11.C_Nspm,b0h.C_Nspm,b4h.C_Nspm,b8h.C_Nspm,bh.C_Nspm);
finding2 = sprintf(['Bump and rebound are asymmetric. At matched |v| = 50 mm/s, ' ...
    'click 11 gives %.1f N bump and %.1f N rebound magnitude; their local rates ' ...
    'are %.0f and %.0f N s/m. Rebound force and velocity are both negative, ' ...
    'so its positive dF/dv is expected in this convention. The plotted source ' ...
    'force is not a signed force-balance equation for the vehicle.'], ...
    b11.Force_N,abs(r11.Force_N),b11.C_Nspm,r11.C_Nspm);
finding3 = sprintf(['The response is nonlinear. From 50 to 200 mm/s, C falls from ' ...
    '%.0f to %.0f N s/m in bump and %.0f to %.0f N s/m in rebound at click 8. ' ...
    'That decrease is consistent with digressive behaviour over this interval. ' ...
    'It is not universal: click 0 rises from %.0f to %.0f N s/m in bump and ' ...
    '%.0f to %.0f N s/m in rebound over the same interval.'], ...
    b8.C_Nspm,b8h.C_Nspm,r8.C_Nspm,r8h.C_Nspm,b0.C_Nspm,b0h.C_Nspm,r0.C_Nspm,r0h.C_Nspm);

nearRows = cell(8,5);
r = 0;
for branch = ["Bump","Rebound"]
    for click = clicks
        r = r+1;
        t = nearZero(nearZero.Branch==branch & nearZero.Click==click,:);
        nearRows(r,:) = {sprintf('%s / %d',branch,click),sprintf('%+.5f',t.FirstVelocity_mmps), ...
            sprintf('%+.5f',t.FirstForce_N),sprintf('%.0f',t.ZeroToFirst_C_Nspm), ...
            sprintf('%.0f',t.Max_C_Nspm)};
    end
end
nB = nearZero(nearZero.Branch=="Bump" & nearZero.Click==11,:);
nR = nearZero(nearZero.Branch=="Rebound" & nearZero.Click==11,:);
nearText = sprintf(['Near zero, click 11 changes from (0,0) to (%+.5f mm/s, %+.5f N) ' ...
    'in bump and (%+.5f mm/s, %+.5f N) in rebound. The zero-to-first secants ' ...
    'are therefore %.0f and %.0f N s/m. Clicks 9-11 share these first points. ' ...
    'These are numerical consequences of the supplied coordinates, not evidence ' ...
    'of a confirmed friction, preload or valve mechanism.'], ...
    nB.FirstVelocity_mmps,nB.FirstForce_N,nR.FirstVelocity_mmps,nR.FirstForce_N, ...
    nB.ZeroToFirst_C_Nspm,nR.ZeroToFirst_C_Nspm);
limitations = ['Limits: differentiation amplifies data noise and rounding. The centred secant ' ...
    'is a finite difference across neighbours, not an exact continuous derivative; ' ...
    'with uneven spacing its accuracy depends on local curvature and sample spacing. ' ...
    'Interpolated C values are estimates from those nodal rates, not the exact ' ...
    'derivative of a piecewise-linear force lookup. The supplied curves do not ' ...
    'establish temperature, test repeatability, hysteresis or measurement uncertainty.'];
nextStep = ['Next step: ask for the dyno test conditions and the origin of the near-zero ' ...
    'points, ideally with repeat runs or denser data around the steep transitions. ' ...
    'More damping is not automatically better. An optimum rear-heave damping study ' ...
    'would also need effective modal mass, spring stiffness, motion ratios and a ' ...
    'defined vehicle-response objective; these curves alone cannot establish an optimum setup.'];
verification = ['Checks run in MATLAB R2025b: all 24 curves agree with the workbook to 1e-9; ' ...
    'paired sorting, finite values, duplicate rejection and source signs checked. ' ...
    'An uneven-grid linear damper returns 1500 N s/m in both branches, and a ' ...
    'hand-calculated nonlinear example checks the actual secant formula and SI scaling.'];

pdf = fullfile(outDir,'TBRe26_rear_heave_CV_report_draft.pdf');
% Pages are overwritten deterministically; only this generated report is removed.
if isfile(pdf), delete(pdf); end
f = reportPage(1,'Rear-heave damper: C-V curves','Task 1 | TBRe VDG onboarding | Draft for review');
textBox(f,[.075 .805 .85 .075],intro,10.5);
textBox(f,[.075 .697 .85 .104],method,10);
picture(f,outDir,'01_force_velocity.png',[.055 .447 .89 .235]);
textBox(f,[.075 .414 .85 .032], ...
    'Figure 1. Supplied F-V curves. Bump stays positive; rebound stays negative.',9);
textBox(f,[.075 .342 .85 .069],comparisonMethod,10);
drawTable(f,[.075 .113 .85 .218],tableHeader,tableRows,[.26 .185 .185 .185 .185]);
textBox(f,[.075 .069 .85 .035], ...
    'Table 1. Speeds are magnitudes [mm/s]: + in bump, - in rebound. F [N]; C [N s/m].',9);
savePage(f,pdf,false);

f = reportPage(2,'What changes with click setting?','Calculated rates | All source values retained');
picture(f,outDir,'02_cv_full.png',[.055 .660 .89 .230]);
textBox(f,[.075 .623 .85 .035], ...
    'Figure 2. Full linear scale, including the large zero-to-first-point rates.',9);
picture(f,outDir,'03_cv_zoom.png',[.055 .365 .89 .230]);
textBox(f,[.075 .325 .85 .038], ...
    'Figure 3. Same data; vertical display limited to 0-6000 N s/m. Values above the axis are clipped in this view only; Figure 2 retains them.',9);
textBox(f,[.075 .222 .85 .098],finding1,10);
textBox(f,[.075 .134 .85 .084],finding2,10);
textBox(f,[.075 .046 .85 .084],finding3,10);
savePage(f,pdf,true);

f = reportPage(3,'Near zero: treat the slope carefully','Limitations | Verification | One sensible next step');
picture(f,outDir,'04_near_zero_force.png',[.055 .671 .89 .220]);
textBox(f,[.075 .634 .85 .035], ...
    'Figure 4. Horizontal zoom of source F-V values. Markers identify supplied points.',9);
textBox(f,[.075 .538 .85 .092],nearText,10);
drawTable(f,[.075 .337 .85 .193], ...
    {'Branch / click','First v','First F','0-first C','Max C'},nearRows,[.26 .185 .185 .185 .185]);
textBox(f,[.075 .305 .85 .028], ...
    'Table 2. First nonzero source point: v [mm/s], F [N]. Both rate columns [N s/m].',9);
textBox(f,[.075 .214 .85 .086],limitations,9.6);
textBox(f,[.075 .125 .85 .085],nextStep,9.6);
textBox(f,[.075 .065 .85 .058],verification,8.8);
textBox(f,[.075 .024 .85 .038], ...
    ['Sources: supplied BLUE Valve Raw Data.xlsx and TBRe_BlueValve_Data.m; ' ...
    'onboarding slides dated 14 and 22 October 2026; recruitment email as quoted in the task brief.'],8);
savePage(f,pdf,true);

% Editable, searchable text companion; template/wording source is this .m file.
md = fullfile(outDir,'report_draft.md');
fid = fopen(md,'w');
assert(fid~=-1,'CV:Report','Cannot open editable report source.');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid,'# TBRe26 rear-heave damper: C-V curves\n\nDraft for review.\n\n%s\n\n%s\n\n%s\n\n',intro,method,comparisonMethod);
fprintf(fid,'![Supplied F-V curves](01_force_velocity.png)\n\n');
markdownTable(fid,tableHeader,tableRows);
fprintf(fid,'\nTable 1: speeds [mm/s], F [N], C [N s/m]; rebound queries are negative.\n\n');
fprintf(fid,'![Full C-V range](02_cv_full.png)\n\n![Vertical zoom](03_cv_zoom.png)\n\n');
fprintf(fid,'The vertical zoom clips values above 6000 N s/m for display only.\n\n%s\n\n%s\n\n%s\n\n',finding1,finding2,finding3);
fprintf(fid,'![Near-zero source points](04_near_zero_force.png)\n\n%s\n\n',nearText);
markdownTable(fid,{'Branch / click','First v [mm/s]','First F [N]','0-first C [N s/m]','Max C [N s/m]'},nearRows);
fprintf(fid,'\n%s\n\n%s\n\n%s\n\n',limitations,nextStep,verification);
fprintf(fid,['Sources: supplied workbook and MATLAB data function; onboarding slides ' ...
    '14 and 22 October 2026; recruitment email as quoted in the task brief.\n']);
end

function row = getComparison(t,branch,click,speed)
row = t(t.Branch==branch & t.Click==click & abs(t.Velocity_mmps)==speed,:);
assert(height(row)==1,'CV:ReportQuery','Report query missing or ambiguous.');
end

function f = reportPage(number,heading,subheading)
f = figure('Visible','off','Theme','light','Color','w','Units','inches','Position',[.5 .5 8.27 11.69]);
% Unfilled page boundary preserves margins without hiding embedded axes.
annotation(f,'rectangle',[0 0 1 1],'FaceColor','none','EdgeColor','white');
textBox(f,[.075 .942 .85 .030],heading,19,true);
textBox(f,[.075 .911 .85 .025],subheading,10);
annotation(f,'line',[.075 .925],[.905 .905],'Color',[.0 .36 .44],'LineWidth',1.5);
textBox(f,[.075 .003 .85 .017],sprintf('TBRe26 rear-heave | Draft - review before submission                                           %d / 3',number),8);
end

function textBox(f,pos,txt,fontSize,bold)
if nargin<5, bold=false; end
weight = 'normal';
if bold, weight='bold'; end
annotation(f,'textbox',pos,'String',txt,'Interpreter','none', ...
    'FontName','Helvetica','FontSize',fontSize,'FontWeight',weight, ...
    'EdgeColor','none','Margin',0,'VerticalAlignment','top','Color',[.10 .14 .18]);
end

function picture(f,outDir,name,pos)
ax = axes(f,'Position',pos);
image(ax,imread(fullfile(outDir,name)));
axis(ax,'image','off');
end

function drawTable(f,pos,headers,rows,widths)
n = size(rows,1)+1;
h = pos(4)/n;
y = pos(2)+pos(4);
for r = 1:n
    y = y-h;
    if r==1, bg=[.88 .93 .94]; elseif mod(r,2)==0, bg=[.96 .97 .98]; else, bg=[1 1 1]; end
    annotation(f,'rectangle',[pos(1) y pos(3) h],'FaceColor',bg,'EdgeColor','none');
    x = pos(1);
    for c = 1:numel(headers)
        if r==1, value=headers{c}; else, value=rows{r-1,c}; end
        textBox(f,[x+.007 y+.001 pos(3)*widths(c)-.010 h-.004],value,9,r==1);
        x = x+pos(3)*widths(c);
    end
end
end

function savePage(f,pdf,append)
exportgraphics(f,pdf,'ContentType','vector','BackgroundColor','white', ...
    'Append',append,'Units','inches','Width',8.27,'Height',11.69);
close(f);
end

function markdownTable(fid,headers,rows)
fprintf(fid,'| %s |\n',strjoin(headers,' | '));
fprintf(fid,'| %s |\n',strjoin(repmat({'---'},size(headers)),' | '));
for r = 1:size(rows,1), fprintf(fid,'| %s |\n',strjoin(rows(r,:),' | ')); end
end
