function curve = incremental_damping(v_mmps, force_N, branch)
%INCREMENTAL_DAMPING Differentiate one signed force-velocity branch.
% Inputs: shaft velocity [mm/s], source force [N], "Bump" or "Rebound".
% Output C_Nspm is incremental dF/dv [N s/m], not the secant F/v.
% No smoothing, point deletion, zero adjustment or extrapolation is used.
validateattributes(v_mmps, {'numeric'}, {'real','vector','finite','nonempty'});
validateattributes(force_N, {'numeric'}, {'real','vector','finite','nonempty'});
assert(numel(v_mmps) == numel(force_N) && numel(v_mmps) >= 3, ...
    'CV:Length', 'Need at least three paired velocity/force samples.');
branch = validatestring(branch, {'Bump','Rebound'});
[v, order] = sort(v_mmps(:));
F = force_N(:);
F = F(order); % Always sort paired measurements together.
assert(all(diff(v) > 0), 'CV:DuplicateVelocity', ...
    'Duplicate velocities require investigation; do not silently average them.');
if strcmp(branch, 'Bump')
    assert(all(v >= 0) && all(F >= 0), 'CV:Sign', 'Bump must retain +v/+F.');
else
    assert(all(v <= 0) && all(F <= 0), 'CV:Sign', 'Rebound must retain -v/-F.');
end
v_mps = v / 1000;
% MATLAB gradient: centred secant over neighbouring coordinates inside;
% one-sided adjacent-point secants at the two endpoints.
C = gradient(F, v_mps);
assert(all(isfinite(C)), 'CV:NonfiniteRate', 'Nonfinite damping rate.');
curve = struct('v_mmps', v, 'v_mps', v_mps, 'F_N', F, 'C_Nspm', C);
end
