function test_incremental_damping()
%TEST_INCREMENTAL_DAMPING Focused numerical tests; no extra toolboxes.
c = 1500; % N s/m, from the supplied linear-damper example.
v = [300, 0, 1, 72, 9, 200]; % Deliberately unsorted, nonuniform mm/s.
for branch = ["Bump", "Rebound"]
    if branch == "Rebound", signedV = -v; else, signedV = v; end
    curve = incremental_damping(signedV, c*signedV/1000, branch);
    assert(all(diff(curve.v_mmps) > 0));
    assert(max(abs(curve.C_Nspm - c)) < 1e-9);
    assert(max(abs(curve.F_N - c*curve.v_mps)) < 1e-12);
end
% Hand-calculated secants: checks actual uneven coordinates and SI scaling.
curve = incremental_damping([30 0 10], [9 0 1], 'Bump');
assert(max(abs(curve.C_Nspm - [100; 300; 400])) < 1e-10);
mustFail(@() incremental_damping([0 1 1], [0 2 3], 'Bump'));
mustFail(@() incremental_damping([0 1 NaN], [0 2 3], 'Bump'));
mustFail(@() incremental_damping([0 1 2], [0 2], 'Bump'));
mustFail(@() incremental_damping([0 -1 -2], [0 -2 -3], 'Bump'));
mustFail(@() incremental_damping([0 1 2], [0 2 Inf], 'Bump'));
fprintf(['PASS: linear 1500 N s/m (both branches), paired sorting, ' ...
    'uneven-grid secants, SI conversion and invalid-input rejection.\n']);
end

function mustFail(action)
failed = false;
try
    action();
catch
    failed = true;
end
assert(failed, 'CV:Test', 'Invalid input was unexpectedly accepted.');
end
