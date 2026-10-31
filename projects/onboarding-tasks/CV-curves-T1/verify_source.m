function audit = verify_source(B, workbook)
%VERIFY_SOURCE Verify all 24 source curves against the supplied workbook.
% Workbook layout: row 1 curve names, rows 4/5 variables/units,
% numerical pairs start at row 6; A:X bump, AA:AX rebound.
headers = readcell(workbook, 'Sheet', 'Sheet1', 'Range', 'A1:AX5');
raw = readmatrix(workbook, 'Sheet', 'Sheet1', 'Range', 'A6:AX1000');
assert(isequal(B.Clicks, 0:11), 'CV:Clicks', 'Expected click settings 0:11.');
branches = {'Bump', 'Rebound'};
audit = table();
for b = 1:2
    for k = 1:numel(B.Clicks)
        col = 2*k - 1 + 26*(b-1);
        expected = sprintf('Primary_%s%d', branches{b}, B.Clicks(k));
        assert(strcmp(headers{1,col}, expected), 'CV:Header', 'Unexpected curve header.');
        assert(strcmp(headers{4,col}, 'Velocity') && ...
            strcmp(headers{4,col+1}, 'Force'), 'CV:Header', 'Unexpected variables.');
        assert(strcmp(headers{5,col}, '(mm/s)') && ...
            strcmp(headers{5,col+1}, '(N)'), 'CV:Units', 'Unexpected source units.');
        pair = raw(:, col:col+1);
        present = ~all(isnan(pair), 2);
        assert(~any(present(find(~present,1):end)), ...
            'CV:Gap', 'Unexpected blank row inside a curve.');
        pair = pair(present,:);
        assert(all(isfinite(pair), 'all'), 'CV:Source', 'Incomplete workbook pair.');
        source = B.(branches{b}){k};
        assert(source.Click == B.Clicks(k), 'CV:Click', 'Wrong click label.');
        fromFunction = [source.v_mmps(:), source.F_N(:)];
        assert(isequal(size(pair), size(fromFunction)), ...
            'CV:SourceSize', 'Workbook and MATLAB curve lengths differ.');
        maxError = max(abs(pair-fromFunction), [], 'all');
        assert(maxError < 1e-9, 'CV:SourceMismatch', 'Workbook and function differ.');
        audit = [audit; table(string(branches{b}), B.Clicks(k), size(pair,1), ...
            maxError, 'VariableNames', {'Branch','Click','Samples','MaxAbsError'})]; %#ok<AGROW>
    end
end
fprintf('PASS: all 24 curves match the workbook (tolerance 1e-9).\n');
fprintf('Verified workbook units: velocity [mm/s], force [N].\n');
end
