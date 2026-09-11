clear; clc; close all;

load('SD1.mat');
D  = X.data;         
tl = X.label;         

n = size(D, 1);
[GBS, A, WJ, R, S] = Algorithm1_GBC(D);
[GB_final, C_B] = Algorithm2_3WD(GBS, A, WJ, R, S, D);
pred = Algorithm3_GBClustering(GB_final, C_B, A);
pred = pred(:);
nmi = calc_NMI(tl, pred);
pre = calc_Precision(tl, pred);
pur = calc_Purity(tl, pred);
gscatter(D(:, 1), D(:, 2), pred);


Result.D          = D;
Result.true_label = tl;
Result.pred_label = pred;
Result.initial_GB_num = numel(GBS);
Result.final_GB_num   = numel(GB_final);
Result.NMI       = nmi;
Result.Precision = pre;
Result.Purity    = pur;
Result.GBS       = GBS;
Result.GB_final  = GB_final;
Result.C_B       = C_B;


function v = calc_NMI(a, b)
% Normalized Mutual Information (symmetric formulation)
a = a(:); b = b(:);
n = numel(a);
ua = unique(a); ub = unique(b);
M = zeros(numel(ua), numel(ub));
for i = 1:numel(ua)
    for j = 1:numel(ub)
        M(i, j) = sum(a == ua(i) & b == ub(j));
    end
end
P  = M / n;
pa = sum(P, 2); pb = sum(P, 1);
Ha = -sum(pa .* log(pa + eps));
Hb = -sum(pb .* log(pb + eps));
Hxy = -sum(P(:) .* log(P(:) + eps));
v = (Ha + Hb - Hxy) / (0.5 * (Ha + Hb) + eps);
end

function v = calc_Precision(a, b)
% Average within-cluster majority accuracy
a = a(:); b = b(:);
ub = unique(b);
tot = 0;
for i = 1:numel(ub)
    sel = (b == ub(i));
    [~, ~, ic] = unique(a(sel));
    cnt = accumarray(ic, 1);
    tot = tot + max(cnt) / numel(sel);
end
v = tot / numel(ub);
end

function v = calc_Purity(a, b)
% Purity = (1/n) * sum over clusters of majority-class counts
a = a(:); b = b(:);
ub = unique(b);
tot = 0;
for i = 1:numel(ub)
    sel = (b == ub(i));
    [~, ~, ic] = unique(a(sel));
    cnt = accumarray(ic, 1);
    tot = tot + max(cnt);
end
v = tot / numel(a);
end
