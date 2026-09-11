function [GBS_star, C_B] = Algorithm2_3WD(GBS0, A, WJ, R, S, X)
if nargin < 6, X = []; end
stable  = {};         
working = GBS0;        
Scur    = S;
iter    = 0;
maxIter = 50;

while ~isempty(working) && iter < maxIter
    iter = iter + 1;
    alpha = prctile(Scur, 75);
    beta  = prctile(Scur, 25);
    next    = {};     
    Snext   = [];
    changed = false;

    for i = 1:numel(working)
        s = Scur(i);
        if s >= alpha
            stable{end + 1} = working{i};
            changed = true;
        elseif s <= beta
                     if numel(working{i}) > 1
                [s1, s2] = split_ball(working{i}, X);
                next{end + 1}  = s1;
                next{end + 1}  = s2;
                Snext(end + 1) = eval_score(s1, X);
                Snext(end + 1) = eval_score(s2, X);
            else
                stable{end + 1} = working{i};
            end
            changed = true;
        else
          
            next{end + 1}  = working{i};
            Snext(end + 1) = s;
        end
    end
    if ~changed
        stable = [stable, next];
        break;
    end
    working = next;
    Scur    = Snext;
end

GBS_star = [stable, working];
m = numel(GBS_star);
if ~isempty(X)
    C_B = zeros(m, size(X, 2));
    for i = 1:m
        C_B(i, :) = mean(X(GBS_star{i}, :), 1);
    end
else
    C_B = zeros(m, 1);
    for i = 1:m
        [~, t] = max(sum(A(GBS_star{i}, :), 2));
        C_B(i) = GBS_star{i}(t);
    end
end
end


function [s1, s2] = split_ball(g, X)

if isempty(X) || numel(g) <= 2
    h  = floor(numel(g) / 2);
    s1 = g(1:h);
    s2 = g(h + 1:numel(g));
    if isempty(s1), s1 = s2(1); s2(1) = []; end
    if isempty(s2), s2 = s1(end); s1(end) = []; end
    return;
end
[~, cidx] = kmeans(X(g, :), 2, 'MaxIter', 100, 'Replicates', 1, 'Start', 'sample');
s1 = g(cidx == 1);
s2 = g(cidx == 2);
end

function sc = eval_score(g, X)
if isempty(X) || numel(g) < 2
    sc = 1;
    return;
end
c  = mean(X(g, :), 1);
d  = sqrt(sum((X(g, :) - c).^2, 2));
pd = pdist(X(g, :));
sc = (1 / (1 + mean(pd))) / (1 + max(d));
end
