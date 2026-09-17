%  Run:  >> jansen_walking_sim
%  (single script, local functions need R2016b or newer)
clear; clc; close all

%% 1. Target foot-tip path (supplied CSV)
T  = readtable('target_foot_tip_path.csv');
xt = T.x_mm(:).';   yt = T.y_mm(:).';
fprintf('TARGET  stride = %.1f mm   lift = %.1f mm\n', ...
        max(xt)-min(xt), max(yt)-min(yt));

%% 2. Jansen "holy numbers" (link lengths, arbitrary units)
%  a,l : ground link  O=(0,0) crank pivot, B=(-a,-l) second fixed pivot
%  m   : crank        b,c : rockers      j,k : couplers
%  d,e : rigid triangle BCE on rocker b   f : link EF
%  g,h,i : rigid foot triangle DFG ;  toe = G
a=38; l=7.8; m=15; b=41.5; c=39.3; d=40.1; e=55.8; ...
f=39.4; g=36.7; h=65.7; i=49; j=50; k=61.9;

th = deg2rad(0:359);                    % one full input revolution, 1 deg steps
A  = [m*cos(th); m*sin(th)];            % crank pin A (driven)
Bv = [-a; -l];                          % fixed pivot B

% --- assembly: circle intersections (branch choice at each step) ---
C = circInt(Bv,b, A,j, +1);             % upper 4-bar  (BC=b, AC=j)
D = circInt(Bv,c, A,k, -1);             % lower 4-bar  (BD=c, AD=k)
E = circInt(Bv,d, C,e, +1);             % rigid triangle BCE (BE=d, CE=e)
F = circInt(D ,g, E,f, +1);             % link DF=g, EF=f
G = circInt(D ,i, F,h, +1);             % foot triangle DFG (DG=i, FG=h) -> toe
% Branch table (fixed for whole cycle -> no branch defect):
%   C:+1  D:-1  E:+1  F:+1  G:+1   (sign = side of the chord, see circInt)

Px = G(1,:);  Py = G(2,:);              % foot-tip (toe) path, raw units

%% 3. Scale uniformly so peak lift = 200 mm  (Problem step 4)
s  = 200 / (max(Py)-min(Py));
Xs = s*Px;  Ys = s*Py - min(s*Py);      % stance phase on y = 0
stride = max(Xs)-min(Xs);
fprintf('SCALE s = %.6f   ->  stride = %.1f mm (target %.1f mm)\n', ...
        s, stride, max(xt)-min(xt));

%% 4. Error vs target (centered x, phase-aligned closed curves)
xc = Xs - mean(Xs);  tc = xt - mean(xt);
best = inf;
for sh = 0:359
    r  = mod((0:359)+sh, 360) + 1;
    best = min(best, sqrt(mean((xc(r)-tc).^2 + (Ys-yt).^2)));
end
fprintf('RMSE vs target (centered, best phase) = %.3f mm\n', best);

%% 5. Grubler-Kutzbach DOF proof
%  Links (n = 8): ground | crank m | rocker b+triangle(bde) | rocker c | ...
%                coupler j | coupler k | link f | foot triangle DFG
%  Joints (j1 = 10 revolute, counted with multiplicity):
%    O : ground-m            (1)
%    B : ground-b, ground-c  (2,3)
%    A : m-j, m-k            (4,5)   <- ternary pin counts twice
%    C : b-j                 (6)
%    D : c-k, k-foot         (7,8)   <- ternary pin counts twice
%    E : (bde)-f             (9)
%    F : f-foot              (10)
n = 8;  j1 = 10;  j2 = 0;
DOF = 3*(n-1) - 2*j1 - j2;              % NOTE: 'DOF', not 'F' - F holds joint positions!
fprintf('GRUBLER  F = 3(%d-1) - 2*%d - %d = %d  -> exactly ONE DOF\n', n, j1, j2, DOF);

%% 6. Common frame: shift whole mechanism so toe sits ON the target path
%  dy lifts the raw mechanism so toe bottom = 0 ;  dx aligns toe left edge with target
dxm = min(xt) - min(Xs);
dym = -min(s*Py);
off = [dxm; dym];
fprintf('Mechanism frame offset: dx = %.1f mm, dy = %.1f mm\n', dxm, dym);

%% 7. Overlay plot: simulation vs supplied target
figure('Color','w','Position',[80 80 950 420])
plot(xt, yt, 'k--', 'LineWidth', 2); hold on; grid on; axis equal
plot(Xs + dxm, Ys, 'r-', 'LineWidth', 1.2)
plot(Xs(1)+dxm, Ys(1), 'ro', 'MarkerFaceColor','r')
text(Xs(1)+dxm, Ys(1), '  start', 'FontSize', 9)
xlabel('x (mm)'); ylabel('y (mm)')
title(sprintf('Jansen toe curve (s = %.4f) vs target - RMSE %.2f mm', s, best))
legend('target foot-tip path (CSV)', 'Jansen simulation', 'Location', 'best')

%% 8. Mechanism sketch at 8 crank angles (mechanism in SAME frame as target)
%  Point layout inside poseAt:  [1]=O  [2]=A  [3]=B  [4]=C  [5]=D  [6]=E  [7]=F  [8]=G(toe)
segs = {[1 3],[1 2],[3 4],[2 4],[3 5],[2 5],[3 6],[4 6],[6 7],[5 7],[5 8],[7 8]};
%      ground  crank  b:B-C  j:A-C  c:B-D  k:A-D  d:B-E  e:C-E  f:E-F  g:D-F  i:D-G  h:F-G
figure('Color','w','Position',[80 560 700 560])
plot(xt, yt, 'k--', 'LineWidth', 1.4); hold on; grid on; axis equal
plot(Xs + dxm, Ys, 'r-', 'LineWidth', 1)
cmap = lines(8);
for q = 1:8
    idx = (q-1)*45 + 1;
    P = poseAt(idx, A, Bv, C, D, E, F, G, s) + off;
    for t = 1:numel(segs)
        plot(P(1,segs{t}), P(2,segs{t}), '-', 'Color', cmap(q,:), 'LineWidth', 1.6)
    end
    plot(P(1,8), P(2,8), 's', 'Color', cmap(q,:), 'MarkerFaceColor', cmap(q,:))
end
% ground line under the stance phase
plot([min(xt)-40 max(xt)+40], [0 0], 'k-', 'LineWidth', 2)
text(min(xt)-30, -25, 'ground')
xlabel('x (mm)'); ylabel('y (mm)')
title('Jansen linkage at 8 crank angles - foot on target path')

%% 9. Animation of one input revolution (same frame)
figure('Color','w','Position',[1080 80 620 500])
plot(xt, yt, 'k--', 'LineWidth', 1.5); hold on; grid on; axis equal
hlinks = plot(zeros(2,numel(segs)), zeros(2,numel(segs)), '-o', 'LineWidth', 2);
htrail = plot(nan, nan, 'r-', 'LineWidth', 1.5);
hground = plot([min(xt)-40 max(xt)+40], [0 0], 'k-', 'LineWidth', 2);
legend('target', 'links', 'toe trail', 'ground')
xlim([min(xt)-100 max(xt)+100]); ylim([-100 1300])
for idx = 1:3:360          % every 3 deg -> 120 frames
    P = poseAt(idx, A, Bv, C, D, E, F, G, s) + off;
    for t = 1:numel(segs)
        set(hlinks(t), 'XData', P(1,segs{t}), 'YData', P(2,segs{t}));
    end
    set(htrail, 'XData', Xs(1:idx)+dxm, 'YData', Ys(1:idx))
    drawnow
end

%% local functions ------------------------------------------------------
function P = poseAt(idx, A, Bv, C, D, E, F, G, s)
%POSEAT  All joint positions at crank index idx, scaled by s.
%   Columns: [1]=O  [2]=A  [3]=B  [4]=C  [5]=D  [6]=E  [7]=F  [8]=G(toe)
    P = zeros(2, 8);
    P(:,1) = [0; 0];
    P(:,2) = A(:,idx);
    P(:,3) = Bv(:);
    P(:,4) = C(:,idx);
    P(:,5) = D(:,idx);
    P(:,6) = E(:,idx);
    P(:,7) = F(:,idx);
    P(:,8) = G(:,idx);
    P = P * s;
end

function P = circInt(P1, r1, P2, r2, side)
%CIRCINT  Intersection of two circles: point at r1 from P1 AND r2 from P2.
%   P1 may be a fixed 2-vector or a 2xN array; P2 is 2xN. side = +1/-1
%   selects the assembly branch (which side of the chord P1-P2).
    if isvector(P1), P1 = repmat(P1(:), 1, size(P2,2)); end
    d  = P2 - P1;
    L  = sqrt(sum(d.^2, 1));
    u  = d ./ L;
    x  = (L.^2 + r1^2 - r2^2) ./ (2*L);
    y  = sqrt(max(r1^2 - x.^2, 0));
    nv = [-u(2,:); u(1,:)];             % unit normal to chord
    P  = P1 + u.*x + nv.*(side*y);
end