%% Walking-linkage evolution: four-bar -> Klann six-bar -> Jansen eight-bar
%  Exports the one-revolution comparison to video (works in MATLAB Online).
%  Run: >> walking_evolution_sim
clear; clc; close all

%% 1. Target path
T  = readtable('target_foot_tip_path.csv');
xt = T.x_mm(:).';  yt = T.y_mm(:).';  yt = yt - min(yt);
TH = deg2rad(0:359);

%% 2. Stage 1: four-bar synthesis (random search + fminsearch polish)
fprintf('Stage 1/3: synthesising four-bar in MATLAB ...\n');
best = struct('e',inf,'p',[]);
rng(7)
for it = 1:2500
    p = [50+250*rand, 10+80*rand, 50+350*rand, 50+350*rand, 50+550*rand, 2*pi*rand];
    L = sort(p(1:4));
    if ~(L(1)+L(4) <= L(2)+L(3) && abs(L(1)-p(2))<1e-9), continue, end
    c = fourbarCurve(p, TH);
    if isempty(c), continue, end
    e = min(scoreRMSE(c.Px, c.Py, xt, yt), scoreRMSE(c.Px, -c.Py, xt, yt));
    if e < best.e, best.e = e; best.p = p;
        fprintf('  iter %5d  RMSE %6.2f mm\n', it, e); end
end
p4  = fminsearch(@(p) fourBarObj(p, TH, xt, yt), best.p, ...
                 optimset('MaxFunEvals',5000,'MaxIter',2500,'TolFun',1e-3,'Display','off'));
c4  = fourbarCurve(p4, TH);
if scoreRMSE(c4.Px,-c4.Py,xt,yt) < scoreRMSE(c4.Px,c4.Py,xt,yt)
    c4.Py = -c4.Py; c4.Cy = -c4.Cy;
end
rmse4 = scoreRMSE(c4.Px, c4.Py, xt, yt);

%% 3. Stage 2: Klann six-bar (US patent 6,260,862 proportions)
P15 = [17.607;11.807]; P9 = [17.818;16.076]; P11 = [12.101;10.186];
A29 = P15 + 2.976*[cos(TH); sin(TH)];
J27 = circInt(A29, 5.506, P11, 3.390, -1);
J35 = circInt(A29, 11.679, J27, 6.236, -1);
J37 = circInt(J35, 10.137, P9, 7.391, +1);
J33 = circInt(J35, 13.442, J37, 22.188, -1);          % foot
snap = norm(J27(:,181)-[9.125;11.807]) + norm(J35(:,181)-[3.024;13.099]) ...
     + norm(J37(:,181)-[11.119;19.200]) + norm(J33(:,181)-[0;0]);
assert(snap < 0.05, 'Klann assembly check failed (branches)');
rmse6 = scoreRMSE(J33(1,:), J33(2,:), xt, yt);

%% 4. Stage 3: Jansen eight-bar (holy numbers)
a=38; l=7.8; m=15; jb=41.5; jc=39.3; jd=40.1; je=55.8; jf=39.4;
jg=36.7; jh=65.7; ji=49; jj=50; jk=61.9;
Ja = m*[cos(TH); sin(TH)];  Jb = [-a; -l];
Jc = circInt(Jb, jb, Ja, jj, +1);
Jd = circInt(Jb, jc, Ja, jk, -1);
Je = circInt(Jb, jd, Jc, je, +1);
Jf = circInt(Jd, jg, Je, jf, +1);
Jg = circInt(Jd, ji, Jf, jh, +1);                     % toe
rmse8 = scoreRMSE(Jg(1,:), Jg(2,:), xt, yt);

%% 5. Common frame (stance on y=0, path centred on target)
[s4, X4, Y4, off4] = frameIt(c4.Px, c4.Py, xt);
[s6, X6, Y6, ~]    = frameIt(J33(1,:), J33(2,:), xt);
[s8, X8, Y8, ~]    = frameIt(Jg(1,:),  Jg(2,:), xt);

%% 6. Summary
st = @(x) (max(x)-min(x));
fprintf('\n================ EVOLUTION SUMMARY ================\n');
fprintf('Stage 1  four-bar (synthesised)  RMSE %7.2f mm   stride %6.1f mm\n', rmse4, st(X4));
fprintf('Stage 2  Klann six-bar (patent)  RMSE %7.2f mm   stride %6.1f mm\n', rmse6, st(X6));
fprintf('Stage 3  Jansen eight-bar        RMSE %7.2f mm   stride %6.1f mm\n', rmse8, st(X8));
fprintf('Target                                                    %6.1f mm\n', st(xt));
fprintf('DOF  4-bar: 3(4-1)-2*4=1 | 6-bar: 3(6-1)-2*7=1 | 8-bar: 3(8-1)-2*10=1\n');

%% 7. Animation -> video file
fig = figure('Color','w','Position',[60 60 1250 850]);
tiledlayout(3,1,'Padding','compact');
ax1 = nexttile; hold(ax1,'on'); grid(ax1,'on'); axis(ax1,'equal')
title(ax1, sprintf('Stage 1 - FOUR-BAR (synthesised)  RMSE %.1f mm', rmse4));
ax2 = nexttile; hold(ax2,'on'); grid(ax2,'on'); axis(ax2,'equal')
title(ax2, sprintf('Stage 2 - KLANN SIX-BAR (patent)  RMSE %.1f mm', rmse6));
ax3 = nexttile; hold(ax3,'on'); grid(ax3,'on'); axis(ax3,'equal')
title(ax3, sprintf('Stage 3 - JANSEN EIGHT-BAR  RMSE %.1f mm', rmse8));
for ax = [ax1 ax2 ax3]
    plot(ax, xt, yt, 'k--', 'LineWidth', 1.6);
    plot(ax, [min(xt)-50 max(xt)+50], [0 0], 'k-', 'LineWidth', 1.5);
    xlim(ax, [min(xt)-80 max(xt)+80]); ylim(ax, [-120 1350]);
    xlabel(ax,'x (mm)'); ylabel(ax,'y (mm)');
end
h1 = mkLinks(ax1, [1 4; 1 2; 2 3; 3 4; 2 5]);
h2 = mkLinks(ax2, [1 2; 2 3; 3 4; 2 4; 8 3; 7 5; 4 5; 5 6; 4 6]);
h3 = mkLinks(ax3, [1 3; 1 2; 3 4; 2 4; 3 5; 2 5; 3 6; 4 6; 6 7; 5 7; 5 8; 7 8]);
tr1 = plot(ax1, nan, nan, 'r-', 'LineWidth', 1.4);
tr2 = plot(ax2, nan, nan, 'r-', 'LineWidth', 1.4);
tr3 = plot(ax3, nan, nan, 'r-', 'LineWidth', 1.4);

try
    vw = VideoWriter('walking_evolution', 'MPEG-4');
catch
    vw = VideoWriter('walking_evolution', 'Motion JPEG AVI');
end
vw.FrameRate = 25; vw.Quality = 90; open(vw);
fprintf('\nRendering one revolution to video ...\n');
for k = 1:180
    idx = mod(2*(k-1), 360) + 1;
    g = p4(1);
    P1 = [[0;0] [c4.Ax(idx);c4.Ay(idx)] [c4.Cx(idx);c4.Cy(idx)] [g;0] [c4.Px(idx);c4.Py(idx)]];
    setLinks(h1, s4*P1 + off4);
    set(tr1, 'XData', X4(1:idx), 'YData', Y4(1:idx));
    P2 = [P15 A29(:,idx) J27(:,idx) J35(:,idx) J37(:,idx) J33(:,idx) P9 P11];
    setLinks(h2, s6scaled(P2, xt, J33));
    set(tr2, 'XData', X6(1:idx), 'YData', Y6(1:idx));
    P3 = [[0;0] Ja(:,idx) Jb Jc(:,idx) Jd(:,idx) Je(:,idx) Jf(:,idx) Jg(:,idx)];
    setLinks(h3, s8scaled(P3, xt, Jg));
    set(tr3, 'XData', X8(1:idx), 'YData', Y8(1:idx));
    drawnow
    writeVideo(vw, getframe(fig));
end
close(vw);
fprintf('Saved %s to MATLAB Drive - download it into your repo results/ folder.\n', vw.Filename);

%% local functions (MUST stay at the end of this file) ===================
function h = mkLinks(ax, segs)
    h = gobjects(size(segs,1),1);
    for t = 1:size(segs,1)
        h(t) = plot(ax, [0 0], [0 0], '-o', 'LineWidth', 2, ...
                    'MarkerSize', 4, 'Color', [0.1 0.35 0.8]);
        h(t).UserData = segs(t,:);
    end
end

function setLinks(h, P)
    for t = 1:numel(h)
        set(h(t), 'XData', P(1, h(t).UserData), 'YData', P(2, h(t).UserData));
    end
end

function c = fourbarCurve(p, TH)
    g=p(1); r1=p(2); r2=p(3); r3=p(4); bp=p(5); beta=p(6);
    Bx=r1*cos(TH); By=r1*sin(TH);
    d = hypot(g-Bx, -By);
    if any(d > r2+r3) || any(d < abs(r2-r3)) || any(d < 1e-9), c = []; return, end
    aa = (r2^2 - r3^2 + d.^2)./(2*d);
    h2 = r2^2 - aa.^2;
    if any(h2 < 0), c = []; return, end
    h = sqrt(h2);
    mx = Bx + aa.*(g-Bx)./d;  my = By + aa.*(-By)./d;
    Cx = mx + h.*By./d;       Cy = my + h.*(g-Bx)./d;
    psi = atan2(Cy-By, Cx-Bx) + beta;
    c.Px = Bx + bp*cos(psi);  c.Py = By + bp*sin(psi);
    c.Ax = Bx; c.Ay = By; c.Cx = Cx; c.Cy = Cy;
end

function e = fourBarObj(p, TH, xt, yt)
    L = sort(p(1:4));
    pen = 0; if ~(L(1)+L(4) <= L(2)+L(3) && abs(L(1)-p(2))<1e-9), pen = 500; end
    c = fourbarCurve(p, TH);
    if isempty(c), e = 1e4; return, end
    e = min(scoreRMSE(c.Px, c.Py, xt, yt), scoreRMSE(c.Px, -c.Py, xt, yt)) + pen;
end

function e = scoreRMSE(px, py, xt, yt)
    py = py - min(py);  sc = 200/(max(py)-min(py));
    px = sc*px;  py = sc*py;  px = px - mean(px);  tc = xt - mean(xt);
    e = inf;
    for sh = 0:6:354
        r = mod((0:359)+sh, 360) + 1;
        e = min(e, sqrt(mean((px(r)-tc).^2 + (py-yt).^2)));
    end
end

function [sc, X, Y, off] = frameIt(px, py, xt)
    sc = 200/(max(py)-min(py));
    X = sc*px;  Y = sc*py - min(sc*py);
    off = [ (min(xt)+max(xt))/2 - (min(X)+max(X))/2 ; -min(sc*py) ];
    X = X + off(1);
end

function P = s6scaled(P, xt, J33)
    sc = 200/(max(J33(2,:))-min(J33(2,:)));
    P = sc*P;
    P(2,:) = P(2,:) - min(sc*J33(2,:));
    P(1,:) = P(1,:) + (min(xt)+max(xt))/2 - sc*(min(J33(1,:))+max(J33(1,:)))/2;
end

function P = s8scaled(P, xt, Jg)
    sc = 200/(max(Jg(2,:))-min(Jg(2,:)));
    P = sc*P;
    P(2,:) = P(2,:) - min(sc*Jg(2,:));
    P(1,:) = P(1,:) + (min(xt)+max(xt))/2 - sc*(min(Jg(1,:))+max(Jg(1,:)))/2;
end

function P = circInt(P1, r1, P2, r2, side)
    if isvector(P1), P1 = repmat(P1(:), 1, size(P2,2)); end
    d  = P2 - P1;
    L  = sqrt(sum(d.^2, 1));
    u  = d ./ L;
    x  = (L.^2 + r1^2 - r2^2) ./ (2*L);
    y  = sqrt(max(r1^2 - x.^2, 0));
    nv = [-u(2,:); u(1,:)];
    P  = P1 + u.*x + nv.*(side*y);
end