function model = fitGimbalResponseModel(csvPath, options)
%FITGIMBALRESPONSEMODEL Fit a simple virtual-gimbal model from sweep data.
%
% The fitted model is:
%
%   omega_dot = a_u * u_delay - a_f * omega
%   theta_dot = omega
%
% This matches the repo MPC plant shape and is suitable for a first SimCamera
% replacement. Delay is selected by sweeping candidate command delays and
% minimizing omega RMSE.

arguments
    csvPath (1, 1) string
    options.ManifestPath (1, 1) string = ""
    options.DelaySec (:, 1) double = (0:0.005:0.15).'
    options.ShowPlots (1, 1) logical = true
end

sweep = loadGimbalResponseSweep(csvPath, ManifestPath=options.ManifestPath);
T = sweep.table;
axes = unique(T.axis);

model = struct();
model.source_csv = csvPath;
model.created = string(datetime("now", TimeZone="local", Format="yyyy-MM-dd HH:mm:ss Z"));
model.axes = struct();

for i = 1:numel(axes)
    axisName = axes(i);
    axisFit = fitAxis(T(T.axis == axisName, :), axisName, options.DelaySec);
    model.axes.(char(axisName)) = axisFit;
    fprintf("%s: delay=%g s  a_u=%g  a_f=%g  tau=%g s  gain=%g  omega RMSE=%g  n=%d\n", ...
        upper(axisName), axisFit.delay_s, axisFit.a_u, axisFit.a_f, ...
        axisFit.tau_s, axisFit.dc_gain, axisFit.rmse_omega, axisFit.num_samples);
end

if options.ShowPlots
    plotFits(T, model);
end
end

function fit = fitAxis(T, axisName, delays)
best = struct("score", inf);

for i = 1:numel(delays)
    delay = delays(i);
    rows = buildFitRows(T, delay);
    if height(rows) < 10
        continue
    end

    y = rows.omega_dot;
    X = [rows.u_delay, -rows.omega];
    coeff = X \ y;
    pred = X * coeff;
    rmse = sqrt(mean((pred - y).^2, "omitnan"));

    if rmse < best.score
        best.score = rmse;
        best.delay = delay;
        best.coeff = coeff;
        best.rows = rows;
        best.pred = pred;
    end
end

if ~isfinite(best.score)
    error("fitGimbalResponseModel:NoFit", "No usable samples for axis %s.", axisName);
end

a_u = best.coeff(1);
a_f = max(1e-9, best.coeff(2));
fit = struct();
fit.axis = axisName;
fit.delay_s = best.delay;
fit.a_u = a_u;
fit.a_f = a_f;
fit.tau_s = 1.0 / a_f;
fit.dc_gain = a_u / a_f;
fit.rmse_omega = best.score;
fit.num_samples = height(best.rows);
fit.command_deadband_rad_s = estimateDeadband(T);
fit.max_observed_rate_rad_s = max(abs(T.omega_rad_s), [], "omitnan");
fit.max_command_rad_s = max(abs(T.cmd_rate_applied_rad_s), [], "omitnan");
fit._rows = best.rows;
fit._omega_dot_pred = best.pred;
end

function rows = buildFitRows(T, delay)
T = sortrows(T, ["setting_id", "trial", "direction", "t"]);
rows = table();
groups = findgroups(T.setting_id, T.trial, T.direction);

for g = 1:max(groups)
    G = T(groups == g, :);
    if height(G) < 3
        continue
    end
    t = G.t;
    u = G.cmd_rate_applied_rad_s;
    omega = G.omega_rad_s;
    uDelay = previousSample(t, u, t - delay, 0.0);
    dt = diff(t);
    domega = diff(omega);
    valid = dt > 0 & isfinite(dt) & isfinite(domega);
    if ~any(valid)
        continue
    end
    out = table();
    out.t = t(2:end);
    out.u_delay = uDelay(1:end-1);
    out.omega = omega(1:end-1);
    out.omega_dot = domega ./ dt;
    out = out(valid, :);
    rows = [rows; out]; %#ok<AGROW>
end
end

function deadband = estimateDeadband(T)
stepRows = T(T.phase == "step", :);
if isempty(stepRows)
    deadband = 0.0;
    return
end
cmds = unique(abs(stepRows.cmd_rate_applied_rad_s));
cmds = sort(cmds(isfinite(cmds) & cmds > 0));
deadband = 0.0;
for i = 1:numel(cmds)
    rows = stepRows(abs(abs(stepRows.cmd_rate_applied_rad_s) - cmds(i)) < 1e-9, :);
    if mean(abs(rows.omega_rad_s), "omitnan") > 0.02
        deadband = max(0.0, cmds(i) * 0.5);
        return
    end
end
end

function yq = previousSample(t, y, tq, defaultValue)
yq = defaultValue * ones(size(tq));
for i = 1:numel(tq)
    idx = find(t <= tq(i), 1, "last");
    if ~isempty(idx)
        yq(i) = y(idx);
    end
end
end

function plotFits(T, model)
axisNames = string(fieldnames(model.axes));
figure(Name="IDCS Gimbal Response Fit", Color="w");
tiledlayout(numel(axisNames), 2, TileSpacing="compact");
for i = 1:numel(axisNames)
    axisName = axisNames(i);
    fit = model.axes.(char(axisName));
    rows = fit._rows;

    nexttile;
    plot(rows.t, rows.omega_dot, "k.", DisplayName="measured");
    hold on;
    plot(rows.t, fit._omega_dot_pred, "r.", DisplayName="model");
    grid on;
    title(upper(axisName) + " omega dot");
    xlabel("time (s)");
    ylabel("rad/s^2");
    legend(Location="best");

    nexttile;
    A = T(T.axis == axisName & T.phase == "step", :);
    scatter(A.cmd_rate_applied_rad_s, A.omega_rad_s, 8, double(A.accel_byte), "filled");
    grid on;
    title(upper(axisName) + " command/rate samples");
    xlabel("command rad/s");
    ylabel("omega rad/s");
    colorbar;
end
end
