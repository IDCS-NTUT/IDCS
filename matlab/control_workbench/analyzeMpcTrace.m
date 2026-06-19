function report = analyzeMpcTrace(tracePath, options)
%ANALYZE MPCTRACE Summarize MPC diagnostics recorded in ControlCmd messages.
%
% report = analyzeMpcTrace(tracePath) loads a control_trace_*.jsonl file and
% extracts per-axis MPC status, reference, prediction, cost-term, and command
% saturation diagnostics from the ControlCmd payloads.

arguments
    tracePath (1, 1) string
    options.ShowPlots (1, 1) logical = true
    options.RateLimit (1, 1) double = 0.5
end

trace = loadControlTrace(tracePath);
yaw = extractAxisMpc(trace, "yaw", options.RateLimit);
pitch = extractAxisMpc(trace, "pitch", options.RateLimit);

report = struct();
report.trace = trace;
report.yaw = yaw;
report.pitch = pitch;

fprintf("Trace counts: detection=%d control=%d camstate=%d\n", ...
    trace.counts.detection, trace.counts.control, trace.counts.camstate);
fprintf("MPC samples: yaw=%d pitch=%d\n", height(yaw), height(pitch));
printAxisSummary("yaw", yaw);
printAxisSummary("pitch", pitch);

if options.ShowPlots
    plotMpcReport(report);
end
end

function axisTable = extractAxisMpc(trace, axisName, rateLimit)
isYaw = axisName == "yaw";
axisField = char(axisName);
records = trace.control;
n = numel(records);

rows = [];
statuses = strings(0, 1);

for i = 1:n
    record = records{i};
    payload = record.payload;
    if string(getOptional(payload, "controller_mode", "")) ~= "mpc"
        continue
    end
    if ~isfield(payload, "mpc") || ~isfield(payload.mpc, axisField)
        continue
    end

    diag = payload.mpc.(axisField);
    row = nan(1, 23);
    row(1) = double(record.rx_s);
    row(2) = getNumber(payload, "frame_id", NaN);
    row(3) = double(getLogical(payload, "target_ok", false));
    row(4) = getErrRad(payload, isYaw);
    if isYaw
        row(5) = getNumber(payload, "pan_rate_cmd", NaN);
    else
        row(5) = getNumber(payload, "tilt_rate_cmd", NaN);
    end
    row(6) = getNumber(diag, "u0", NaN);
    row(7) = abs(row(5)) >= (rateLimit - 1e-9);
    row(8) = getNestedNumber(diag, ["solver", "iter"], NaN);
    row(9) = getNestedNumber(diag, ["solver", "status_val"], NaN);
    row(10) = getNumber(diag, "cost", NaN);
    row(11) = getNestedNumber(diag, ["refs", "theta_ref0"], NaN);
    row(12) = getNestedNumber(diag, ["refs", "omega_ref0"], NaN);
    row(13) = getNestedNumber(diag, ["refs", "effect_delay_s"], NaN);
    row(14) = getNestedNumber(diag, ["pred", "theta_pred0"], NaN);
    row(15) = getNestedNumber(diag, ["pred", "omega_pred0"], NaN);
    row(16) = getNestedNumber(diag, ["terms", "theta"], 0.0);
    row(17) = getNestedNumber(diag, ["terms", "omega"], 0.0);
    row(18) = getNestedNumber(diag, ["terms", "dtheta"], 0.0);
    row(19) = getNestedNumber(diag, ["terms", "effort"], 0.0);
    row(20) = getNestedNumber(diag, ["terms", "slew"], 0.0);
    row(21) = getNestedNumber(diag, ["terms", "slack"], 0.0);
    row(22) = abs(row(14) - row(11));
    row(23) = abs(row(15) - row(12));

    rows = [rows; row]; %#ok<AGROW>
    statuses(end + 1, 1) = string(getOptional(diag, "status", "missing")); %#ok<AGROW>
end

if isempty(rows)
    axisTable = table();
    return
end

t0 = min(rows(:, 1), [], "omitnan");
rows(:, 1) = rows(:, 1) - t0;

axisTable = array2table(rows, VariableNames=[
    "t", "frame_id", "target_ok", "err_rad", "cmd", "u0", "saturated", ...
    "solver_iter", "solver_status_val", "cost", ...
    "theta_ref0", "omega_ref0", "effect_delay_s", ...
    "theta_pred0", "omega_pred0", ...
    "term_theta", "term_omega", "term_dtheta", ...
    "term_effort", "term_slew", "term_slack", ...
    "abs_theta_ref_pred_gap", "abs_omega_ref_pred_gap"]);
axisTable.status = statuses;
axisTable.saturated = logical(axisTable.saturated);
axisTable.target_ok = logical(axisTable.target_ok);
end

function printAxisSummary(axisName, data)
if isempty(data) || height(data) == 0
    fprintf("%s: no MPC diagnostics found\n", axisName);
    return
end

fprintf("\n%s MPC\n", upper(axisName));
printStatusCounts(data.status);
fprintf("  command abs: mean=%g p95=%g max=%g saturated=%d/%d\n", ...
    mean(abs(data.cmd), "omitnan"), pctile(abs(data.cmd), 95), ...
    max(abs(data.cmd), [], "omitnan"), nnz(data.saturated), height(data));
fprintf("  error abs:   mean=%g p95=%g max=%g\n", ...
    mean(abs(data.err_rad), "omitnan"), pctile(abs(data.err_rad), 95), ...
    max(abs(data.err_rad), [], "omitnan"));
fprintf("  solver iter: median=%g p95=%g max=%g\n", ...
    median(data.solver_iter, "omitnan"), pctile(data.solver_iter, 95), ...
    max(data.solver_iter, [], "omitnan"));
fprintf("  ref/pred theta gap abs: mean=%g p95=%g max=%g\n", ...
    mean(data.abs_theta_ref_pred_gap, "omitnan"), ...
    pctile(data.abs_theta_ref_pred_gap, 95), ...
    max(data.abs_theta_ref_pred_gap, [], "omitnan"));
fprintf("  ref/pred omega gap abs: mean=%g p95=%g max=%g\n", ...
    mean(data.abs_omega_ref_pred_gap, "omitnan"), ...
    pctile(data.abs_omega_ref_pred_gap, 95), ...
    max(data.abs_omega_ref_pred_gap, [], "omitnan"));
end

function printStatusCounts(statuses)
u = unique(statuses);
for i = 1:numel(u)
    fprintf("  status %-28s %d\n", char(u(i)), nnz(statuses == u(i)));
end
end

function plotMpcReport(report)
figure(Name="IDCS MPC Trace Analysis", Color="w");
tiledlayout(3, 2, TileSpacing="compact");
plotAxis(report.yaw, "Yaw");
plotAxis(report.pitch, "Pitch");
end

function plotAxis(data, label)
if isempty(data) || height(data) == 0
    nexttile; title(label + " no MPC data");
    nexttile; title(label + " no MPC data");
    nexttile; title(label + " no MPC data");
    return
end

nexttile;
plot(data.t, data.err_rad, "k", DisplayName="err rad");
hold on;
plot(data.t, data.cmd, "b", DisplayName="cmd");
plot(data.t, data.u0, "c:", DisplayName="raw u0");
grid on;
title(label + " command/error");
xlabel("time (s)");
legend(Location="best");

nexttile;
plot(data.t, data.term_theta, DisplayName="theta");
hold on;
plot(data.t, data.term_omega, DisplayName="omega");
plot(data.t, data.term_dtheta, DisplayName="dtheta");
plot(data.t, data.term_effort, DisplayName="effort");
plot(data.t, data.term_slew, DisplayName="slew");
plot(data.t, data.term_slack, DisplayName="slack");
grid on;
title(label + " cost terms");
xlabel("time (s)");
legend(Location="best");

nexttile;
plot(data.t, data.abs_theta_ref_pred_gap, "r", DisplayName="theta gap");
hold on;
plot(data.t, data.abs_omega_ref_pred_gap, "b", DisplayName="omega gap");
plot(data.t, data.effect_delay_s, "k:", DisplayName="effect delay");
grid on;
title(label + " ref/pred gap");
xlabel("time (s)");
legend(Location="best");
end

function value = getOptional(s, fieldName, defaultValue)
if isstruct(s) && isfield(s, fieldName)
    value = s.(fieldName);
else
    value = defaultValue;
end
end

function value = getNumber(s, fieldName, defaultValue)
if isstruct(s) && isfield(s, fieldName)
    value = double(s.(fieldName));
else
    value = defaultValue;
end
end

function value = getLogical(s, fieldName, defaultValue)
if isstruct(s) && isfield(s, fieldName)
    value = logical(s.(fieldName));
else
    value = defaultValue;
end
end

function value = getNestedNumber(s, path, defaultValue)
value = defaultValue;
current = s;
for i = 1:numel(path)
    key = char(path(i));
    if ~isstruct(current) || ~isfield(current, key)
        return
    end
    current = current.(key);
end
value = double(current);
end

function value = getErrRad(payload, isYaw)
value = NaN;
if ~isfield(payload, "err_rad")
    return
end
err = double(payload.err_rad);
if numel(err) < 2
    return
end
if isYaw
    value = err(1);
else
    value = err(2);
end
end

function value = pctile(x, p)
x = x(isfinite(x));
if isempty(x)
    value = NaN;
    return
end
x = sort(x(:));
idx = 1 + (numel(x) - 1) * (p / 100);
lo = floor(idx);
hi = ceil(idx);
if lo == hi
    value = x(lo);
else
    frac = idx - lo;
    value = (1 - frac) * x(lo) + frac * x(hi);
end
end
