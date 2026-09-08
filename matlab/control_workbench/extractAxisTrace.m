function axisData = extractAxisTrace(trace, axisName, options)
%EXTRACTAXISTRACE Align command and CamState samples for one axis.
%
% axisData = extractAxisTrace(trace, "yaw", DelaySec=0.05) uses recorder time
% and zero-order hold command alignment. DelaySec shifts commands forward so
% the sample at time t uses the latest command from t - DelaySec.

arguments
    trace (1, 1) struct
    axisName (1, 1) string {mustBeMember(axisName, ["yaw", "pitch"])}
    options.DelaySec (1, 1) double = 0.0
end

isYaw = axisName == "yaw";

[tCmd, frameCmd, cmd, errRad, targetOk] = extractControl(trace.control, isYaw);
[tCam, frameCam, theta, omega] = extractCamState(trace.camstate, isYaw);

t0 = minFinite([tCmd; tCam]);
if isnan(t0)
    error("extractAxisTrace:NoTime", "Trace does not contain usable %s timing.", axisName);
end

tCmd = tCmd - t0;
tCam = tCam - t0;

cmdAtCam = previousSample(tCmd, cmd, tCam - options.DelaySec, 0.0);

axisData = struct();
axisData.axis = axisName;
axisData.delaySec = options.DelaySec;
axisData.command = table(tCmd, frameCmd, cmd, errRad, targetOk, ...
    VariableNames=["t", "frame_id", "cmd", "err_rad", "target_ok"]);
axisData.camstate = table(tCam, frameCam, theta, omega, ...
    VariableNames=["t", "frame_id", "theta", "omega"]);
axisData.sample = table(tCam, cmdAtCam, theta, omega, ...
    VariableNames=["t", "cmd", "theta", "omega"]);
end

function [t, frame, cmd, errRad, targetOk] = extractControl(records, isYaw)
n = numel(records);
t = nan(n, 1);
frame = nan(n, 1);
cmd = nan(n, 1);
errRad = nan(n, 1);
targetOk = false(n, 1);

for i = 1:n
    record = records{i};
    payload = record.payload;
    t(i) = double(record.rx_s);
    frame(i) = getNumber(payload, "frame_id", NaN);
    if isYaw
        cmd(i) = getNumber(payload, "pan_rate_cmd", NaN);
    else
        cmd(i) = getNumber(payload, "tilt_rate_cmd", NaN);
    end
    if isfield(payload, "err_rad")
        err = double(payload.err_rad);
        if numel(err) >= 2
            errRad(i) = err(1 + ~isYaw);
        end
    end
    if isfield(payload, "target_ok")
        targetOk(i) = logical(payload.target_ok);
    end
end

valid = isfinite(t) & isfinite(cmd);
t = t(valid);
frame = frame(valid);
cmd = cmd(valid);
errRad = errRad(valid);
targetOk = targetOk(valid);
end

function [t, frame, theta, omega] = extractCamState(records, isYaw)
n = numel(records);
t = nan(n, 1);
frame = nan(n, 1);
theta = nan(n, 1);
omega = nan(n, 1);

for i = 1:n
    record = records{i};
    payload = record.payload;
    t(i) = double(record.rx_s);
    frame(i) = getNumber(payload, "frame_id", NaN);
    if isYaw
        theta(i) = getNumber(payload, "pan", NaN);
        omega(i) = getNumber(payload, "pan_rate", NaN);
    else
        theta(i) = getNumber(payload, "tilt", NaN);
        omega(i) = getNumber(payload, "tilt_rate", NaN);
    end
end

valid = isfinite(t) & isfinite(theta);
t = t(valid);
frame = frame(valid);
theta = theta(valid);
omega = omega(valid);
end

function value = getNumber(s, fieldName, defaultValue)
if isfield(s, fieldName)
    value = double(s.(fieldName));
else
    value = defaultValue;
end
end

function value = minFinite(values)
values = values(isfinite(values));
if isempty(values)
    value = NaN;
else
    value = min(values);
end
end

function yq = previousSample(t, y, tq, defaultValue)
yq = defaultValue * ones(size(tq));
if isempty(t) || isempty(y)
    return
end
[t, order] = sort(t(:));
y = y(order);
for i = 1:numel(tq)
    idx = find(t <= tq(i), 1, "last");
    if ~isempty(idx)
        yq(i) = y(idx);
    end
end
end
