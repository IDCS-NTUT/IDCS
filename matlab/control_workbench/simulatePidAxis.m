function sim = simulatePidAxis(plant, pid, options)
%SIMULATEPIDAXIS Simulate one closed-loop axis with PID rate command output.
%
% The controller output u is the commanded axis rate. The plant converts that
% command into measured omega and theta using the first-order rate model.

arguments
    plant (1, 1) struct
    pid (1, 1) struct
    options.Ts (1, 1) double = 0.01
    options.StopTime (1, 1) double = 5.0
    options.Reference (1, 1) double = 0.1
    options.RateLimit (1, 1) double = 0.5
    options.AccelLimit (1, 1) double = 3.5
    options.InitialTheta (1, 1) double = 0.0
    options.InitialOmega (1, 1) double = 0.0
    options.SettlingThreshold (1, 1) double = 0.02
end

t = (0:options.Ts:options.StopTime).';
n = numel(t);
theta = zeros(n, 1);
omega = zeros(n, 1);
u = zeros(n, 1);
err = zeros(n, 1);
theta(1) = options.InitialTheta;
omega(1) = options.InitialOmega;

integ = 0.0;
prevErr = options.Reference - theta(1);
prevU = 0.0;

for k = 2:n
    err(k - 1) = options.Reference - theta(k - 1);
    integ = integ + err(k - 1) * options.Ts;
    deriv = (err(k - 1) - prevErr) / options.Ts;

    desiredU = pid.kp * err(k - 1) + pid.ki * integ + pid.kd * deriv;
    desiredU = clamp(desiredU, -options.RateLimit, options.RateLimit);
    maxDelta = options.AccelLimit * options.Ts;
    u(k - 1) = clamp(desiredU, prevU - maxDelta, prevU + maxDelta);

    theta(k) = theta(k - 1) + omega(k - 1) * options.Ts;
    omegaDot = plant.a_u * u(k - 1) - plant.a_f * omega(k - 1);
    omega(k) = omega(k - 1) + omegaDot * options.Ts;

    prevErr = err(k - 1);
    prevU = u(k - 1);
end

err(end) = options.Reference - theta(end);
u(end) = u(end - 1);

metrics = stepMetrics(t, theta, options.Reference, options.SettlingThreshold);
sim = struct();
sim.t = t;
sim.reference = options.Reference * ones(n, 1);
sim.theta = theta;
sim.omega = omega;
sim.cmd = u;
sim.err = err;
sim.metrics = metrics;
end

function metrics = stepMetrics(t, y, ref, threshold)
finalErr = abs(ref - y(end));
overshoot = max(0.0, max(y - ref));
absErr = abs(ref - y);
settlingTime = NaN;
for i = 1:numel(t)
    if all(absErr(i:end) <= threshold)
        settlingTime = t(i);
        break
    end
end
metrics = struct( ...
    "finalError", finalErr, ...
    "overshoot", overshoot, ...
    "settlingTime", settlingTime, ...
    "rmsError", sqrt(mean(absErr.^2)));
end

function y = clamp(x, lo, hi)
y = min(hi, max(lo, x));
end
