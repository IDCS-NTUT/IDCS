function sim = simulateAxisPlant(t, u, plant, x0)
%SIMULATEAXISPLANT Simulate theta_dot=omega, omega_dot=a_u*u-a_f*omega.

arguments
    t (:, 1) double
    u (:, 1) double
    plant (1, 1) struct
    x0 (2, 1) double = [0; 0]
end

if numel(t) ~= numel(u)
    error("simulateAxisPlant:SizeMismatch", "t and u must have the same length.");
end

theta = nan(size(t));
omega = nan(size(t));
theta(1) = x0(1);
omega(1) = x0(2);

for k = 2:numel(t)
    dt = t(k) - t(k - 1);
    if ~isfinite(dt) || dt <= 0
        theta(k) = theta(k - 1);
        omega(k) = omega(k - 1);
        continue
    end
    theta(k) = theta(k - 1) + omega(k - 1) * dt;
    omegaDot = plant.a_u * u(k - 1) - plant.a_f * omega(k - 1);
    omega(k) = omega(k - 1) + omegaDot * dt;
end

sim = table(t, u, theta, omega, VariableNames=["t", "u", "theta", "omega"]);
end
