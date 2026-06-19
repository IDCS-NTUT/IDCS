function fit = estimateAxisPlant(axisData)
%ESTIMATEAXISPLANT Fit omega_dot = a_u*u - a_f*omega from trace samples.

sample = axisData.sample;
t = sample.t;
u = sample.cmd;
theta = sample.theta;
omega = sample.omega;

valid = isfinite(t) & isfinite(u) & isfinite(theta) & isfinite(omega);
t = t(valid);
u = u(valid);
theta = theta(valid);
omega = omega(valid);

dt = diff(t);
omegaDot = diff(omega) ./ dt;
u0 = u(1:end - 1);
omega0 = omega(1:end - 1);

fitMask = isfinite(dt) & dt > 0 & isfinite(omegaDot) & isfinite(u0) & isfinite(omega0);
X = [u0(fitMask), -omega0(fitMask)];
y = omegaDot(fitMask);

if size(X, 1) < 10
    error("estimateAxisPlant:NotEnoughSamples", ...
        "Need at least 10 usable samples for %s, got %d.", axisData.axis, size(X, 1));
end

beta = X \ y;
a_u = beta(1);
a_f = beta(2);

sim = simulateAxisPlant(t, u, struct("a_u", a_u, "a_f", a_f), [theta(1); omega(1)]);
omegaErr = omega - sim.omega;
thetaErr = theta - sim.theta;

fit = struct();
fit.axis = axisData.axis;
fit.a_u = a_u;
fit.a_f = a_f;
fit.numSamples = size(X, 1);
fit.rmseOmega = sqrt(mean(omegaErr.^2, "omitnan"));
fit.rmseTheta = sqrt(mean(thetaErr.^2, "omitnan"));
fit.sample = sample;
fit.sim = sim;
end
