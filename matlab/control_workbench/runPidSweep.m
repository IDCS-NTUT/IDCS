function sweep = runPidSweep(plant, options)
%RUNPIDSWEEP Brute-force a small PID grid against the simple axis plant.

arguments
    plant (1, 1) struct
    options.Axis (1, 1) string = "yaw"
    options.Kp (:, 1) double = linspace(0.2, 2.0, 10).'
    options.Ki (:, 1) double = [0; 0.05; 0.1; 0.2]
    options.Kd (:, 1) double = [0; 0.05; 0.1; 0.2]
    options.Reference (1, 1) double = 0.1
    options.StopTime (1, 1) double = 5.0
end

rows = [];
bestScore = inf;
best = struct();

for kp = options.Kp.'
    for ki = options.Ki.'
        for kd = options.Kd.'
            pid = struct("kp", kp, "ki", ki, "kd", kd);
            sim = simulatePidAxis(plant, pid, ...
                Reference=options.Reference, StopTime=options.StopTime);
            m = sim.metrics;
            settlePenalty = m.settlingTime;
            if isnan(settlePenalty)
                settlePenalty = options.StopTime * 2.0;
            end
            score = m.rmsError + 0.25 * m.overshoot + 0.02 * settlePenalty;
            rows = [rows; kp, ki, kd, score, m.rmsError, m.overshoot, m.settlingTime]; %#ok<AGROW>
            if score < bestScore
                bestScore = score;
                best.pid = pid;
                best.sim = sim;
                best.score = score;
            end
        end
    end
end

sweep = struct();
sweep.axis = options.Axis;
sweep.table = array2table(rows, VariableNames=[ ...
    "kp", "ki", "kd", "score", "rmsError", "overshoot", "settlingTime"]);
sweep.best = best;

fprintf("Best %s PID: kp=%g ki=%g kd=%g score=%g\n", ...
    options.Axis, best.pid.kp, best.pid.ki, best.pid.kd, best.score);

figure(Name="IDCS PID Sweep", Color="w");
plot(best.sim.t, best.sim.reference, "k:", DisplayName="reference");
hold on;
plot(best.sim.t, best.sim.theta, "r", DisplayName="theta");
plot(best.sim.t, best.sim.cmd, "b--", DisplayName="cmd");
grid on;
title("Best " + options.Axis + " PID step response");
xlabel("time (s)");
legend(Location="best");
end
