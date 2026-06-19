function results = runTracePlantFit(tracePath, options)
%RUNTRACEPLANTFIT Load a trace, fit yaw/pitch plants, and plot comparison.

arguments
    tracePath (1, 1) string
    options.DelaySec (1, 1) double = 0.05
    options.ShowPlots (1, 1) logical = true
end

trace = loadControlTrace(tracePath);
yaw = extractAxisTrace(trace, "yaw", DelaySec=options.DelaySec);
pitch = extractAxisTrace(trace, "pitch", DelaySec=options.DelaySec);

yawFit = estimateAxisPlant(yaw);
pitchFit = estimateAxisPlant(pitch);

results = struct();
results.trace = trace;
results.yaw = yaw;
results.pitch = pitch;
results.plant = struct("yaw", yawFit, "pitch", pitchFit);

fprintf("Trace counts: detection=%d control=%d camstate=%d\n", ...
    trace.counts.detection, trace.counts.control, trace.counts.camstate);
fprintf("Yaw fit:   a_u=% .6g  a_f=% .6g  omega RMSE=% .6g  theta RMSE=% .6g  n=%d\n", ...
    yawFit.a_u, yawFit.a_f, yawFit.rmseOmega, yawFit.rmseTheta, yawFit.numSamples);
fprintf("Pitch fit: a_u=% .6g  a_f=% .6g  omega RMSE=% .6g  theta RMSE=% .6g  n=%d\n", ...
    pitchFit.a_u, pitchFit.a_f, pitchFit.rmseOmega, pitchFit.rmseTheta, pitchFit.numSamples);

if options.ShowPlots
    plotFit(results);
end
end

function plotFit(results)
figure(Name="IDCS Plant Fit", Color="w");
tiledlayout(2, 2, TileSpacing="compact");

nexttile;
plotAxisTheta(results.yaw, results.plant.yaw, "Yaw theta");

nexttile;
plotAxisOmega(results.yaw, results.plant.yaw, "Yaw omega");

nexttile;
plotAxisTheta(results.pitch, results.plant.pitch, "Pitch theta");

nexttile;
plotAxisOmega(results.pitch, results.plant.pitch, "Pitch omega");
end

function plotAxisTheta(axisData, fit, titleText)
plot(axisData.sample.t, axisData.sample.theta, "k", DisplayName="recorded");
hold on;
plot(fit.sim.t, fit.sim.theta, "r--", DisplayName="fit");
grid on;
title(titleText);
xlabel("time (s)");
ylabel("rad");
legend(Location="best");
end

function plotAxisOmega(axisData, fit, titleText)
plot(axisData.sample.t, axisData.sample.omega, "k", DisplayName="recorded");
hold on;
plot(fit.sim.t, fit.sim.omega, "r--", DisplayName="fit");
plot(axisData.sample.t, axisData.sample.cmd, "b:", DisplayName="cmd");
grid on;
title(titleText);
xlabel("time (s)");
ylabel("rad/s");
legend(Location="best");
end
