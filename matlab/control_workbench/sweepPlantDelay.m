function sweep = sweepPlantDelay(tracePath, options)
%SWEEPPLANTDELAY Estimate command/state delay by sweeping fit quality.
%
% The trace alignment uses recorder receive time. This helper tries several
% command delays and refits the yaw/pitch rate plants at each delay. Pick the
% delay with the lowest omega RMSE first; theta RMSE is secondary because angle
% integration accumulates bias over long traces.

arguments
    tracePath (1, 1) string
    options.DelaySec (:, 1) double = (0:0.01:0.20).'
    options.ShowPlot (1, 1) logical = true
end

trace = loadControlTrace(tracePath);
rows = nan(numel(options.DelaySec), 7);

for i = 1:numel(options.DelaySec)
    delaySec = options.DelaySec(i);
    yaw = extractAxisTrace(trace, "yaw", DelaySec=delaySec);
    pitch = extractAxisTrace(trace, "pitch", DelaySec=delaySec);
    yawFit = estimateAxisPlant(yaw);
    pitchFit = estimateAxisPlant(pitch);

    meanOmegaRmse = 0.5 * (yawFit.rmseOmega + pitchFit.rmseOmega);
    meanThetaRmse = 0.5 * (yawFit.rmseTheta + pitchFit.rmseTheta);
    rows(i, :) = [
        delaySec, ...
        yawFit.rmseOmega, pitchFit.rmseOmega, meanOmegaRmse, ...
        yawFit.rmseTheta, pitchFit.rmseTheta, meanThetaRmse];
end

tableOut = array2table(rows, VariableNames=[
    "delaySec", ...
    "yawOmegaRmse", "pitchOmegaRmse", "meanOmegaRmse", ...
    "yawThetaRmse", "pitchThetaRmse", "meanThetaRmse"]);

[~, bestIdx] = min(tableOut.meanOmegaRmse);
sweep = struct();
sweep.table = tableOut;
sweep.best = tableOut(bestIdx, :);

fprintf("Best delay by mean omega RMSE: %.3f s\n", sweep.best.delaySec);
fprintf("  yaw omega RMSE=%g pitch omega RMSE=%g mean=%g\n", ...
    sweep.best.yawOmegaRmse, sweep.best.pitchOmegaRmse, sweep.best.meanOmegaRmse);

if options.ShowPlot
    figure(Name="IDCS Plant Delay Sweep", Color="w");
    tiledlayout(2, 1, TileSpacing="compact");

    nexttile;
    plot(tableOut.delaySec, tableOut.yawOmegaRmse, "r", DisplayName="yaw");
    hold on;
    plot(tableOut.delaySec, tableOut.pitchOmegaRmse, "b", DisplayName="pitch");
    plot(tableOut.delaySec, tableOut.meanOmegaRmse, "k--", DisplayName="mean");
    grid on;
    xlabel("delay (s)");
    ylabel("omega RMSE (rad/s)");
    title("Rate fit vs command delay");
    legend(Location="best");

    nexttile;
    plot(tableOut.delaySec, tableOut.yawThetaRmse, "r", DisplayName="yaw");
    hold on;
    plot(tableOut.delaySec, tableOut.pitchThetaRmse, "b", DisplayName="pitch");
    plot(tableOut.delaySec, tableOut.meanThetaRmse, "k--", DisplayName="mean");
    grid on;
    xlabel("delay (s)");
    ylabel("theta RMSE (rad)");
    title("Angle fit vs command delay");
    legend(Location="best");
end
end
