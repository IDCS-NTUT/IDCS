function figs = plotGimbalResponseSweep(csvPath, options)
%PLOTGIMBALRESPONSESweep Plot raw gimbal response sweep result data.
%
% plotGimbalResponseSweep(csvPath) opens figures for command/response
% timelines, per-trial overlays, sample gaps, and reply latency. When
% OutputDir is provided, the figures are saved as PNG files.

arguments
    csvPath (1, 1) string
    options.ManifestPath (1, 1) string = ""
    options.OutputDir (1, 1) string = ""
end

sweep = loadGimbalResponseSweep(csvPath, ManifestPath=options.ManifestPath);
T = sweep.raw_table;
if isempty(T)
    error("plotGimbalResponseSweep:NoData", "Sweep CSV has no rows to plot.");
end

if options.OutputDir ~= ""
    if ~isfolder(options.OutputDir)
        mkdir(options.OutputDir);
    end
end

figs = struct();
figs.timeline = plotTimeline(T);
saveIfRequested(figs.timeline, options.OutputDir, "timeline.png");

figs.trials = plotTrialOverlays(T);
saveIfRequested(figs.trials, options.OutputDir, "trial_overlays.png");

figs.timing = plotTiming(T);
saveIfRequested(figs.timing, options.OutputDir, "timing_quality.png");

if isfield(sweep, "quality") && ~isempty(fieldnames(sweep.quality))
    fprintf("Loaded quality summary from manifest.\n");
end
end

function fig = plotTimeline(T)
axesList = unique(T.axis);
fig = figure(Name="IDCS Gimbal Response Timeline", Color="w");
tiledlayout(numel(axesList), 2, TileSpacing="compact");

for i = 1:numel(axesList)
    axisName = axesList(i);
    A = T(T.axis == axisName, :);

    nexttile;
    hold on;
    plot(A.t, A.cmd_rate_rad_s, Color=[0.55 0.55 0.55], DisplayName="requested");
    plot(A.t, A.cmd_rate_applied_rad_s, "b", DisplayName="applied");
    plotMarkers(A);
    grid on;
    title(upper(axisName) + " command");
    xlabel("time (s)");
    ylabel("rad/s");
    legend(Location="best");

    nexttile;
    yyaxis left;
    plot(A.t, A.angle_rad, "k", DisplayName="angle");
    ylabel("angle rad");
    yyaxis right;
    plot(A.t, A.omega_rad_s, "r", DisplayName="omega");
    ylabel("omega rad/s");
    plotMarkers(A);
    grid on;
    title(upper(axisName) + " response");
    xlabel("time (s)");
end
end

function fig = plotTrialOverlays(T)
axesList = unique(T.axis);
fig = figure(Name="IDCS Gimbal Response Trial Overlays", Color="w");
tiledlayout(numel(axesList), 2, TileSpacing="compact");

for i = 1:numel(axesList)
    axisName = axesList(i);
    A = T(T.axis == axisName, :);
    groups = findgroups(A.setting_id, A.trial, A.direction, A.accel_byte);

    nexttile;
    hold on;
    for g = 1:max(groups)
        G = A(groups == g, :);
        plot(G.phase_elapsed_s, G.cmd_rate_applied_rad_s);
    end
    grid on;
    title(upper(axisName) + " applied command overlays");
    xlabel("phase time (s)");
    ylabel("rad/s");

    nexttile;
    hold on;
    for g = 1:max(groups)
        G = A(groups == g, :);
        plot(G.phase_elapsed_s, G.omega_rad_s);
    end
    grid on;
    title(upper(axisName) + " omega overlays");
    xlabel("phase time (s)");
    ylabel("rad/s");
end
end

function fig = plotTiming(T)
fig = figure(Name="IDCS Gimbal Response Timing Quality", Color="w");
tiledlayout(2, 2, TileSpacing="compact");

nexttile;
if ismember("response_rx_monotonic_ns", string(T.Properties.VariableNames))
    rx = sort(T.response_rx_monotonic_ns);
    gapsMs = diff(rx) / 1e6;
    histogram(gapsMs(isfinite(gapsMs)));
else
    text(0.1, 0.5, "response_rx_monotonic_ns missing");
end
grid on;
title("sample gap");
xlabel("ms");
ylabel("count");

nexttile;
latency = T.reply_latency_ms;
histogram(latency(isfinite(latency)));
grid on;
title("reply latency");
xlabel("ms");
ylabel("count");

nexttile;
pending = T.pending_query_count;
plot(T.t, pending, ".");
grid on;
title("pending encoder queries");
xlabel("time (s)");
ylabel("count");

nexttile;
bad = T.limit_blocked ~= 0 | T.valid_encoder == 0 | T.send_dropped ~= 0 | T.missing_reply ~= 0;
plot(T.t, double(bad), ".");
ylim([-0.1, 1.1]);
grid on;
title("quality flags");
xlabel("time (s)");
ylabel("bad sample");
end

function plotMarkers(A)
badLimit = A.limit_blocked ~= 0;
badEncoder = A.valid_encoder == 0;
sendDropped = A.send_dropped ~= 0;
if any(badLimit)
    scatter(A.t(badLimit), A.cmd_rate_applied_rad_s(badLimit), 20, "m", "filled", DisplayName="limit");
end
if any(badEncoder)
    scatter(A.t(badEncoder), A.cmd_rate_applied_rad_s(badEncoder), 20, "r", "filled", DisplayName="invalid");
end
if any(sendDropped)
    scatter(A.t(sendDropped), A.cmd_rate_applied_rad_s(sendDropped), 20, "x", DisplayName="send drop");
end
end

function saveIfRequested(fig, outputDir, fileName)
if outputDir == ""
    return
end
path = fullfile(outputDir, fileName);
exportgraphics(fig, path, Resolution=140);
fprintf("Wrote %s\n", path);
end
