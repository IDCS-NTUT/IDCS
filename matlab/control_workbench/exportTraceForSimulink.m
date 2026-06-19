function data = exportTraceForSimulink(tracePath, options)
%EXPORTTRACEFORSIMULINK Export trace signals as timeseries objects.

arguments
    tracePath (1, 1) string
    options.DelaySec (1, 1) double = 0.05
    options.AssignToBase (1, 1) logical = true
end

trace = loadControlTrace(tracePath);
yaw = extractAxisTrace(trace, "yaw", DelaySec=options.DelaySec);
pitch = extractAxisTrace(trace, "pitch", DelaySec=options.DelaySec);

data = struct();
data.yaw_cmd = timeseries(yaw.sample.cmd, yaw.sample.t, Name="yaw_cmd");
data.yaw_theta = timeseries(yaw.sample.theta, yaw.sample.t, Name="yaw_theta");
data.yaw_omega = timeseries(yaw.sample.omega, yaw.sample.t, Name="yaw_omega");
data.pitch_cmd = timeseries(pitch.sample.cmd, pitch.sample.t, Name="pitch_cmd");
data.pitch_theta = timeseries(pitch.sample.theta, pitch.sample.t, Name="pitch_theta");
data.pitch_omega = timeseries(pitch.sample.omega, pitch.sample.t, Name="pitch_omega");

if options.AssignToBase
    names = fieldnames(data);
    for i = 1:numel(names)
        assignin("base", names{i}, data.(names{i}));
    end
    fprintf("Exported %d timeseries objects to the base workspace.\n", numel(names));
end
end
