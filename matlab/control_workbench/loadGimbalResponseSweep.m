function sweep = loadGimbalResponseSweep(csvPath, options)
%LOADGIMBALRESPONSESWEEP Load a raw gimbal response sweep CSV.
%
% The sweep is produced by jetson.tools.gimbal_response_sweep. This loader
% keeps only usable encoder samples and normalizes common fields for fitting.

arguments
    csvPath (1, 1) string
    options.ManifestPath (1, 1) string = ""
end

T = readtable(csvPath, TextType="string");

required = [
    "axis", "phase", "cmd_rate_applied_rad_s", ...
    "angle_rad", "omega_rad_s", "elapsed_s", ...
    "setting_id", "trial", "direction", "accel_byte", "limit_blocked"];
missing = setdiff(required, string(T.Properties.VariableNames));
if ~isempty(missing)
    error("loadGimbalResponseSweep:MissingColumns", ...
        "Sweep CSV is missing required columns: %s", strjoin(missing, ", "));
end

valid = isfinite(T.elapsed_s) ...
    & isfinite(T.angle_rad) ...
    & isfinite(T.cmd_rate_applied_rad_s) ...
    & T.limit_blocked == 0;
if ismember("omega_rad_s", string(T.Properties.VariableNames))
    valid = valid & isfinite(T.omega_rad_s);
end
T = T(valid, :);

T.axis = string(T.axis);
T.phase = string(T.phase);
T.t = T.elapsed_s - min(T.elapsed_s, [], "omitnan");

sweep = struct();
sweep.path = csvPath;
sweep.table = T;
sweep.axes = unique(T.axis);
sweep.manifest = struct();

manifestPath = options.ManifestPath;
if manifestPath == ""
    candidate = replace(csvPath, ".csv", ".json");
    if isfile(candidate)
        manifestPath = candidate;
    end
end
if manifestPath ~= "" && isfile(manifestPath)
    sweep.manifest = jsondecode(fileread(manifestPath));
end
end
