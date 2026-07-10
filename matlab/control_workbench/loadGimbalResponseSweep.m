function sweep = loadGimbalResponseSweep(csvPath, options)
%LOADGIMBALRESPONSESWEEP Load a raw gimbal response sweep CSV.
%
% The sweep is produced by jetson.tools.gimbal_response_sweep. This loader
% keeps usable encoder samples and normalizes common fields for fitting or
% plotting. Version-2 recorder columns are optional so older CSV files still
% load.

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

T = addOptionalColumns(T);
T.axis = string(T.axis);
T.phase = string(T.phase);
T.profile = string(T.profile);
T.requested_rate_source = string(T.requested_rate_source);
T.t = T.elapsed_s - min(T.elapsed_s, [], "omitnan");
rawT = T;

valid = isfinite(T.elapsed_s) ...
    & isfinite(T.angle_rad) ...
    & isfinite(T.cmd_rate_applied_rad_s) ...
    & T.limit_blocked == 0;
if ismember("valid_encoder", string(T.Properties.VariableNames))
    valid = valid & (T.valid_encoder ~= 0 | isnan(T.valid_encoder));
end
if ismember("omega_rad_s", string(T.Properties.VariableNames))
    valid = valid & isfinite(T.omega_rad_s);
end
T = T(valid, :);

sweep = struct();
sweep.path = csvPath;
sweep.table = T;
sweep.raw_table = rawT;
sweep.axes = unique(T.axis);
sweep.profiles = unique(T.profile);
sweep.manifest = struct();
sweep.quality = struct();

manifestPath = options.ManifestPath;
if manifestPath == ""
    candidate = replace(csvPath, ".csv", ".json");
    if isfile(candidate)
        manifestPath = candidate;
    end
end
if manifestPath ~= "" && isfile(manifestPath)
    sweep.manifest = jsondecode(fileread(manifestPath));
    if isfield(sweep.manifest, "quality")
        sweep.quality = sweep.manifest.quality;
    end
end
end

function T = addOptionalColumns(T)
names = string(T.Properties.VariableNames);
n = height(T);

stringDefaults = struct( ...
    "profile", "step", ...
    "requested_rate_source", "", ...
    "command_cmd_ids", "", ...
    "reply_cmd_id", "", ...
    "reply_bytes_hex", "", ...
    "reply_parsed_json", "");
stringFields = string(fieldnames(stringDefaults));
for i = 1:numel(stringFields)
    field = stringFields(i);
    if ~ismember(field, names)
        fieldName = char(field);
        T.(fieldName) = repmat(stringDefaults.(fieldName), n, 1);
    end
end

numericDefaults = struct( ...
    "segment_id", NaN, ...
    "profile_step_idx", NaN, ...
    "profile_elapsed_s", NaN, ...
    "send_ok", 1, ...
    "valid_encoder", 1, ...
    "settle_phase", 0, ...
    "settled", 0, ...
    "settle_timeout", 0, ...
    "missing_reply", 0, ...
    "send_dropped", 0, ...
    "reply_latency_ms", NaN, ...
    "pending_query_count", NaN, ...
    "dropped_query_count", NaN, ...
    "stale_reply_count", NaN);
numericFields = string(fieldnames(numericDefaults));
for i = 1:numel(numericFields)
    field = numericFields(i);
    if ~ismember(field, names)
        fieldName = char(field);
        T.(fieldName) = repmat(numericDefaults.(fieldName), n, 1);
    end
end
end
