function trace = loadControlTrace(tracePath)
%LOADCONTROLTRACE Load a JSONL trace from tools/record_control_trace.py.
%
% trace = loadControlTrace(tracePath) returns raw event records grouped by
% stream. The raw payloads are kept so the extraction functions can evolve
% without changing the loader.

arguments
    tracePath (1, 1) string
end

fid = fopen(tracePath, "r");
if fid < 0
    error("loadControlTrace:FileOpen", "Could not open trace: %s", tracePath);
end
closer = onCleanup(@() fclose(fid));

trace = struct();
trace.path = tracePath;
trace.meta = {};
trace.detection = {};
trace.control = {};
trace.camstate = {};
trace.decodeError = {};
trace.other = {};

lineNo = 0;
while true
    line = fgetl(fid);
    if ~ischar(line)
        break
    end
    lineNo = lineNo + 1;
    line = strtrim(line);
    if line == ""
        continue
    end

    try
        record = jsondecode(line);
    catch err
        warning("loadControlTrace:BadJson", ...
            "Skipping invalid JSON at line %d: %s", lineNo, err.message);
        continue
    end

    record.lineNo = lineNo;
    record.rx_s = NaN;
    if isfield(record, "rx_monotonic_ns")
        record.rx_s = double(record.rx_monotonic_ns) * 1e-9;
    end

    recordType = string(getOptional(record, "type", ""));
    if recordType == "meta"
        trace.meta{end + 1, 1} = record; %#ok<AGROW>
        continue
    end
    if recordType == "decode_error"
        trace.decodeError{end + 1, 1} = record; %#ok<AGROW>
        continue
    end
    if recordType ~= "event"
        trace.other{end + 1, 1} = record; %#ok<AGROW>
        continue
    end

    stream = string(getOptional(record, "stream", ""));
    switch stream
        case "detection"
            trace.detection{end + 1, 1} = record; %#ok<AGROW>
        case "control"
            trace.control{end + 1, 1} = record; %#ok<AGROW>
        case "camstate"
            trace.camstate{end + 1, 1} = record; %#ok<AGROW>
        otherwise
            trace.other{end + 1, 1} = record; %#ok<AGROW>
    end
end

trace.counts = struct( ...
    "detection", numel(trace.detection), ...
    "control", numel(trace.control), ...
    "camstate", numel(trace.camstate), ...
    "decodeError", numel(trace.decodeError), ...
    "other", numel(trace.other));
end

function value = getOptional(s, fieldName, defaultValue)
if isstruct(s) && isfield(s, fieldName)
    value = s.(fieldName);
else
    value = defaultValue;
end
end
