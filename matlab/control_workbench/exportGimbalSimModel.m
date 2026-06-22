function exportGimbalSimModel(model, outputPath)
%EXPORTGIMBALSIMMODEL Write fitted virtual-gimbal parameters to JSON.
%
% The JSON is intended for a future PC SimCamera calibrated-dynamics mode.

arguments
    model (1, 1) struct
    outputPath (1, 1) string
end

clean = rmfieldPrivate(model);
text = jsonencode(clean, PrettyPrint=true);
fid = fopen(outputPath, "w");
if fid < 0
    error("exportGimbalSimModel:FileOpen", "Could not open %s for writing.", outputPath);
end
closer = onCleanup(@() fclose(fid));
fprintf(fid, "%s\n", text);
fprintf("Wrote gimbal simulation model: %s\n", outputPath);
end

function out = rmfieldPrivate(in)
out = in;
names = fieldnames(out);
for i = 1:numel(names)
    name = names{i};
    if startsWith(name, "_")
        out = rmfield(out, name);
        continue
    end
    if isstruct(out.(name))
        if numel(out.(name)) == 1
            out.(name) = rmfieldPrivate(out.(name));
        else
            for k = 1:numel(out.(name))
                out.(name)(k) = rmfieldPrivate(out.(name)(k));
            end
        end
    end
end
end
