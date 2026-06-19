function setupControlWorkbench()
%SETUPCONTROLWORKBENCH Add the IDCS MATLAB control workbench to the path.

root = fileparts(mfilename("fullpath"));
addpath(root);
fprintf("IDCS control workbench added: %s\n", root);
end
