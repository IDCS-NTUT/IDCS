function buildGimbalPlantModel(plant, options)
%BUILDGIMBALPLANTMODEL Create a simple yaw/pitch Simulink plant model.
%
% buildGimbalPlantModel(results.plant) creates idcs_gimbal_plant.slx with two
% state-space blocks:
%   x = [theta; omega]
%   theta_dot = omega
%   omega_dot = a_u*u - a_f*omega

arguments
    plant struct = defaultPlant()
    options.ModelName (1, 1) string = "idcs_gimbal_plant"
end

modelName = char(options.ModelName);
modelFile = options.ModelName + ".slx";
if bdIsLoaded(modelName)
    close_system(modelName, 0);
end
if exist(modelFile, "file")
    delete(modelFile);
end

new_system(modelName);
open_system(modelName);
set_param(modelName, Solver="ode4", FixedStep="0.01", StopTime="5");

addAxis(modelName, "yaw", plant.yaw, 40);
addAxis(modelName, "pitch", plant.pitch, 210);

save_system(modelName);
fprintf("Created Simulink model: %s.slx\n", modelName);
end

function plant = defaultPlant()
axis = struct("a_u", 30.0, "a_f", 1.0);
plant = struct("yaw", axis, "pitch", axis);
end

function addAxis(modelName, axisName, axisPlant, y0)
axisTitle = upper(extractBefore(axisName + "_", "_"));
inName = axisName + "_cmd";
plantName = axisTitle + " Plant";
demuxName = axisTitle + " Demux";
thetaName = axisName + "_theta";
omegaName = axisName + "_omega";

add_block("simulink/Sources/In1", modelName + "/" + inName, ...
    Position=[40 y0 70 y0 + 20]);
add_block("simulink/Continuous/State-Space", modelName + "/" + plantName, ...
    Position=[145 y0 - 15 245 y0 + 35]);
add_block("simulink/Signal Routing/Demux", modelName + "/" + demuxName, ...
    Outputs="2", Position=[295 y0 - 12 300 y0 + 42]);
add_block("simulink/Sinks/Out1", modelName + "/" + thetaName, ...
    Position=[390 y0 - 20 420 y0]);
add_block("simulink/Sinks/Out1", modelName + "/" + omegaName, ...
    Position=[390 y0 + 25 420 y0 + 45]);

A = sprintf("[0 1; 0 -%.17g]", axisPlant.a_f);
B = sprintf("[0; %.17g]", axisPlant.a_u);
set_param(modelName + "/" + plantName, A=A, B=B, C="[1 0; 0 1]", D="[0; 0]");

add_line(modelName, inName + "/1", plantName + "/1");
add_line(modelName, plantName + "/1", demuxName + "/1");
add_line(modelName, demuxName + "/1", thetaName + "/1");
add_line(modelName, demuxName + "/2", omegaName + "/1");
end
