# Configuration key contract

Source: `common/config_fields.json`. Maps and `[]` entries describe structure; CLI edits leaves or whole lists. No schema defaults are inserted. Numeric values are finite; absent bounds mean the complete finite domain of the declared type. Existing typed parsers add authoritative cross-field/range checks. Native tracker profile fields are inventoried here as restart-only; native nvinfer/label files remain opaque restart artifacts.

| Key | Type | Domain / default | Tier |
|---|---|---|---|
| `BaseConfig` | map | structure | `restart` |
| `BaseConfig.checkClassMatch` | int | finite | `restart` |
| `BaseConfig.minDetectorConfidence` | number | finite | `restart` |
| `DataAssociator` | map | structure | `restart` |
| `DataAssociator.associationMatcherType` | int | finite | `restart` |
| `DataAssociator.dataAssociatorType` | int | finite | `restart` |
| `DataAssociator.matchingScoreWeight4Iou` | number | finite | `restart` |
| `DataAssociator.matchingScoreWeight4SizeSimilarity` | number | finite | `restart` |
| `DataAssociator.matchingScoreWeight4VisualSimilarity` | number | finite | `restart` |
| `DataAssociator.minMatchingScore4Iou` | number | finite | `restart` |
| `DataAssociator.minMatchingScore4Overall` | number | finite | `restart` |
| `DataAssociator.minMatchingScore4SizeSimilarity` | number | finite | `restart` |
| `DataAssociator.minMatchingScore4TentativeIou` | number | finite | `restart` |
| `DataAssociator.minMatchingScore4VisualSimilarity` | number | finite | `restart` |
| `DataAssociator.tentativeDetectorConfidence` | number | finite | `restart` |
| `StateEstimator` | map | structure | `restart` |
| `StateEstimator.measurementNoiseVar4Detector` | number | finite | `restart` |
| `StateEstimator.measurementNoiseVar4Tracker` | number | finite | `restart` |
| `StateEstimator.processNoiseVar4Loc` | number | finite | `restart` |
| `StateEstimator.processNoiseVar4Size` | number | finite | `restart` |
| `StateEstimator.processNoiseVar4Vel` | number | finite | `restart` |
| `StateEstimator.stateEstimatorType` | int | finite | `restart` |
| `TargetManagement` | map | structure | `restart` |
| `TargetManagement.earlyTerminationAge` | int | finite | `restart` |
| `TargetManagement.enableBboxUnClipping` | int | finite | `restart` |
| `TargetManagement.maxShadowTrackingAge` | int | finite | `restart` |
| `TargetManagement.maxTargetsPerStream` | int | finite | `restart` |
| `TargetManagement.minIouDiff4NewTarget` | number | finite | `restart` |
| `TargetManagement.minTrackerConfidence` | number | finite | `restart` |
| `TargetManagement.outputShadowTracks` | int | finite | `restart` |
| `TargetManagement.preserveStreamUpdateOrder` | int | finite | `restart` |
| `TargetManagement.probationAge` | int | finite | `restart` |
| `TrajectoryManagement` | map | structure | `restart` |
| `TrajectoryManagement.useUniqueID` | int | finite | `restart` |
| `VisualTracker` | map | structure | `restart` |
| `VisualTracker.featureFocusOffsetFactor_y` | number | finite | `restart` |
| `VisualTracker.featureImgSizeLevel` | int | finite | `restart` |
| `VisualTracker.filterChannelWeightsLr` | number | finite | `restart` |
| `VisualTracker.filterLr` | number | finite | `restart` |
| `VisualTracker.gaussianSigma` | number | finite | `restart` |
| `VisualTracker.useColorNames` | int | finite | `restart` |
| `VisualTracker.useHog` | int | finite | `restart` |
| `VisualTracker.visualTrackerType` | int | finite | `restart` |
| `building` | map | structure | `restart` |
| `building.albedo_map` | str | declared type | `restart` |
| `building.normal_map` | str | declared type | `restart` |
| `building.uv_scale` | list | declared type | `restart` |
| `building.uv_scale[]` | number | finite | `restart` |
| `camera` | map | structure | `restart` |
| `camera.intrinsics` | map | structure | `restart` |
| `camera.intrinsics.fov_deg` | map | structure | `restart` |
| `camera.intrinsics.fov_deg.h` | number | finite | `restart` |
| `camera.intrinsics.fov_deg.v` | number | finite | `restart` |
| `camera.intrinsics.source` | str | declared type | `restart` |
| `camera.known_size_ranging` | map | structure | `restart` |
| `camera.known_size_ranging.class_aspect_ratio_limits` | map | structure | `fixed` |
| `camera.known_size_ranging.class_aspect_ratio_limits.drone` | map | structure | `fixed` |
| `camera.known_size_ranging.class_aspect_ratio_limits.drone.max` | number | finite | `fixed` |
| `camera.known_size_ranging.class_aspect_ratio_limits.drone.min` | number | finite | `fixed` |
| `camera.known_size_ranging.class_aspect_ratio_limits.person` | map | structure | `fixed` |
| `camera.known_size_ranging.class_aspect_ratio_limits.person.max` | number | finite | `fixed` |
| `camera.known_size_ranging.class_aspect_ratio_limits.person.min` | number | finite | `fixed` |
| `camera.known_size_ranging.class_sizes_m` | map | structure | `restart` |
| `camera.known_size_ranging.class_sizes_m.drone` | number | [0, inf] | `live` |
| `camera.known_size_ranging.class_sizes_m.person` | number | [0, inf] | `live` |
| `camera.known_size_ranging.dimension` | str | ["height", "width", "average", "avg"] | `restart` |
| `camera.known_size_ranging.ema_alpha` | number | finite | `fixed` |
| `camera.known_size_ranging.enabled` | bool | declared type | `restart` |
| `camera.known_size_ranging.min_pixels` | number | finite | `fixed` |
| `camera.known_size_ranging.quality` | map | structure | `fixed` |
| `camera.known_size_ranging.quality.edge_margin_px` | number | finite | `fixed` |
| `camera.known_size_ranging.quality.hold_s` | number | finite | `fixed` |
| `camera.known_size_ranging.quality.jump_slack_fraction` | number | finite | `fixed` |
| `camera.known_size_ranging.quality.max_fractional_change_per_s` | number | finite | `fixed` |
| `camera_sensor` | map | structure | `restart` |
| `camera_sensor.exposure` | map | structure | `restart` |
| `camera_sensor.exposure.max_ev` | number | finite | `restart` |
| `camera_sensor.exposure.min_ev` | number | finite | `restart` |
| `camera_sensor.exposure.mode` | str | declared type | `restart` |
| `camera_sensor.exposure.value` | number | finite | `restart` |
| `camera_sensor.lens` | map | structure | `restart` |
| `camera_sensor.lens.distortion_coeffs` | list | declared type | `restart` |
| `camera_sensor.lens.distortion_coeffs[]` | number | finite | `restart` |
| `camera_sensor.lens.vignette` | number | finite | `restart` |
| `camera_sensor.noise` | map | structure | `restart` |
| `camera_sensor.noise.read_noise_sigma` | number | finite | `restart` |
| `camera_sensor.noise.shot_noise_scale` | number | finite | `restart` |
| `camera_sensor.rolling_shutter` | map | structure | `restart` |
| `camera_sensor.rolling_shutter.enabled` | bool | declared type | `restart` |
| `camera_sensor.rolling_shutter.readout_time_ms` | number | finite | `restart` |
| `camera_sensor.white_balance` | int | finite | `restart` |
| `camstate_devices` | map | structure | `restart` |
| `camstate_devices.mag_addr` | int | finite | `restart` |
| `camstate_devices.mpu_addr` | int | finite | `restart` |
| `camstate_devices.mpu_bus` | int | finite | `restart` |
| `camstate_devices.publish_hz` | number | finite | `restart` |
| `control` | map | structure | `restart` |
| `control.aim_mode` | str | declared type | `restart` |
| `control.laser` | map | structure | `restart` |
| `control.laser.default_distance_m` | number | finite | `restart` |
| `control.laser.tolerance_px` | number | finite | `restart` |
| `control.laser.use_range` | str | declared type | `restart` |
| `control.sign_convention` | map | structure | `restart` |
| `control.sign_convention.pitch_positive` | str | declared type | `restart` |
| `control.sign_convention.yaw_positive` | str | declared type | `restart` |
| `controller` | map | structure | `restart` |
| `controller.accel_limit_rad_s2` | number | [1e-12, 20]; reader default 3.5 | `idle` |
| `controller.camera_fov_y_deg` | number / null | [1, 179]; reader default null | `restart` |
| `controller.clock` | map | structure | `restart` |
| `controller.clock.basis` | str | ["slew_limited_ntp", "same_host", "assumed"] | `restart` |
| `controller.clock.drift_ppm` | int | finite | `restart` |
| `controller.coast_s` | number | [1e-12, 2]; reader default null | `idle` |
| `controller.endpoints` | map | structure | `restart` |
| `controller.endpoints.clock` | str | declared type | `restart` |
| `controller.endpoints.diagnostics_bind` | str | declared type | `restart` |
| `controller.endpoints.gimbal_sub` | str | declared type | `restart` |
| `controller.endpoints.intent_bind` | str | declared type | `restart` |
| `controller.endpoints.manual_bind` | str | declared type | `restart` |
| `controller.endpoints.record_bind` | str | declared type | `restart` |
| `controller.endpoints.snapshot_sub` | str | declared type | `restart` |
| `controller.feedforward_accel_sigma_rad_s2` | number | [1e-12, 20] | `live` |
| `controller.feedforward_scale` | number | [0, 1] | `live` |
| `controller.idle_return_rate_rad_s` | number | finite; reader default 0.3 | `live` |
| `controller.idle_return_s` | number | [1e-12, 60]; reader default null | `live` |
| `controller.local_clock` | str | ["pc", "jetson"] | `restart` |
| `controller.manual_accel_limit_rad_s2` | number | [1e-12, 20]; reader default 2.0 | `idle` |
| `controller.manual_rate_limit_rad_s` | number | finite; reader default 0.5 (min(0.5, controller.rate_limit_rad_s)) | `idle` |
| `controller.max_capture_age_ms` | int | [1, 250] | `idle` |
| `controller.max_travel_rad` | number | [1e-12, 3.141592653589793] | `idle` |
| `controller.mode` | str | ["shadow", "live"] | `restart` |
| `controller.pitch_kp` | number | [1, 20] | `live` |
| `controller.predict` | number | [0, 1] | `live` |
| `controller.rate_limit_rad_s` | number | [0.22383847656827274, 1] | `idle` |
| `controller.source_clock` | str | ["pc_monotonic", "jetson_monotonic"] | `restart` |
| `controller.yaw_kp` | number | [1, 20] | `live` |
| `damage_by_class` | map | structure | `restart` |
| `damage_by_class.decoy` | number | finite | `restart` |
| `damage_by_class.drone` | number | finite | `restart` |
| `damage_by_class.loitering_drone` | number | finite | `restart` |
| `damage_by_class.munition` | number | finite | `restart` |
| `dataset` | map | structure | `restart` |
| `dataset.directory` | str | declared type | `restart` |
| `deepstream` | map | structure | `restart` |
| `deepstream.argus_flip_method` | int | finite | `restart` |
| `deepstream.argus_fps` | int | finite | `restart` |
| `deepstream.argus_height` | int | finite | `restart` |
| `deepstream.argus_sensor_id` | int | finite | `restart` |
| `deepstream.argus_sensor_mode` | int | finite | `restart` |
| `deepstream.argus_wb_mode` | int | finite | `restart` |
| `deepstream.argus_width` | int | finite | `restart` |
| `deepstream.id_stitching` | map | structure | `restart` |
| `deepstream.id_stitching.enabled` | bool | declared type | `live` |
| `deepstream.id_stitching.max_gap_frames` | int | [1, 10000] | `live` |
| `deepstream.input_mode` | str | ["rtp", "argus"] | `restart` |
| `deepstream.nvinfer_config` | str | declared type | `restart` |
| `deepstream.return_iframe_interval` | int | finite | `restart` |
| `deepstream.shadow_tracks` | map | structure | `restart` |
| `deepstream.shadow_tracks.enabled` | bool | declared type | `live` |
| `deepstream.shadow_tracks.max_age_frames` | int | [0, 10000] | `live` |
| `deepstream.shadow_tracks.min_confidence` | number | [0, 1] | `live` |
| `deepstream.target_selection` | bool | declared type | `restart` |
| `deepstream.tracker` | str | ["nvsort", "nvdcf", "none"] | `restart` |
| `deepstream.tracker_config` | str / null | declared type | `restart` |
| `deepstream.verified_rtp_headers` | bool | declared type | `restart` |
| `episodes` | map | structure | `restart` |
| `episodes.max_targets` | int | finite | `restart` |
| `episodes.max_targets_tensor` | int | finite | `restart` |
| `episodes.min_targets` | int | finite | `restart` |
| `episodes.num_episodes` | int | finite | `restart` |
| `episodes.scenario_weights` | map | structure | `restart` |
| `episodes.scenario_weights.crossing_targets` | number | finite | `restart` |
| `episodes.scenario_weights.decoy_distractors` | number | finite | `restart` |
| `episodes.scenario_weights.direct_rush` | number | finite | `restart` |
| `episodes.scenario_weights.high_damage_slow` | number | finite | `restart` |
| `episodes.scenario_weights.overload` | number | finite | `restart` |
| `episodes.scenario_weights.staggered_arrivals` | number | finite | `restart` |
| `gimbal` | map | structure | `restart` |
| `gimbal.camstate_pitch_sign` | number | [-1.0, 1.0] | `restart` |
| `gimbal.camstate_source` | str | declared type | `restart` |
| `gimbal.camstate_yaw_sign` | number | [-1.0, 1.0] | `restart` |
| `gimbal.counts_per_rev` | int | finite | `fixed` |
| `gimbal.encoder_imu_horizon_enabled` | bool | declared type | `restart` |
| `gimbal.feedback_hz` | number | finite | `restart` |
| `gimbal.intent_command_runtime_ms` | int | finite | `restart` |
| `gimbal.intent_watchdog_ms` | number | finite | `restart` |
| `gimbal.parameter_file` | null / str | declared type | `restart` |
| `gimbal.pitch_accel_byte` | int | [0, 255] | `restart` |
| `gimbal.pitch_divergence_thresh_rad` | number | finite | `restart` |
| `gimbal.pitch_encoder_authority` | str | declared type | `restart` |
| `gimbal.pitch_gear_ratio` | number | [1e-12, inf] | `restart` |
| `gimbal.pitch_max_rad` | number | finite | `restart` |
| `gimbal.pitch_min_rad` | number | finite | `restart` |
| `gimbal.pitch_motor_a_addr` | int | [1, 255] | `restart` |
| `gimbal.pitch_motor_a_sign` | number | [-1.0, 1.0] | `restart` |
| `gimbal.pitch_motor_b_addr` | int | [1, 255] | `restart` |
| `gimbal.pitch_motor_b_enabled` | bool | declared type | `restart` |
| `gimbal.pitch_motor_b_sign` | number | [-1.0, 1.0] | `restart` |
| `gimbal.pitch_rate_limit_rad_s` | number | [1e-12, inf] | `restart` |
| `gimbal.position_feedback` | str | declared type | `restart` |
| `gimbal.respond_on_writes` | bool | declared type | `restart` |
| `gimbal.startup_calibration_enabled` | bool | declared type | `restart` |
| `gimbal.startup_encoder_zero_enabled` | bool | declared type | `restart` |
| `gimbal.steps_per_rev` | int | finite | `fixed` |
| `gimbal.yaw_accel_byte` | int | [0, 255] | `restart` |
| `gimbal.yaw_addr` | int | [1, 255] | `restart` |
| `gimbal.yaw_gear_ratio` | number | [1e-12, inf] | `restart` |
| `gimbal.yaw_group_addr` | null / number | [1, 255] | `restart` |
| `gimbal.yaw_max_rad` | null / number | finite | `restart` |
| `gimbal.yaw_min_rad` | null / number | finite | `restart` |
| `gimbal.yaw_motor_sign` | number | [-1.0, 1.0] | `restart` |
| `gimbal.yaw_rate_limit_rad_s` | number | [1e-12, inf] | `restart` |
| `ground` | map | structure | `restart` |
| `ground.albedo_map` | str | declared type | `restart` |
| `ground.color` | list | declared type | `restart` |
| `ground.color[]` | int | finite | `restart` |
| `ground.metallic` | number | finite | `restart` |
| `ground.normal_map` | str | declared type | `restart` |
| `ground.roughness` | number | finite | `restart` |
| `ground.uv_scale` | list | declared type | `restart` |
| `ground.uv_scale[]` | number | finite | `restart` |
| `ibl` | map | structure | `restart` |
| `ibl.enabled` | bool | declared type | `restart` |
| `ibl.hdr_env_map` | str | declared type | `restart` |
| `ibl.intensity` | number | finite | `restart` |
| `ibl.irradiance_cubemap_resolution` | int | finite | `restart` |
| `ibl.max_prefilter_mip` | int | finite | `restart` |
| `ibl.prefilter_cubemap_resolution` | int | finite | `restart` |
| `laser` | map | structure | `restart` |
| `laser.dir_cam` | map | structure | `restart` |
| `laser.dir_cam.x` | number | finite | `live` |
| `laser.dir_cam.y` | number | finite | `live` |
| `laser.dir_cam.z` | number | finite | `live` |
| `laser.offset_m` | map | structure | `restart` |
| `laser.offset_m.x` | number | finite | `live` |
| `laser.offset_m.y` | number | finite | `live` |
| `laser.offset_m.z` | number | finite | `live` |
| `laser.render` | map | structure | `fixed` |
| `laser.render.beam_length_m` | number | finite | `fixed` |
| `laser.render.color_bgr` | list | declared type | `fixed` |
| `laser.render.color_bgr[]` | int | finite | `fixed` |
| `laser.render.hit_tolerance_px` | number | finite | `fixed` |
| `laser.render.thickness_px` | int | finite | `fixed` |
| `model` | map | structure | `restart` |
| `model.attention_dropout` | number | finite | `restart` |
| `model.attention_heads` | int | finite | `restart` |
| `model.context_size` | int | finite | `restart` |
| `model.hidden_size` | int | finite | `restart` |
| `model.use_target_attention` | bool | declared type | `restart` |
| `motors` | map | structure | `restart` |
| `motors.1` | map | structure | `restart` |
| `motors.1.parameters` | list | declared type | `restart` |
| `motors.1.parameters[]` | int | finite | `restart` |
| `motors.2` | map | structure | `restart` |
| `motors.2.parameters` | list | declared type | `restart` |
| `motors.2.parameters[]` | int | finite | `restart` |
| `motors.3` | map | structure | `restart` |
| `motors.3.parameters` | list | declared type | `restart` |
| `motors.3.parameters[]` | int | finite | `restart` |
| `net` | map | structure | `restart` |
| `net.header_push` | str | declared type | `restart` |
| `net.jetson_ip` | str | declared type | `restart` |
| `net.pc_bind_ip` | str / null | declared type | `restart` |
| `net.return_ip` | str | declared type | `restart` |
| `net.return_mirror_ips` | list | declared type | `restart` |
| `net.return_mirror_ips[]` | str | declared type | `restart` |
| `net.return_rtcp_feedback_port` | int | [1, 65535] | `restart` |
| `net.return_rtcp_sender_report_port` | int | [1, 65535] | `restart` |
| `net.rtp_port` | int | [1, 65535] | `restart` |
| `net.rtp_return_port` | int | [1, 65535] | `restart` |
| `net.zmq_control` | str | declared type | `restart` |
| `net.zmq_control_diagnostics` | str | declared type | `restart` |
| `net.zmq_gimbal_state` | str | declared type | `restart` |
| `net.zmq_manual_state` | str | declared type | `restart` |
| `net.zmq_operator_command` | str | declared type | `restart` |
| `net.zmq_operator_pc` | str | declared type | `restart` |
| `net.zmq_operator_selection` | str | declared type | `restart` |
| `net.zmq_perception_v2` | str | declared type | `restart` |
| `net.zmq_return_frame_identity` | str | declared type | `restart` |
| `net.zmq_sim_activity` | str | declared type | `restart` |
| `net.zmq_source_clock_sync` | str | declared type | `restart` |
| `net.zmq_visual_roi_perception` | str | declared type | `restart` |
| `net.zmq_visual_roi_seed` | str | declared type | `restart` |
| `normalization` | map | structure | `restart` |
| `normalization.max_angle_rad` | number | finite | `restart` |
| `normalization.max_breakthrough_time_s` | number | finite | `restart` |
| `normalization.max_closing_speed_m_s` | number | finite | `restart` |
| `normalization.max_damage_weight` | number | finite | `restart` |
| `normalization.max_distance_m` | number | finite | `restart` |
| `normalization.max_time_to_engage_s` | number | finite | `restart` |
| `normalization.max_track_age_s` | number | finite | `restart` |
| `normalization.max_track_observations` | int | finite | `restart` |
| `operator` | map | structure | `restart` |
| `operator.health_file` | str | declared type | `restart` |
| `operator.hud` | map | structure | `restart` |
| `operator.hud.diagnostics_sub` | str | declared type | `restart` |
| `operator.hud.perception_sub` | str | declared type | `restart` |
| `operator.hud.pose_sub` | str | declared type | `restart` |
| `operator.hud.push_endpoint` | str | declared type | `restart` |
| `operator.lock_lost_s` | number | [0.1, 10] | `live` |
| `operator.modes` | map | structure | `restart` |
| `operator.modes.camera` | map | structure | `restart` |
| `operator.modes.camera.label` | str | declared type | `restart` |
| `operator.modes.camera.pc_ready_units` | list | declared type | `restart` |
| `operator.modes.camera.pc_ready_units[]` | str | declared type | `restart` |
| `operator.modes.camera.pc_start` | list | declared type | `restart` |
| `operator.modes.camera.pc_start[]` | str | declared type | `restart` |
| `operator.modes.camera.pc_stop` | list | declared type | `restart` |
| `operator.modes.camera.pc_stop[]` | str | declared type | `restart` |
| `operator.modes.camera.ready_units` | list | declared type | `restart` |
| `operator.modes.camera.ready_units[]` | str | declared type | `restart` |
| `operator.modes.camera.source_clock` | str | declared type | `restart` |
| `operator.modes.camera.start` | list | declared type | `restart` |
| `operator.modes.camera.start[]` | str | declared type | `restart` |
| `operator.modes.camera.stop` | list | declared type | `restart` |
| `operator.modes.camera.stop[]` | str | declared type | `restart` |
| `operator.modes.sim` | map | structure | `restart` |
| `operator.modes.sim.label` | str | declared type | `restart` |
| `operator.modes.sim.pc_ready_units` | list | declared type | `restart` |
| `operator.modes.sim.pc_ready_units[]` | str | declared type | `restart` |
| `operator.modes.sim.pc_start` | list | declared type | `restart` |
| `operator.modes.sim.pc_start[]` | str | declared type | `restart` |
| `operator.modes.sim.pc_stop` | list | declared type | `restart` |
| `operator.modes.sim.pc_stop[]` | str | declared type | `restart` |
| `operator.modes.sim.ready_units` | list | declared type | `restart` |
| `operator.modes.sim.ready_units[]` | str | declared type | `restart` |
| `operator.modes.sim.source_clock` | str | declared type | `restart` |
| `operator.modes.sim.start` | list | declared type | `restart` |
| `operator.modes.sim.start[]` | str | declared type | `restart` |
| `operator.modes.sim.stop` | list | declared type | `restart` |
| `operator.modes.sim.stop[]` | str | declared type | `restart` |
| `operator.modes.sim_swarm` | map | structure | `restart` |
| `operator.modes.sim_swarm.label` | str | declared type | `restart` |
| `operator.modes.sim_swarm.pc_ready_units` | list | declared type | `restart` |
| `operator.modes.sim_swarm.pc_ready_units[]` | str | declared type | `restart` |
| `operator.modes.sim_swarm.pc_start` | list | declared type | `restart` |
| `operator.modes.sim_swarm.pc_start[]` | str | declared type | `restart` |
| `operator.modes.sim_swarm.pc_stop` | list | declared type | `restart` |
| `operator.modes.sim_swarm.pc_stop[]` | str | declared type | `restart` |
| `operator.modes.sim_swarm.ready_units` | list | declared type | `restart` |
| `operator.modes.sim_swarm.ready_units[]` | str | declared type | `restart` |
| `operator.modes.sim_swarm.source_clock` | str | declared type | `restart` |
| `operator.modes.sim_swarm.start` | list | declared type | `restart` |
| `operator.modes.sim_swarm.start[]` | str | declared type | `restart` |
| `operator.modes.sim_swarm.stop` | list | declared type | `restart` |
| `operator.modes.sim_swarm.stop[]` | str | declared type | `restart` |
| `operator.modes.standby` | map | structure | `restart` |
| `operator.modes.standby.label` | str | declared type | `restart` |
| `operator.modes.standby.pc_stop` | list | declared type | `restart` |
| `operator.modes.standby.pc_stop[]` | str | declared type | `restart` |
| `operator.modes.standby.start` | list | declared type | `restart` |
| `operator.modes.standby.start[]` | str | declared type | `restart` |
| `operator.modes.standby.stop` | list | declared type | `restart` |
| `operator.modes.standby.stop[]` | str | declared type | `restart` |
| `operator.motor_units` | list | declared type | `restart` |
| `operator.motor_units[]` | str | declared type | `restart` |
| `operator.readiness_timeout_s` | int | finite | `restart` |
| `operator.recorder_unit` | str | declared type | `restart` |
| `operator.sim_activity_sub` | str | declared type | `restart` |
| `operator.sim_score_endpoint` | str | declared type | `restart` |
| `operator.sim_selection_bind` | str | declared type | `restart` |
| `operator.state_file` | str | declared type | `restart` |
| `output` | map | structure | `restart` |
| `output.directory` | str | declared type | `restart` |
| `pbr` | map | structure | `restart` |
| `pbr.default_albedo` | list | declared type | `restart` |
| `pbr.default_albedo[]` | int | finite | `restart` |
| `pbr.default_metallic` | number | finite | `restart` |
| `pbr.default_roughness` | number | finite | `restart` |
| `pbr.normal_y_flip` | bool | declared type | `restart` |
| `pbr.texture_search_paths` | list | declared type | `restart` |
| `pbr.texture_search_paths[]` | str | declared type | `restart` |
| `pbr.use_normal_maps` | bool | declared type | `restart` |
| `perception` | map | structure | `restart` |
| `perception.class_labels` | map | structure | `restart` |
| `perception.class_labels.0` | str | declared type | `restart` |
| `perception.class_labels.1` | str | declared type | `restart` |
| `performance` | map | structure | `restart` |
| `performance.anisotropic_filter` | int | finite | `restart` |
| `performance.msaa_samples` | int | finite | `restart` |
| `performance.quality` | str | declared type | `restart` |
| `performance.reduce_post_resolution` | number | finite | `restart` |
| `planner` | map | structure | `restart` |
| `planner.beam_width` | int | finite | `restart` |
| `planner.exact_search_limit` | int | finite | `restart` |
| `planner.pitch_accel_limit_rad_s2` | number | finite | `restart` |
| `planner.pitch_rate_limit_rad_s` | number | finite | `restart` |
| `planner.switch_absolute_damage_gain` | number | finite | `restart` |
| `planner.switch_relative_improvement` | number | finite | `restart` |
| `planner.timing` | map | structure | `restart` |
| `planner.timing.base_track_lock_s` | number | finite | `restart` |
| `planner.timing.confirm_distance_scale_s_per_m` | number | finite | `restart` |
| `planner.timing.confirm_time_s` | number | finite | `restart` |
| `planner.timing.effect_distance_scale_s_per_m` | number | finite | `restart` |
| `planner.timing.effect_time_s` | number | finite | `restart` |
| `planner.timing.low_conf_penalty_s` | number | finite | `restart` |
| `planner.timing.low_conf_threshold` | number | finite | `restart` |
| `planner.timing.low_continuity_penalty_s` | number | finite | `restart` |
| `planner.timing.min_track_observations` | int | finite | `restart` |
| `planner.timing.missing_range_penalty_s` | number | finite | `restart` |
| `planner.timing.predictive_penalty_s` | number | finite | `restart` |
| `planner.timing.recover_track_lock_s` | number | finite | `restart` |
| `planner.timing.search_track_lock_s` | number | finite | `restart` |
| `planner.timing.settle_margin_s` | number | finite | `restart` |
| `planner.yaw_accel_limit_rad_s2` | number | finite | `restart` |
| `planner.yaw_rate_limit_rad_s` | number | finite | `restart` |
| `post_processing` | map | structure | `restart` |
| `post_processing.bloom` | map | structure | `restart` |
| `post_processing.bloom.blur_radius` | int | finite | `restart` |
| `post_processing.bloom.enabled` | bool | declared type | `restart` |
| `post_processing.bloom.intensity` | number | finite | `restart` |
| `post_processing.bloom.threshold` | number | finite | `restart` |
| `post_processing.dof` | map | structure | `restart` |
| `post_processing.dof.aperture` | number | finite | `restart` |
| `post_processing.dof.enabled` | bool | declared type | `restart` |
| `post_processing.dof.focus_distance` | number | finite | `restart` |
| `post_processing.fxaa` | bool | declared type | `restart` |
| `post_processing.motion_blur` | map | structure | `restart` |
| `post_processing.motion_blur.enabled` | bool | declared type | `restart` |
| `post_processing.motion_blur.sample_count` | int | finite | `restart` |
| `post_processing.ssao` | map | structure | `restart` |
| `post_processing.ssao.enabled` | bool | declared type | `restart` |
| `post_processing.ssao.intensity` | number | finite | `restart` |
| `post_processing.ssao.radius` | number | finite | `restart` |
| `post_processing.taa` | map | structure | `restart` |
| `post_processing.taa.enabled` | bool | declared type | `restart` |
| `post_processing.taa.feedback` | number | finite | `restart` |
| `post_processing.tone_mapping` | str | declared type | `restart` |
| `presets` | map | structure | `restart` |
| `presets.high` | map | structure | `restart` |
| `presets.high.pbr` | map | structure | `restart` |
| `presets.high.pbr.default_roughness` | number | finite | `restart` |
| `presets.high.performance` | map | structure | `restart` |
| `presets.high.performance.msaa_samples` | int | finite | `restart` |
| `presets.high.performance.quality` | str | declared type | `restart` |
| `presets.high.post_processing` | map | structure | `restart` |
| `presets.high.post_processing.bloom` | map | structure | `restart` |
| `presets.high.post_processing.bloom.enabled` | bool | declared type | `restart` |
| `presets.high.post_processing.ssao` | map | structure | `restart` |
| `presets.high.post_processing.ssao.enabled` | bool | declared type | `restart` |
| `presets.high.post_processing.taa` | map | structure | `restart` |
| `presets.high.post_processing.taa.enabled` | bool | declared type | `restart` |
| `presets.high.shadows` | map | structure | `restart` |
| `presets.high.shadows.enabled` | bool | declared type | `restart` |
| `presets.high.shadows.resolution` | int | finite | `restart` |
| `presets.low` | map | structure | `restart` |
| `presets.low.pbr` | map | structure | `restart` |
| `presets.low.pbr.default_roughness` | number | finite | `restart` |
| `presets.low.performance` | map | structure | `restart` |
| `presets.low.performance.msaa_samples` | int | finite | `restart` |
| `presets.low.performance.quality` | str | declared type | `restart` |
| `presets.low.post_processing` | map | structure | `restart` |
| `presets.low.post_processing.bloom` | map | structure | `restart` |
| `presets.low.post_processing.bloom.enabled` | bool | declared type | `restart` |
| `presets.low.post_processing.ssao` | map | structure | `restart` |
| `presets.low.post_processing.ssao.enabled` | bool | declared type | `restart` |
| `presets.low.post_processing.taa` | map | structure | `restart` |
| `presets.low.post_processing.taa.enabled` | bool | declared type | `restart` |
| `presets.low.shadows` | map | structure | `restart` |
| `presets.low.shadows.enabled` | bool | declared type | `restart` |
| `presets.medium` | map | structure | `restart` |
| `presets.medium.pbr` | map | structure | `restart` |
| `presets.medium.pbr.default_roughness` | number | finite | `restart` |
| `presets.medium.performance` | map | structure | `restart` |
| `presets.medium.performance.msaa_samples` | int | finite | `restart` |
| `presets.medium.performance.quality` | str | declared type | `restart` |
| `presets.medium.post_processing` | map | structure | `restart` |
| `presets.medium.post_processing.bloom` | map | structure | `restart` |
| `presets.medium.post_processing.bloom.enabled` | bool | declared type | `restart` |
| `presets.medium.post_processing.ssao` | map | structure | `restart` |
| `presets.medium.post_processing.ssao.enabled` | bool | declared type | `restart` |
| `presets.medium.shadows` | map | structure | `restart` |
| `presets.medium.shadows.enabled` | bool | declared type | `restart` |
| `presets.medium.shadows.resolution` | int | finite | `restart` |
| `recorder` | map | structure | `restart` |
| `recorder.max_total_gb` | int | finite | `restart` |
| `recorder.root` | str | declared type | `restart` |
| `recorder.segment_s` | int | finite | `restart` |
| `recorder.streams` | list | declared type | `restart` |
| `recorder.streams[]` | map | structure | `restart` |
| `recorder.streams[].endpoint` | str | declared type | `restart` |
| `recorder.streams[].name` | str | declared type | `restart` |
| `recorder.streams[].subscribe` | str | declared type | `restart` |
| `renderer` | map | structure | `restart` |
| `renderer.backend` | str | declared type | `restart` |
| `renderer.debug` | int | finite | `restart` |
| `renderer.far` | number | finite | `restart` |
| `renderer.height` | int | finite | `restart` |
| `renderer.near` | number | finite | `restart` |
| `renderer.width` | int | finite | `restart` |
| `rpi` | map | structure | `restart` |
| `rpi.gpio` | map | structure | `restart` |
| `rpi.gpio.input_active_levels` | map | structure | `restart` |
| `rpi.gpio.input_active_levels.emergency` | str | declared type | `restart` |
| `rpi.gpio.input_modes` | map | structure | `restart` |
| `rpi.gpio.input_modes.control_switch` | str | declared type | `restart` |
| `rpi.gpio.input_pull` | str | declared type | `restart` |
| `rpi.gpio.input_pulls` | map | structure | `restart` |
| `rpi.gpio.input_pulls.control_switch` | str | declared type | `restart` |
| `rpi.gpio.input_pulls.fire_control` | str | declared type | `restart` |
| `rpi.gpio.input_pulls.safety` | str | declared type | `restart` |
| `rpi.gpio.inputs` | map | structure | `restart` |
| `rpi.gpio.inputs.control_switch` | int | finite | `restart` |
| `rpi.gpio.inputs.emergency` | int | finite | `restart` |
| `rpi.gpio.inputs.fire` | int | finite | `restart` |
| `rpi.gpio.inputs.fire_control` | int | finite | `restart` |
| `rpi.gpio.inputs.safety` | int | finite | `restart` |
| `rpi.gpio.output_active_level` | str | declared type | `restart` |
| `rpi.gpio.outputs` | map | structure | `restart` |
| `rpi.gpio.outputs.fire_control_lamp` | int | finite | `restart` |
| `rpi.gpio.outputs.green_light` | int | finite | `restart` |
| `rpi.gpio.outputs.red_light` | int | finite | `restart` |
| `rpi.gpio.outputs.safety_lamp` | int | finite | `restart` |
| `rpi.gpio.outputs.yellow_light` | int | finite | `restart` |
| `rpi.gpio.shared_lamp_inputs` | map | structure | `restart` |
| `rpi.gpio.shared_lamp_inputs.fire_control` | str | declared type | `restart` |
| `rpi.gpio.shared_lamp_inputs.safety` | str | declared type | `restart` |
| `rpi.operator_display` | map | structure | `restart` |
| `rpi.operator_display.hud_sim_bind` | str | declared type | `restart` |
| `rpi.operator_display.panel_state_endpoint` | str | declared type | `restart` |
| `rpi.operator_display.roi_control_endpoint` | str | declared type | `restart` |
| `rpi.operator_display.roi_lamp_endpoint` | str | declared type | `restart` |
| `rpi.operator_display.session_env` | map | structure | `restart` |
| `rpi.operator_display.session_env.WAYLAND_DISPLAY` | str | declared type | `restart` |
| `rpi.operator_display.session_env.XDG_RUNTIME_DIR` | str | declared type | `restart` |
| `rpi.operator_display.sink` | str | declared type | `restart` |
| `rpi.runtime_control` | map | structure | `restart` |
| `rpi.runtime_control.indicators` | map | structure | `fixed` |
| `rpi.runtime_control.indicators.control_stale_s` | number | finite | `fixed` |
| `rpi.runtime_control.indicators.fire_control_off_delay_s` | number | finite | `fixed` |
| `rpi.runtime_control.indicators.hardware_stale_s` | number | finite | `fixed` |
| `rpi.runtime_control.indicators.perception_stale_s` | number | finite | `fixed` |
| `rpi.runtime_control.indicators.safety_off_delay_s` | number | finite | `fixed` |
| `rpi.runtime_control.invert_pitch` | bool | declared type | `restart` |
| `rpi.runtime_control.invert_yaw` | bool | declared type | `restart` |
| `rpi.runtime_control.sim_panel_pub_endpoint` | str | declared type | `restart` |
| `serial_io` | map | structure | `restart` |
| `serial_io.actuation_state_heartbeat_ms` | int | finite | `restart` |
| `serial_io.publish_actuation_state` | bool | declared type | `restart` |
| `serial_io.publish_command_events` | bool | declared type | `restart` |
| `serial_io.schedule` | list | declared type | `restart` |
| `serial_io.schedule[]` | map | structure | `restart` |
| `serial_io.schedule[].addr` | int | finite | `restart` |
| `serial_io.schedule[].expect_reply` | bool | declared type | `restart` |
| `serial_io.schedule[].expected_len` | int | finite | `restart` |
| `serial_io.schedule[].func` | str | declared type | `restart` |
| `serial_io.schedule[].interval_ms` | int | finite | `restart` |
| `serial_io.schedule[].name` | str | declared type | `restart` |
| `serial_io.schedule[].payload` | list | declared type | `restart` |
| `serial_io.schedule[].payload[]` | str | declared type | `restart` |
| `serial_io.schedule[].priority` | str | declared type | `restart` |
| `serial_io.schedule[].target` | str | declared type | `restart` |
| `serial_io.startup` | list | declared type | `restart` |
| `serial_io.startup[]` | map | structure | `restart` |
| `serial_io.startup[].addr` | int | finite | `restart` |
| `serial_io.startup[].expect_reply` | bool | declared type | `restart` |
| `serial_io.startup[].expected_len` | int | finite | `restart` |
| `serial_io.startup[].func` | str | declared type | `restart` |
| `serial_io.startup[].name` | str | declared type | `restart` |
| `serial_io.startup[].payload` | list | declared type | `restart` |
| `serial_io.startup[].payload[]` | int | finite | `restart` |
| `serial_io.startup[].priority` | str | declared type | `restart` |
| `serial_io.startup[].target` | str | declared type | `restart` |
| `shadows` | map | structure | `restart` |
| `shadows.bias` | number | finite | `restart` |
| `shadows.cascade_count` | int | finite | `restart` |
| `shadows.debug_stats` | bool | declared type | `restart` |
| `shadows.enabled` | bool | declared type | `restart` |
| `shadows.method` | str | declared type | `restart` |
| `shadows.resolution` | int | finite | `restart` |
| `shadows.softness` | number | finite | `restart` |
| `shadows.strength` | number | finite | `restart` |
| `sim` | map | structure | `restart` |
| `sim.camera` | map | structure | `restart` |
| `sim.camera.fov_x_deg` | number | finite | `restart` |
| `sim.camera.fov_y_deg` | number | finite | `restart` |
| `sim.debug` | bool | declared type | `restart` |
| `sim.freeze_frame` | bool | declared type | `restart` |
| `sim.jetson_cam_state_stale_timeout_s` | number | finite | `restart` |
| `sim.plant_model` | map | structure | `restart` |
| `sim.plant_model.mode` | str | declared type | `restart` |
| `sim.renderer` | str | declared type | `restart` |
| `sim.renderer_opts` | map | structure | `restart` |
| `sim.renderer_opts.draw_ground_grid` | bool | declared type | `restart` |
| `sim.renderer_opts.target_render_mode` | str | declared type | `restart` |
| `sim.scene` | map | structure | `restart` |
| `sim.scene.buildings` | list | declared type | `restart` |
| `sim.scene.buildings[]` | map | structure | `restart` |
| `sim.scene.buildings[].base_centre` | list | declared type | `restart` |
| `sim.scene.buildings[].base_centre[]` | number | finite | `restart` |
| `sim.scene.buildings[].color` | list | declared type | `restart` |
| `sim.scene.buildings[].color[]` | int | finite | `restart` |
| `sim.scene.buildings[].footprint` | list | declared type | `restart` |
| `sim.scene.buildings[].footprint[]` | number | finite | `restart` |
| `sim.scene.buildings[].height` | number | finite | `restart` |
| `sim.scene.cubes` | list | declared type | `restart` |
| `sim.scene.cubes[]` | str | declared type | `restart` |
| `sim.scene.mode` | str | declared type | `restart` |
| `sim.scene.planner_eval` | map | structure | `restart` |
| `sim.scene.planner_eval.breach_zone` | str | declared type | `restart` |
| `sim.scene.planner_eval.engage_dwell_s` | number | finite | `restart` |
| `sim.scene.planner_eval.match_radius_px` | number | finite | `restart` |
| `sim.scene.planner_eval.max_active_targets` | int | finite | `restart` |
| `sim.scene.planner_eval.seed` | int | finite | `restart` |
| `sim.scene.planner_eval.spawn_distance_m` | list | declared type | `restart` |
| `sim.scene.planner_eval.spawn_distance_m[]` | number | finite | `restart` |
| `sim.scene.planner_eval.spawn_interval_s` | list | declared type | `restart` |
| `sim.scene.planner_eval.spawn_interval_s[]` | number | finite | `restart` |
| `sim.scene.planner_eval.spawn_view` | str | declared type | `restart` |
| `sim.scene.planner_eval.speed_m_s` | list | declared type | `restart` |
| `sim.scene.planner_eval.speed_m_s[]` | number | finite | `restart` |
| `sim.scene.planner_eval.sprite` | str | declared type | `restart` |
| `sim.scene.planner_eval.width` | number | finite | `restart` |
| `sim.scene.targets` | list | declared type | `restart` |
| `sim.scene.targets[]` | map | structure | `restart` |
| `sim.scene.targets[].color` | list | declared type | `restart` |
| `sim.scene.targets[].color[]` | int | finite | `restart` |
| `sim.scene.targets[].height` | number | finite | `restart` |
| `sim.scene.targets[].movement` | map | structure | `restart` |
| `sim.scene.targets[].movement.dynamics` | map | structure | `restart` |
| `sim.scene.targets[].movement.dynamics.arrival_radius_m` | number | finite | `restart` |
| `sim.scene.targets[].movement.dynamics.enabled` | bool | declared type | `restart` |
| `sim.scene.targets[].movement.dynamics.max_accel_m_s2` | number | finite | `restart` |
| `sim.scene.targets[].movement.dynamics.max_decel_m_s2` | number | finite | `restart` |
| `sim.scene.targets[].movement.points` | list | declared type | `restart` |
| `sim.scene.targets[].movement.points[]` | list | declared type | `restart` |
| `sim.scene.targets[].movement.points[][]` | number | finite | `restart` |
| `sim.scene.targets[].movement.speed_m_s` | number | finite | `restart` |
| `sim.scene.targets[].movement.type` | str | declared type | `restart` |
| `sim.scene.targets[].orientation` | list | declared type | `restart` |
| `sim.scene.targets[].orientation[]` | number | finite | `restart` |
| `sim.scene.targets[].sprite` | str | declared type | `restart` |
| `sim.scene.targets[].width` | number | finite | `restart` |
| `sim.use_jetson_cam_state` | bool | declared type | `restart` |
| `sky` | map | structure | `restart` |
| `sky.hdr` | str | declared type | `restart` |
| `sky.intensity` | number | finite | `restart` |
| `sky.procedural` | map | structure | `restart` |
| `sky.procedural.horizon_color` | list | declared type | `restart` |
| `sky.procedural.horizon_color[]` | number | finite | `restart` |
| `sky.procedural.model` | str | declared type | `restart` |
| `sky.procedural.sun_azimuth` | number | finite | `restart` |
| `sky.procedural.sun_color` | list | declared type | `restart` |
| `sky.procedural.sun_color[]` | number | finite | `restart` |
| `sky.procedural.sun_elevation` | number | finite | `restart` |
| `sky.procedural.turbidity` | number | finite | `restart` |
| `sky.procedural.zenith_color` | list | declared type | `restart` |
| `sky.procedural.zenith_color[]` | number | finite | `restart` |
| `sky.type` | str | declared type | `restart` |
| `source` | str | declared type | `restart` |
| `split` | map | structure | `restart` |
| `split.heldout` | number | finite | `restart` |
| `split.seed` | int | finite | `restart` |
| `split.test` | number | finite | `restart` |
| `split.train` | number | finite | `restart` |
| `split.val` | number | finite | `restart` |
| `swarm_eval` | map | structure | `restart` |
| `swarm_eval.beam_width` | int | [0, inf] | `live` |
| `swarm_eval.damage_by_class` | map | structure | `live` |
| `swarm_eval.damage_by_class.drone` | number | [0, inf] | `live` |
| `swarm_eval.damage_by_class.person` | number | [0, inf] | `live` |
| `swarm_eval.default_damage_weight` | number | [0, inf] | `live` |
| `swarm_eval.enabled` | bool | declared type | `live` |
| `swarm_eval.exact_search_limit` | int | [0, inf] | `live` |
| `swarm_eval.excluded_target_classes` | list | declared type | `live` |
| `swarm_eval.excluded_target_classes[]` | str | declared type | `live` |
| `swarm_eval.hostile_levels` | list | declared type | `live` |
| `swarm_eval.hostile_levels[]` | str | declared type | `live` |
| `swarm_eval.learned_model` | map | structure | `live` |
| `swarm_eval.learned_model.async_worker` | bool | declared type | `restart` |
| `swarm_eval.learned_model.backend` | str | ["auto", "torch", "tensorrt"] | `restart` |
| `swarm_eval.learned_model.enabled` | bool | declared type | `restart` |
| `swarm_eval.learned_model.fallback_to_planner` | bool | declared type | `restart` |
| `swarm_eval.learned_model.max_result_age_ms` | number | finite | `restart` |
| `swarm_eval.learned_model.max_targets_tensor` | int | finite | `restart` |
| `swarm_eval.learned_model.max_update_rate_hz` | null / number | finite | `restart` |
| `swarm_eval.learned_model.model_path` | str | declared type | `restart` |
| `swarm_eval.learned_model.normalization` | map | structure | `fixed` |
| `swarm_eval.learned_model.normalization.max_angle_rad` | number | finite | `fixed` |
| `swarm_eval.learned_model.normalization.max_breakthrough_time_s` | number | finite | `fixed` |
| `swarm_eval.learned_model.normalization.max_closing_speed_m_s` | number | finite | `fixed` |
| `swarm_eval.learned_model.normalization.max_damage_weight` | number | finite | `fixed` |
| `swarm_eval.learned_model.normalization.max_distance_m` | number | finite | `fixed` |
| `swarm_eval.learned_model.normalization.max_time_to_engage_s` | number | finite | `fixed` |
| `swarm_eval.learned_model.normalization.max_track_age_s` | number | finite | `fixed` |
| `swarm_eval.learned_model.normalization.max_track_observations` | int | finite | `fixed` |
| `swarm_eval.max_engage_distance_m` | number | [0, inf] | `live` |
| `swarm_eval.switch_absolute_damage_gain` | number | [0, inf] | `live` |
| `swarm_eval.switch_relative_improvement` | number | [0, inf] | `live` |
| `swarm_eval.timing` | map | structure | `live` |
| `swarm_eval.timing.base_track_lock_s` | number | [0, inf] | `live` |
| `swarm_eval.timing.confirm_distance_scale_s_per_m` | number | [0, inf] | `live` |
| `swarm_eval.timing.confirm_time_s` | number | [0, inf] | `live` |
| `swarm_eval.timing.effect_distance_scale_s_per_m` | number | [0, inf] | `live` |
| `swarm_eval.timing.effect_time_s` | number | [0, inf] | `live` |
| `swarm_eval.timing.low_conf_penalty_s` | number | [0, inf] | `live` |
| `swarm_eval.timing.low_conf_threshold` | number | [0, inf] | `live` |
| `swarm_eval.timing.low_continuity_penalty_s` | number | [0, inf] | `live` |
| `swarm_eval.timing.min_track_observations` | int | [0, inf] | `live` |
| `swarm_eval.timing.missing_range_penalty_s` | number | [0, inf] | `live` |
| `swarm_eval.timing.predictive_penalty_s` | number | [0, inf] | `live` |
| `swarm_eval.timing.recover_track_lock_s` | number | [0, inf] | `live` |
| `swarm_eval.timing.search_track_lock_s` | number | [0, inf] | `live` |
| `swarm_eval.timing.settle_margin_s` | number | [0, inf] | `live` |
| `threat_eval` | map | structure | `restart` |
| `threat_eval.defended_asset` | map | structure | `restart` |
| `threat_eval.defended_asset.id` | str | declared type | `restart` |
| `threat_eval.defended_asset.position_world` | list | declared type | `restart` |
| `threat_eval.defended_asset.position_world[]` | number | finite | `restart` |
| `threat_eval.enabled` | bool | declared type | `restart` |
| `threat_eval.zones` | map | structure | `restart` |
| `threat_eval.zones.critical` | map | structure | `live` |
| `threat_eval.zones.critical.radius_m` | number | [0, inf] | `live` |
| `threat_eval.zones.critical.type` | str | ["circle"] | `live` |
| `threat_eval.zones.restricted` | map | structure | `live` |
| `threat_eval.zones.restricted.radius_m` | number | [0, inf] | `live` |
| `threat_eval.zones.restricted.type` | str | ["circle"] | `live` |
| `threat_eval.zones.warning` | map | structure | `live` |
| `threat_eval.zones.warning.radius_m` | number | [0, inf] | `live` |
| `threat_eval.zones.warning.type` | str | ["circle"] | `live` |
| `training` | map | structure | `restart` |
| `training.batch_size` | int | finite | `restart` |
| `training.early_stopping_patience` | int | finite | `restart` |
| `training.epochs` | int | finite | `restart` |
| `training.learning_rate` | number | finite | `restart` |
| `training.regret_loss_weight` | number | finite | `restart` |
| `training.regret_sample_weight` | number | finite | `restart` |
| `training.seed` | int | finite | `restart` |
| `training.threat_class_loss_weight` | number | finite | `restart` |
| `training.value_loss_weight` | number | finite | `restart` |
| `training.weight_decay` | number | finite | `restart` |
| `tuning` | map | structure | `restart` |
| `tuning.agreement` | map | structure | `restart` |
| `tuning.agreement.max_median_cost_error` | number | finite | `restart` |
| `tuning.agreement.max_sim_choice_penalty` | number | finite | `restart` |
| `tuning.axes` | map | structure | `restart` |
| `tuning.axes.pitch` | map | structure | `restart` |
| `tuning.axes.pitch.addr` | int | finite | `restart` |
| `tuning.axes.pitch.motor_sign` | int | finite | `restart` |
| `tuning.axes.pitch.step_sign` | int | finite | `restart` |
| `tuning.axes.yaw` | map | structure | `restart` |
| `tuning.axes.yaw.addr` | int | finite | `restart` |
| `tuning.axes.yaw.motor_sign` | int | finite | `restart` |
| `tuning.axes.yaw.step_sign` | int | finite | `restart` |
| `tuning.config` | str | declared type | `restart` |
| `tuning.config_extra` | str | declared type | `restart` |
| `tuning.detection` | map | structure | `restart` |
| `tuning.detection.dropout` | number | finite | `restart` |
| `tuning.detection.noise_rad` | number | finite | `restart` |
| `tuning.feedforward_configs` | list | declared type | `restart` |
| `tuning.feedforward_configs[]` | list | declared type | `restart` |
| `tuning.feedforward_configs[][]` | number | finite | `restart` |
| `tuning.fit` | map | structure | `restart` |
| `tuning.fit.max_theta_rmse_rad` | number | finite | `restart` |
| `tuning.gear_ratio` | number | finite | `restart` |
| `tuning.hardware_sweep` | map | structure | `restart` |
| `tuning.hardware_sweep.guard_rad` | number | finite | `restart` |
| `tuning.hardware_sweep.kp_factors` | list | declared type | `restart` |
| `tuning.hardware_sweep.kp_factors[]` | number | finite | `restart` |
| `tuning.hardware_sweep.repeats` | int | finite | `restart` |
| `tuning.home` | map | structure | `restart` |
| `tuning.home.acc` | int | finite | `restart` |
| `tuning.home.speed_rpm` | int | finite | `restart` |
| `tuning.home.tolerance_counts` | int | finite | `restart` |
| `tuning.latency` | map | structure | `restart` |
| `tuning.latency.min_clock_verified_fraction` | number | finite | `restart` |
| `tuning.latency.min_tracking_ticks` | int | finite | `restart` |
| `tuning.limits` | map | structure | `restart` |
| `tuning.limits.accel_byte` | int | finite | `restart` |
| `tuning.limits.accel_bytes` | list | declared type | `restart` |
| `tuning.limits.accel_bytes[]` | int | finite | `restart` |
| `tuning.limits.levels` | list | declared type | `restart` |
| `tuning.limits.levels[]` | int | finite | `restart` |
| `tuning.limits.margin_levels` | int | finite | `restart` |
| `tuning.limits.max_rate_rad_s` | number | finite | `restart` |
| `tuning.limits.min_rate_rad_s` | number | finite | `restart` |
| `tuning.live_ab` | map | structure | `restart` |
| `tuning.live_ab.max_capture_rms_mrad` | map | structure | `restart` |
| `tuning.live_ab.max_capture_rms_mrad.pitch` | number | finite | `restart` |
| `tuning.live_ab.max_capture_rms_mrad.yaw` | number | finite | `restart` |
| `tuning.live_ab.min_pairs` | int | finite | `restart` |
| `tuning.loop` | map | structure | `restart` |
| `tuning.loop.accel_limit_rad_s2` | number | finite | `restart` |
| `tuning.loop.fps` | number | finite | `restart` |
| `tuning.loop.tick_hz` | number | finite | `restart` |
| `tuning.port` | str | declared type | `restart` |
| `tuning.scenarios` | list | declared type | `restart` |
| `tuning.scenarios[]` | str | declared type | `restart` |
| `tuning.sim` | map | structure | `restart` |
| `tuning.sim.kp_max` | number | finite | `restart` |
| `tuning.sim.kp_min` | number | finite | `restart` |
| `tuning.sim.kp_points` | int | finite | `restart` |
| `tuning.sysid` | map | structure | `restart` |
| `tuning.sysid.common` | map | structure | `restart` |
| `tuning.sysid.common.command_refresh_s` | number | finite | `restart` |
| `tuning.sysid.common.command_runtime_ms` | int | finite | `restart` |
| `tuning.sysid.common.directions` | str | declared type | `restart` |
| `tuning.sysid.common.pre_roll_s` | number | finite | `restart` |
| `tuning.sysid.common.profile` | str | declared type | `restart` |
| `tuning.sysid.common.rest_s` | number | finite | `restart` |
| `tuning.sysid.common.sample_hz` | number | finite | `restart` |
| `tuning.sysid.min_motion_fraction` | number | finite | `restart` |
| `tuning.sysid.train` | map | structure | `restart` |
| `tuning.sysid.train.post_roll_s` | number | finite | `restart` |
| `tuning.sysid.train.rates` | list | declared type | `restart` |
| `tuning.sysid.train.rates[]` | number | finite | `restart` |
| `tuning.sysid.train.repeat` | int | finite | `restart` |
| `tuning.sysid.train.step_s` | number | finite | `restart` |
| `tuning.sysid.validation` | map | structure | `restart` |
| `tuning.sysid.validation.post_roll_s` | number | finite | `restart` |
| `tuning.sysid.validation.rates` | list | declared type | `restart` |
| `tuning.sysid.validation.rates[]` | number | finite | `restart` |
| `tuning.sysid.validation.repeat` | int | finite | `restart` |
| `tuning.sysid.validation.step_s` | number | finite | `restart` |
| `video` | map | structure | `restart` |
| `video.active_profile` | str | declared type | `restart` |
| `video.active_return_profile` | str | declared type | `restart` |
| `video.profiles` | map | structure | `restart` |
| `video.profiles.1080p` | map | structure | `restart` |
| `video.profiles.1080p.bitrate_kbps` | int | finite | `restart` |
| `video.profiles.1080p.fps` | int | finite | `restart` |
| `video.profiles.1080p.height` | int | finite | `restart` |
| `video.profiles.1080p.width` | int | finite | `restart` |
| `video.profiles.720p` | map | structure | `restart` |
| `video.profiles.720p.bitrate_kbps` | int | finite | `restart` |
| `video.profiles.720p.fps` | int | finite | `restart` |
| `video.profiles.720p.height` | int | finite | `restart` |
| `video.profiles.720p.width` | int | finite | `restart` |
| `video.profiles.720p30` | map | structure | `restart` |
| `video.profiles.720p30.bitrate_kbps` | int | finite | `restart` |
| `video.profiles.720p30.fps` | int | finite | `restart` |
| `video.profiles.720p30.height` | int | finite | `restart` |
| `video.profiles.720p30.width` | int | finite | `restart` |
| `video.return_jitter_ms` | int | finite | `restart` |
| `video.return_queue_buffers` | int | finite | `restart` |
