/* Bounded, camera-only nvarguscamerasrc timing probe. No video or control output.
 * The optional GstBufferMetaData qdata contains a sensor SOF timestamp on
 * supported NVIDIA plugin versions; absent metadata is reported, not guessed.
 */
#include <gst/gst.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
    gint64 frame_num;
    gint64 timestamp;
    void *sensor_data;
} ArgusAuxData;

typedef struct {
    GMainLoop *loop;
    GstElement *pipeline;
    FILE *output;
    guint64 frames;
    guint64 sensor_meta_frames;
    guint64 plausible_sensor_frames;
    gboolean failed;
} Survey;

static guint64 monotonic_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (guint64)ts.tv_sec * 1000000000ULL + (guint64)ts.tv_nsec;
}

static GstPadProbeReturn on_buffer(GstPad *pad, GstPadProbeInfo *info, gpointer data)
{
    (void)pad;
    Survey *survey = (Survey *)data;
    GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buffer) return GST_PAD_PROBE_OK;
    const guint64 pad_ns = monotonic_ns();
    const guint64 pts_ns = GST_BUFFER_PTS(buffer);
    const GQuark quark = g_quark_from_static_string("GstBufferMetaData");
    const ArgusAuxData *meta = (const ArgusAuxData *)gst_mini_object_get_qdata(
        GST_MINI_OBJECT_CAST(buffer), quark);
    ++survey->frames;
    if (meta) {
        ++survey->sensor_meta_frames;
        const guint64 sensor_ns = (guint64)meta->timestamp;
        if (sensor_ns <= pad_ns && pad_ns - sensor_ns < 2000000000ULL)
            ++survey->plausible_sensor_frames;
        fprintf(survey->output,
            "{\"frame_index\":%llu,\"sensor_frame_number\":%lld,"
            "\"sensor_start_ns\":%lld,\"buffer_pts_ns\":%llu,"
            "\"pad_monotonic_ns\":%llu}\n",
            (unsigned long long)survey->frames, (long long)meta->frame_num,
            (long long)meta->timestamp, (unsigned long long)pts_ns,
            (unsigned long long)pad_ns);
    } else {
        fprintf(survey->output,
            "{\"frame_index\":%llu,\"sensor_metadata_available\":false,"
            "\"buffer_pts_ns\":%llu,\"pad_monotonic_ns\":%llu}\n",
            (unsigned long long)survey->frames, (unsigned long long)pts_ns,
            (unsigned long long)pad_ns);
    }
    if (survey->frames % 60 == 0) fflush(survey->output);
    return GST_PAD_PROBE_OK;
}

static gboolean on_bus(GstBus *bus, GstMessage *message, gpointer data)
{
    (void)bus;
    Survey *survey = (Survey *)data;
    if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR) {
        GError *error = NULL;
        gchar *debug = NULL;
        gst_message_parse_error(message, &error, &debug);
        fprintf(stderr, "camera pipeline error: %s (%s)\n",
            error ? error->message : "unknown", debug ? debug : "no debug");
        if (error) g_error_free(error);
        g_free(debug);
        survey->failed = TRUE;
        g_main_loop_quit(survey->loop);
    } else if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_EOS) {
        g_main_loop_quit(survey->loop);
    }
    return TRUE;
}

static gboolean on_timeout(gpointer data)
{
    Survey *survey = (Survey *)data;
    g_main_loop_quit(survey->loop);
    return G_SOURCE_REMOVE;
}

int main(int argc, char **argv)
{
    if (argc != 3 && argc != 4) {
        fprintf(stderr, "usage: probe_gst_argus_sensor_meta DURATION_SECONDS SAMPLES_JSONL [FIXED_EXPOSURE_NS]\n");
        return 2;
    }
    char *end = NULL;
    const double seconds = strtod(argv[1], &end);
    if (!end || *end || !(seconds > 0.0 && seconds <= 120.0)) {
        fprintf(stderr, "duration must be in (0, 120] seconds\n");
        return 2;
    }
    guint64 exposure_ns = 0;
    if (argc == 4) {
        end = NULL;
        exposure_ns = strtoull(argv[3], &end, 10);
        if (!end || *end || exposure_ns < 1000000ULL || exposure_ns > 15000000ULL) {
            fprintf(stderr, "fixed exposure must be 1000000..15000000 ns\n");
            return 2;
        }
    }
    FILE *output = fopen(argv[2], "w");
    if (!output) {
        perror("samples output");
        return 2;
    }
    gst_init(&argc, &argv);
    GError *parse_error = NULL;
    GstElement *pipeline = gst_parse_launch(
        "nvarguscamerasrc name=camera sensor-id=0 sensor-mode=4 ! "
        "video/x-raw(memory:NVMM),width=1280,height=720,framerate=60/1,format=NV12 ! "
        "fakesink sync=false async=false", &parse_error);
    if (!pipeline) {
        fprintf(stderr, "pipeline parse error: %s\n", parse_error ? parse_error->message : "unknown");
        if (parse_error) g_error_free(parse_error);
        fclose(output);
        return 1;
    }
    Survey survey = {0};
    survey.loop = g_main_loop_new(NULL, FALSE);
    survey.pipeline = pipeline;
    survey.output = output;
    GstElement *camera = gst_bin_get_by_name(GST_BIN(pipeline), "camera");
    if (camera && exposure_ns) {
        char range[64];
        snprintf(range, sizeof(range), "%llu %llu",
            (unsigned long long)exposure_ns, (unsigned long long)exposure_ns);
        g_object_set(camera, "exposuretimerange", range, NULL);
    }
    GstPad *source = camera ? gst_element_get_static_pad(camera, "src") : NULL;
    if (!source) {
        fprintf(stderr, "camera source pad unavailable\n");
        if (camera) gst_object_unref(camera);
        gst_object_unref(pipeline);
        g_main_loop_unref(survey.loop);
        fclose(output);
        return 1;
    }
    gst_pad_add_probe(source, GST_PAD_PROBE_TYPE_BUFFER, on_buffer, &survey, NULL);
    GstBus *bus = gst_element_get_bus(pipeline);
    gst_bus_add_watch(bus, on_bus, &survey);
    g_timeout_add((guint)(seconds * 1000.0), on_timeout, &survey);
    if (gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE)
        survey.failed = TRUE;
    else
        g_main_loop_run(survey.loop);
    gst_element_set_state(pipeline, GST_STATE_NULL);
    fflush(output);
    fclose(output);
    gst_object_unref(bus);
    gst_object_unref(source);
    gst_object_unref(camera);
    gst_object_unref(pipeline);
    g_main_loop_unref(survey.loop);
    printf("{\"frames\":%llu,\"fixed_exposure_command_ns\":%llu,\"sensor_meta_frames\":%llu,"
           "\"plausible_sensor_frames\":%llu,\"pipeline_error\":%s}\n",
           (unsigned long long)survey.frames,
           (unsigned long long)exposure_ns,
           (unsigned long long)survey.sensor_meta_frames,
           (unsigned long long)survey.plausible_sensor_frames,
           survey.failed ? "true" : "false");
    return survey.failed || !survey.frames || survey.sensor_meta_frames != survey.frames ||
           survey.plausible_sensor_frames != survey.frames ? 1 : 0;
}
