// Bounded, single-target NVIDIA VPI CUDA backend. No detector or truth input.
#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/Pyramid.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/CropScaler.h>
#include <vpi/algo/DCFTracker.h>
#include <vpi/algo/GaussianPyramid.h>
#include <vpi/algo/HarrisCorners.h>
#include <vpi/algo/OpticalFlowPyrLK.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int MAX_POINTS = 64;
constexpr int PATCH = 24;
thread_local std::string error;
void check(VPIStatus status) {
    if (status != VPI_SUCCESS) {
        char message[VPI_MAX_STATUS_MESSAGE_LENGTH];
        vpiGetLastStatusMessage(message, sizeof(message));
        throw std::runtime_error(std::string(vpiStatusGetName(status)) + ": " + message);
    }
}
// Lock lifetime includes exceptions, so a failed frame cannot leave an array locked.
struct ArrayLock {
    VPIArray array;
    VPIArrayData data{};
    ArrayLock(VPIArray a, VPILockMode mode = VPI_LOCK_READ) : array(a) {
        check(vpiArrayLockData(a, mode, VPI_ARRAY_BUFFER_HOST_AOS, &data));
    }
    ~ArrayLock() { vpiArrayUnlock(array); }
    template<class T> T &at(int i) {
        return *reinterpret_cast<T *>(static_cast<char *>(data.buffer.aos.data)
                                       + i * data.buffer.aos.strideBytes);
    }
    int size() const { return *data.buffer.aos.sizePointer; }
    void size(int n) { *data.buffer.aos.sizePointer = n; }
};
}

extern "C" {
struct VLResult {
    float box[4];
    float psr, peak;
    int32_t count;
    float before[2 * MAX_POINTS], after[2 * MAX_POINTS], fb[MAX_POINTS];
    double elapsed_ms;
};
}

namespace {
struct Tracker {
    int width, height, buffer_kind = -1;
    int roi_width = 0, roi_height = 0;
    bool seeded = false, pending = false;
    VPIStream stream{};
    VPIPayload crop{}, dcf{}, harris{}, lk{};
    VPIImage input{}, gray{}, patches{}, feature_gray{}, response{};
    VPIPyramid previous{}, current{};
    VPIArray targets{}, localized{}, corners{}, scores{}, points{}, forward{}, backward{}, fs{}, bs{};
    VPIHarrisCornerDetectorParams hp{};
    VPIOpticalFlowPyrLKParams lp{};
    Tracker(int w, int h) : width(w), height(h) {}
    ~Tracker() {
        if (stream) vpiStreamSync(stream);
        for (auto a : {targets, localized, corners, scores, points, forward, backward, fs, bs})
            vpiArrayDestroy(a);
        for (auto p : {crop, dcf, harris, lk}) vpiPayloadDestroy(p);
        vpiImageDestroy(feature_gray); // Destroy the view before its parent.
        for (auto i : {input, gray, patches, response}) vpiImageDestroy(i);
        vpiPyramidDestroy(previous); vpiPyramidDestroy(current);
        vpiStreamDestroy(stream);
    }
    void initialize() {
        check(vpiStreamCreate(VPI_BACKEND_CUDA, &stream));
        VPIDCFTrackerCreationParams dp{};
        check(vpiInitDCFTrackerCreationParams(&dp));
        dp.flags |= VPI_DCFTRACKER_DETERMINISTIC;
        dp.featurePatchSize = PATCH;
        check(vpiCreateDCFTracker(VPI_BACKEND_CUDA, 1, 1, &dp, &dcf));
        check(vpiCreateCropScaler(VPI_BACKEND_CUDA, 1, 1, &crop));
        check(vpiInitHarrisCornerDetectorParams(&hp));
        hp.strengthThresh = 2; hp.minNMSDistance = 4;
        check(vpiCreateOpticalFlowPyrLK(VPI_BACKEND_CUDA, width, height,
                                       VPI_IMAGE_FORMAT_U8, 3, .5f, &lk));
        check(vpiInitOpticalFlowPyrLKParams(VPI_BACKEND_CUDA, &lp));
        lp.numIterations = 12;
        check(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_U8, VPI_BACKEND_CUDA, &gray));
        check(vpiImageCreate(PATCH * dp.hogCellSize, PATCH * dp.hogCellSize,
                             VPI_IMAGE_FORMAT_RGBA8, 0, &patches));
        check(vpiImageCreate(PATCH, PATCH, VPI_IMAGE_FORMAT_F32, 0, &response));
        check(vpiPyramidCreate(width, height, VPI_IMAGE_FORMAT_U8, 3, .5f, 0, &previous));
        check(vpiPyramidCreate(width, height, VPI_IMAGE_FORMAT_U8, 3, .5f, 0, &current));
        check(vpiArrayCreate(1, VPI_ARRAY_TYPE_DCF_TRACKED_BOUNDING_BOX, 0, &targets));
        check(vpiArrayCreate(1, VPI_ARRAY_TYPE_DCF_TRACKED_BOUNDING_BOX, 0, &localized));
        // Native-pixel ROI Harris, filtered before selecting at most 64 points.
        check(vpiArrayCreate(20000, VPI_ARRAY_TYPE_KEYPOINT_F32, 0, &corners));
        check(vpiArrayCreate(20000, VPI_ARRAY_TYPE_U32, 0, &scores));
        for (auto a : {&points, &forward, &backward})
            check(vpiArrayCreate(MAX_POINTS, VPI_ARRAY_TYPE_KEYPOINT_F32, 0, a));
        for (auto a : {&fs, &bs}) check(vpiArrayCreate(MAX_POINTS, VPI_ARRAY_TYPE_U8, 0, a));
    }
    void validate_box(const float *b) const {
        for (int i = 0; i < 4; ++i)
            if (!std::isfinite(b[i])) throw std::runtime_error("nonfinite box");
        if (b[0] < 0 || b[1] < 0 || b[2] < 12 || b[3] < 12 ||
            b[0] + b[2] > width || b[1] + b[3] > height)
            throw std::runtime_error("box outside image or smaller than 12 pixels");
    }
    void wrap(void *pointer, int pitch, int kind) {
        if (!pointer || (kind != 2 && pitch < 4 * width) || kind < 0 || kind > 2)
            throw std::runtime_error("invalid RGBA input");
        VPIImageData d{};
        if (kind == 2) {
            d.bufferType = VPI_IMAGE_BUFFER_EGLIMAGE;
            d.buffer.egl = pointer;
        } else {
            d.bufferType = kind == 0 ? VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR
                                    : VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;
            d.buffer.pitch.format = VPI_IMAGE_FORMAT_RGBA8;
            d.buffer.pitch.numPlanes = 1;
            auto &p = d.buffer.pitch.planes[0];
            p.width = width; p.height = height; p.pitchBytes = pitch;
            p.pBase = static_cast<VPIByte *>(pointer);
        }
        if (input && kind != buffer_kind) { vpiImageDestroy(input); input = nullptr; }
        if (!input) check(vpiImageCreateWrapper(&d, nullptr, VPI_BACKEND_CUDA, &input));
        else check(vpiImageSetWrapper(input, &d));
        buffer_kind = kind;
        check(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, input, gray, nullptr));
        check(vpiSubmitGaussianPyramidGenerator(stream, VPI_BACKEND_CUDA, gray, current, VPI_BORDER_CLAMP));
    }
    void box(const float *b, bool is_new) {
        ArrayLock a(targets, VPI_LOCK_READ_WRITE); a.size(1);
        auto &t = a.at<VPIDCFTrackedBoundingBox>(0);
        t = {};
        t.bbox = {b[0], b[1], b[2], b[3]};
        t.state = is_new ? VPI_TRACKING_STATE_NEW : VPI_TRACKING_STATE_TRACKED;
        t.seqIndex = 0; t.filterLR = .02f; t.filterChannelWeightsLR = .02f;
    }
    void crop_targets() {
        check(vpiSubmitCropScalerBatch(stream, VPI_BACKEND_CUDA, crop, &input, 1,
                                       targets, PATCH * 4, PATCH * 4, patches));
    }
    void learn() {
        crop_targets();
        check(vpiSubmitDCFTrackerUpdateBatch(stream, VPI_BACKEND_CUDA, dcf,
                                             nullptr, 0, nullptr, nullptr, patches, targets, nullptr));
    }
    int features(const float *b) {
        // Do NOT detect on the DCF resized patch. Resampling corners into source
        // coordinates introduces errors and can destroy the LK scale estimate.
        if (!harris) {
            roi_width = std::min(width, std::max(64, int(std::ceil(b[2]*1.5f/16))*16));
            roi_height = std::min(height, std::max(64, int(std::ceil(b[3]*1.5f/16))*16));
            check(vpiCreateHarrisCornerDetector(VPI_BACKEND_CUDA, roi_width, roi_height, &harris));
        }
        if (b[2] > roi_width || b[3] > roi_height)
            throw std::runtime_error("target exceeded initial feature window; new seed required");
        VPIRectangleI roi{std::clamp(int(b[0]+b[2]/2-roi_width/2),0,width-roi_width),
                          std::clamp(int(b[1]+b[3]/2-roi_height/2),0,height-roi_height),
                          roi_width, roi_height};
        check(vpiStreamSync(stream));
        if (!feature_gray) check(vpiImageCreateView(gray, &roi, VPI_BACKEND_CUDA, &feature_gray));
        else check(vpiImageSetView(feature_gray, gray, &roi));
        check(vpiSubmitHarrisCornerDetector(stream, VPI_BACKEND_CUDA, harris, feature_gray,
                                            corners, scores, &hp));
        check(vpiStreamSync(stream));
        ArrayLock c(corners), s(scores), p(points, VPI_LOCK_READ_WRITE);
        std::vector<std::pair<uint32_t, VPIKeypointF32>> candidates;
        // Avoid outside/background and box-boundary features.
        for (int i = 0; i < c.size(); ++i) {
            auto k = c.at<VPIKeypointF32>(i);
            k.x += roi.x;
            k.y += roi.y;
            if (k.x >= b[0] + 3 && k.x < b[0] + b[2] - 3 &&
                k.y >= b[1] + 3 && k.y < b[1] + b[3] - 3)
                candidates.emplace_back(s.at<uint32_t>(i), k);
        }
        std::sort(candidates.begin(), candidates.end(), [](const auto &a, const auto &b) {
            if (a.first != b.first) return a.first > b.first;
            if (a.second.y != b.second.y) return a.second.y < b.second.y;
            return a.second.x < b.second.x;
        });
        int n = 0;
        for (const auto &candidate : candidates) {
            bool near = false;
            for (int i = 0; i < n; ++i) {
                auto other = p.at<VPIKeypointF32>(i);
                if (std::hypot(other.x-candidate.second.x, other.y-candidate.second.y) < 4) {
                    near = true; break;
                }
            }
            if (!near) p.at<VPIKeypointF32>(n++) = candidate.second;
            if (n == MAX_POINTS) break;
        }
        p.size(n);
        return n;
    }
    void correlation(VLResult &r) {
        VPIImageData d{};
        check(vpiImageLockData(response, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &d));
        auto &p = d.buffer.pitch.planes[0];
        float maximum = -INFINITY; int mx = 0, my = 0;
        auto value = [&p](int x, int y) {
            return reinterpret_cast<float *>(p.pBase + y * p.pitchBytes)[x];
        };
        for (int y = 0; y < PATCH; ++y) for (int x = 0; x < PATCH; ++x)
            if (value(x, y) > maximum) { maximum = value(x, y); mx = x; my = y; }
        double sum = 0, squares = 0; int n = 0;
        for (int y = 0; y < PATCH; ++y) for (int x = 0; x < PATCH; ++x) {
            int dx = std::abs(x - mx), dy = std::abs(y - my);
            if (std::min(dx, PATCH - dx) <= 2 && std::min(dy, PATCH - dy) <= 2) continue;
            float v = value(x, y); sum += v; squares += v * v; ++n;
        }
        r.peak = maximum;
        r.psr = (maximum - sum / n) / std::sqrt(std::max(1e-12, squares / n - (sum / n) * (sum / n)));
        vpiImageUnlock(response);
    }
    void propose(VLResult &r) {
        crop_targets();
        check(vpiSubmitDCFTrackerLocalizeBatch(stream, VPI_BACKEND_CUDA, dcf, nullptr, 0,
                                               nullptr, patches, targets, localized, response, nullptr, nullptr));
        int n;
        { ArrayLock p(points); n = p.size(); }
        for (auto a : {fs, bs}) {
            ArrayLock s(a, VPI_LOCK_READ_WRITE); s.size(n);
            for (int i = 0; i < n; ++i) s.at<uint8_t>(i) = 0;
        }
        if (n) {
            check(vpiSubmitOpticalFlowPyrLK(stream, VPI_BACKEND_CUDA, lk, previous, current,
                                           points, forward, fs, &lp));
            // Submission reads the input array's size on the host. Finish the
            // producer before using its output as the reverse pass input.
            check(vpiStreamSync(stream));
            check(vpiSubmitOpticalFlowPyrLK(stream, VPI_BACKEND_CUDA, lk, current, previous,
                                           forward, backward, bs, &lp));
        }
        check(vpiStreamSync(stream));
        { ArrayLock a(localized); auto b = a.at<VPIDCFTrackedBoundingBox>(0).bbox;
          r.box[0] = b.left; r.box[1] = b.top; r.box[2] = b.width; r.box[3] = b.height; }
        correlation(r);
        ArrayLock p(points), f(forward), b(backward), a(fs), z(bs);
        for (int i = 0; i < n; ++i) {
            if (a.at<uint8_t>(i) || z.at<uint8_t>(i)) continue;
            auto p0 = p.at<VPIKeypointF32>(i), p1 = f.at<VPIKeypointF32>(i), p2 = b.at<VPIKeypointF32>(i);
            int k = r.count++;
            r.before[2*k] = p0.x; r.before[2*k+1] = p0.y;
            r.after[2*k] = p1.x; r.after[2*k+1] = p1.y;
            r.fb[k] = std::hypot(p0.x - p2.x, p0.y - p2.y);
        }
        pending = true;
    }
};
template<class F> int guarded(F f) {
    try { f(); error.clear(); return 0; }
    catch (const std::exception &e) { error = e.what(); return -1; }
}
}

extern "C" {
int vl_abi_version() { return 1; }
const char *vl_error() { return error.c_str(); }
void *vl_create(int width, int height) {
    Tracker *t = nullptr;
    if (guarded([&] {
        if (width < 64 || height < 64 || width > 4096 || height > 2160)
            throw std::runtime_error("unsupported frame dimensions");
        t = new Tracker(width, height); t->initialize();
    })) { delete t; return nullptr; }
    return t;
}
void vl_destroy(void *handle) { delete static_cast<Tracker *>(handle); }
// Input pointer must remain alive through commit/discard. kind=0 host, 1 CUDA, 2 EGL.
int vl_process(void *handle, void *rgba, int pitch, int kind, const float *seed, VLResult *out) {
    return guarded([&] {
        if (!handle || !out) throw std::runtime_error("null handle/result");
        auto &t = *static_cast<Tracker *>(handle);
        if (t.pending) throw std::runtime_error("previous proposal not committed/discarded");
        if (!t.seeded && !seed) throw std::runtime_error("seed required");
        if (t.seeded && seed) throw std::runtime_error("new seed requires new tracker");
        *out = {};
        auto start = std::chrono::steady_clock::now();
        t.wrap(rgba, pitch, kind);
        if (seed) {
            t.validate_box(seed); t.box(seed, true); t.learn();
            out->count = t.features(seed); std::copy(seed, seed+4, out->box);
            std::swap(t.previous, t.current); t.seeded = true;
        } else t.propose(*out);
        check(vpiStreamSync(t.stream));
        out->elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count();
    });
}
int vl_commit(void *handle, const float *box, int update_template) {
    return guarded([&] {
        if (!handle || !box) throw std::runtime_error("null handle/box");
        auto &t = *static_cast<Tracker *>(handle);
        if (!t.pending) throw std::runtime_error("no pending proposal");
        t.validate_box(box); t.box(box, false);
        if (update_template) t.learn();
        t.features(box);
        check(vpiStreamSync(t.stream));
        std::swap(t.previous, t.current); t.pending = false;
    });
}
int vl_discard(void *handle) {
    return guarded([&] {
        if (!handle) throw std::runtime_error("null handle");
        auto &t = *static_cast<Tracker *>(handle);
        check(vpiStreamSync(t.stream)); t.pending = false;
        // Previous accepted image/box/features/template remain unchanged.
    });
}
}
