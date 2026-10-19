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
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
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
// Contrast-limited adaptive histogram equalisation on a small U8 window, used
// only to find corners; LK and DCF keep the unmodified image. Per-tile clipped
// histograms, bilinearly blended between tile centres.
void clahe(const uint8_t *src, uint8_t *dst, int w, int h, int tx, int ty, float clip) {
    std::vector<std::array<uint8_t, 256>> lut(tx * ty);
    const int tw = (w + tx - 1) / tx, th = (h + ty - 1) / ty;
    for (int j = 0; j < ty; ++j) for (int i = 0; i < tx; ++i) {
        const int x0 = i * tw, y0 = j * th, x1 = std::min(w, x0 + tw), y1 = std::min(h, y0 + th);
        const int area = std::max(1, (x1 - x0) * (y1 - y0));
        std::array<int, 256> hist{};
        for (int y = y0; y < y1; ++y) for (int x = x0; x < x1; ++x) ++hist[src[y * w + x]];
        const int limit = std::max(1, int(clip * area / 256));
        int excess = 0;
        for (auto &v : hist) if (v > limit) { excess += v - limit; v = limit; }
        for (int k = 0; k < 256; ++k) hist[k] += excess / 256 + (k < excess % 256 ? 1 : 0);
        auto &l = lut[j * tx + i];
        int sum = 0;
        for (int k = 0; k < 256; ++k) {
            sum += hist[k];
            l[k] = uint8_t(std::min(255L, std::lround(sum * 255.0 / area)));
        }
    }
    for (int y = 0; y < h; ++y) {
        const float fy = (y + .5f) / th - .5f;
        const int j0 = std::clamp(int(std::floor(fy)), 0, ty - 1), j1 = std::min(j0 + 1, ty - 1);
        const float wy = std::clamp(fy - j0, 0.f, 1.f);
        for (int x = 0; x < w; ++x) {
            const float fx = (x + .5f) / tw - .5f;
            const int i0 = std::clamp(int(std::floor(fx)), 0, tx - 1), i1 = std::min(i0 + 1, tx - 1);
            const float wx = std::clamp(fx - i0, 0.f, 1.f);
            const uint8_t v = src[y * w + x];
            const float top = lut[j0 * tx + i0][v] * (1 - wx) + lut[j0 * tx + i1][v] * wx;
            const float bottom = lut[j1 * tx + i0][v] * (1 - wx) + lut[j1 * tx + i1][v] * wx;
            dst[y * w + x] = uint8_t(std::lround(top * (1 - wy) + bottom * wy));
        }
    }
}
}

extern "C" {
struct VLResult {
    float box[4];
    float psr, peak;
    int32_t count;
    float before[2 * MAX_POINTS], after[2 * MAX_POINTS], fb[MAX_POINTS];
    double elapsed_ms;
    uint64_t point_ids[MAX_POINTS];
    int32_t input_count, forward_count, backward_count;
    int32_t reference_count;
    float reference_before[2*MAX_POINTS], reference_after[2*MAX_POINTS], reference_fb[MAX_POINTS];
};
}

namespace {
struct Tracker {
    int width, height, buffer_kind = -1;
    int roi_width = 0, roi_height = 0;
    bool seeded = false, pending = false;
    uint64_t ids[MAX_POINTS]{}, next_id = 1;
    VLResult proposal{};
    VPIStream stream{};
    VPIPayload crop{}, dcf{}, harris{}, lk{};
    VPIImage input{}, gray{}, patches{}, feature_gray{}, feature_host{}, response{};
    // Trial switch (IDCS_VPI_FEATURE_CLAHE=<clip limit>, 0/unset = off).
    float clahe_clip = 0;
    std::vector<uint8_t> roi_source, roi_enhanced;
    bool probing_seed = false;
    std::vector<std::pair<uint32_t, VPIKeypointF32>> seed_evidence;
    VPIPyramid previous{}, current{}, reference{};
    VPIArray targets{}, localized{}, corners{}, scores{}, points{}, forward{}, backward{}, fs{}, bs{};
    VPIArray reference_points{}, reference_forward{}, reference_backward{}, reference_fs{}, reference_bs{};
    float seed_box[4]{};
    VPIHarrisCornerDetectorParams hp{};
    VPIOpticalFlowPyrLKParams lp{};
    Tracker(int w, int h) : width(w), height(h) {}
    ~Tracker() {
        if (stream) vpiStreamSync(stream);
        for (auto a : {targets, localized, corners, scores, points, forward, backward, fs, bs,
                       reference_points,reference_forward,reference_backward,reference_fs,reference_bs})
            vpiArrayDestroy(a);
        for (auto p : {crop, dcf, harris, lk}) vpiPayloadDestroy(p);
        vpiImageDestroy(feature_gray); // Destroy the view before its parent.
        vpiImageDestroy(feature_host); // Wraps roi_enhanced, destroyed after this body.
        for (auto i : {input, gray, patches, response}) vpiImageDestroy(i);
        vpiPyramidDestroy(previous); vpiPyramidDestroy(current);
        vpiPyramidDestroy(reference);
        vpiStreamDestroy(stream);
    }
    void initialize() {
        if (const char *value = std::getenv("IDCS_VPI_FEATURE_CLAHE")) {
            char *end = nullptr;
            clahe_clip = std::strtof(value, &end);
            if (end == value || *end || !(clahe_clip >= 0 && clahe_clip <= 40))
                throw std::runtime_error("IDCS_VPI_FEATURE_CLAHE must be a clip limit 0..40");
        }
        check(vpiStreamCreate(VPI_BACKEND_CUDA, &stream));
        VPIDCFTrackerCreationParams dp{};
        check(vpiInitDCFTrackerCreationParams(&dp));
        dp.flags |= VPI_DCFTRACKER_DETERMINISTIC;
        dp.featurePatchSize = PATCH;
        check(vpiCreateDCFTracker(VPI_BACKEND_CUDA, 1, 1, &dp, &dcf));
        check(vpiCreateCropScaler(VPI_BACKEND_CUDA, 1, 1, &crop));
        check(vpiInitHarrisCornerDetectorParams(&hp));
        // Bench 2026-10-19 (IMX219, lab light): at 2, a hand against the wall
        // gave 1 corner and desk equipment 10; at 0.5 they give 21 and 43 while
        // plain wall and ceiling stay at 0 (counts saturate below 0.5).
        hp.strengthThresh = .5f; hp.minNMSDistance = 4;
        check(vpiCreateOpticalFlowPyrLK(VPI_BACKEND_CUDA, width, height,
                                       VPI_IMAGE_FORMAT_U8, 3, .5f, &lk));
        check(vpiInitOpticalFlowPyrLKParams(VPI_BACKEND_CUDA, &lp));
        lp.numIterations = 12;
        // Host access is seed-only diagnostics (and the opt-in CLAHE trial).
        check(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_U8,
                             VPI_BACKEND_CUDA | VPI_BACKEND_CPU, &gray));
        check(vpiImageCreate(PATCH * dp.hogCellSize, PATCH * dp.hogCellSize,
                             VPI_IMAGE_FORMAT_RGBA8, 0, &patches));
        check(vpiImageCreate(PATCH, PATCH, VPI_IMAGE_FORMAT_F32, 0, &response));
        check(vpiPyramidCreate(width, height, VPI_IMAGE_FORMAT_U8, 3, .5f, 0, &previous));
        check(vpiPyramidCreate(width, height, VPI_IMAGE_FORMAT_U8, 3, .5f, 0, &current));
        check(vpiPyramidCreate(width, height, VPI_IMAGE_FORMAT_U8, 3, .5f, 0, &reference));
        check(vpiArrayCreate(1, VPI_ARRAY_TYPE_DCF_TRACKED_BOUNDING_BOX, 0, &targets));
        check(vpiArrayCreate(1, VPI_ARRAY_TYPE_DCF_TRACKED_BOUNDING_BOX, 0, &localized));
        // Native-pixel ROI Harris, filtered before selecting at most 64 points.
        check(vpiArrayCreate(20000, VPI_ARRAY_TYPE_KEYPOINT_F32, 0, &corners));
        check(vpiArrayCreate(20000, VPI_ARRAY_TYPE_U32, 0, &scores));
        for (auto a : {&points, &forward, &backward,&reference_points,&reference_forward,&reference_backward})
            check(vpiArrayCreate(MAX_POINTS, VPI_ARRAY_TYPE_KEYPOINT_F32, 0, a));
        for (auto a : {&fs, &bs,&reference_fs,&reference_bs}) check(vpiArrayCreate(MAX_POINTS, VPI_ARRAY_TYPE_U8, 0, a));
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
    int features(const float *b, bool append = false) {
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
        if (clahe_clip > 0) {
            // Copy the window, equalise it on the host and detect on that copy.
            roi_source.resize(size_t(roi_width) * roi_height);
            roi_enhanced.resize(roi_source.size());
            VPIImageData d{};
            check(vpiImageLockData(gray, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &d));
            const auto &plane = d.buffer.pitch.planes[0];
            for (int y = 0; y < roi_height; ++y)
                std::memcpy(&roi_source[size_t(y) * roi_width],
                            static_cast<const uint8_t *>(plane.pBase) + size_t(roi.y + y) * plane.pitchBytes + roi.x,
                            roi_width);
            vpiImageUnlock(gray);
            clahe(roi_source.data(), roi_enhanced.data(), roi_width, roi_height,
                  std::max(1, (roi_width + 64) / 128), std::max(1, (roi_height + 64) / 128), clahe_clip);
            VPIImageData e{};
            e.bufferType = VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR;
            e.buffer.pitch.format = VPI_IMAGE_FORMAT_U8;
            e.buffer.pitch.numPlanes = 1;
            auto &p = e.buffer.pitch.planes[0];
            p.width = roi_width; p.height = roi_height; p.pitchBytes = roi_width;
            p.pBase = roi_enhanced.data();
            if (!feature_host) check(vpiImageCreateWrapper(&e, nullptr, VPI_BACKEND_CUDA, &feature_host));
            else check(vpiImageSetWrapper(feature_host, &e));
            check(vpiSubmitHarrisCornerDetector(stream, VPI_BACKEND_CUDA, harris, feature_host,
                                                corners, scores, &hp));
        } else {
            if (!feature_gray) check(vpiImageCreateView(gray, &roi, VPI_BACKEND_CUDA, &feature_gray));
            else check(vpiImageSetView(feature_gray, gray, &roi));
            check(vpiSubmitHarrisCornerDetector(stream, VPI_BACKEND_CUDA, harris, feature_gray,
                                                corners, scores, &hp));
        }
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
        if (probing_seed) seed_evidence = candidates;
        int n = append ? p.size() : 0;
        for (const auto &candidate : candidates) {
            bool near = false;
            for (int i = 0; i < n; ++i) {
                auto other = p.at<VPIKeypointF32>(i);
                if (std::hypot(other.x-candidate.second.x, other.y-candidate.second.y) < 4) {
                    near = true; break;
                }
            }
            if (!near) {
                ids[n] = next_id++;
                p.at<VPIKeypointF32>(n++) = candidate.second;
            }
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
        r.input_count = n;
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
            if (!a.at<uint8_t>(i)) ++r.forward_count;
            if (a.at<uint8_t>(i) || z.at<uint8_t>(i)) continue;
            ++r.backward_count;
            auto p0 = p.at<VPIKeypointF32>(i), p1 = f.at<VPIKeypointF32>(i), p2 = b.at<VPIKeypointF32>(i);
            int k = r.count++;
            r.before[2*k] = p0.x; r.before[2*k+1] = p0.y;
            r.after[2*k] = p1.x; r.after[2*k+1] = p1.y;
            r.fb[k] = std::hypot(p0.x - p2.x, p0.y - p2.y);
            r.point_ids[k] = ids[i];
        }
        // Independent image evidence: never chain this reference through a
        // sequence of adjacent LK updates, which can walk/coalesce on an object.
        // DCF supplies a search initialisation, not a fabricated measurement.
        int rn;
        { ArrayLock rp(reference_points), rf(reference_forward,VPI_LOCK_READ_WRITE),
                    rb(reference_backward,VPI_LOCK_READ_WRITE);
          rn = rp.size(); rf.size(rn); rb.size(rn);
          for (int i = 0; i < rn; ++i) {
              auto point = rp.at<VPIKeypointF32>(i);
              rf.at<VPIKeypointF32>(i) = {
                  r.box[0]+r.box[2]/2+(point.x-seed_box[0]-seed_box[2]/2)*r.box[2]/seed_box[2],
                  r.box[1]+r.box[3]/2+(point.y-seed_box[1]-seed_box[3]/2)*r.box[3]/seed_box[3]};
              rb.at<VPIKeypointF32>(i) = point;
          }
        }
        for (auto array : {reference_fs,reference_bs}) {
            ArrayLock status(array,VPI_LOCK_READ_WRITE); status.size(rn);
            for (int i = 0; i < rn; ++i) status.at<uint8_t>(i) = 0;
        }
        auto reference_lp = lp; reference_lp.useInitialFlow = 1;
        if (rn) {
            check(vpiSubmitOpticalFlowPyrLK(stream,VPI_BACKEND_CUDA,lk,reference,current,
                                           reference_points,reference_forward,reference_fs,&reference_lp));
            check(vpiStreamSync(stream));
            check(vpiSubmitOpticalFlowPyrLK(stream,VPI_BACKEND_CUDA,lk,current,reference,
                                           reference_forward,reference_backward,reference_bs,&reference_lp));
            check(vpiStreamSync(stream));
        }
        { ArrayLock rp(reference_points), rf(reference_forward), rb(reference_backward),
                    ra(reference_fs), rz(reference_bs);
          for (int i = 0; i < rn; ++i) {
              if (ra.at<uint8_t>(i) || rz.at<uint8_t>(i)) continue;
              auto p0 = rp.at<VPIKeypointF32>(i),p1 = rf.at<VPIKeypointF32>(i),p2 = rb.at<VPIKeypointF32>(i);
              int k = r.reference_count++;
              r.reference_before[2*k] = p0.x; r.reference_before[2*k+1] = p0.y;
              r.reference_after[2*k] = p1.x; r.reference_after[2*k+1] = p1.y;
              r.reference_fb[k] = std::hypot(p0.x-p2.x,p0.y-p2.y);
          }
        }
        proposal = r;
        pending = true;
    }
};
template<class F> int guarded(F f) {
    try { f(); error.clear(); return 0; }
    catch (const std::exception &e) { error = e.what(); return -1; }
}
}

extern "C" {
int vl_abi_version() { return 4; }
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
// Additive seed-only API. Probe does not learn DCF or establish a seed/reference.
// Existing vl_process/commit ABI and lifetime semantics remain unchanged.
int vl_seed_probe(void *handle, void *rgba, int pitch, int kind, const float *box,
                  float *xy, uint32_t *scores, int capacity, int *count,
                  uint8_t *crop, int crop_capacity, int *crop_width, int *crop_height) {
    return guarded([&] {
        if (!handle || !box || !xy || !scores || !count || !crop || !crop_width || !crop_height)
            throw std::runtime_error("null seed probe argument");
        auto &t = *static_cast<Tracker *>(handle);
        if (t.pending) throw std::runtime_error("previous proposal not committed/discarded");
        if (t.seeded) throw std::runtime_error("new seed requires new tracker");
        t.validate_box(box);
        const int x = int(box[0]), y = int(box[1]);
        const int w = int(std::ceil(box[0]+box[2]))-x;
        const int h = int(std::ceil(box[1]+box[3]))-y;
        if (capacity < 20000 || crop_capacity < w*h)
            throw std::runtime_error("seed probe buffer too small");
        t.wrap(rgba, pitch, kind);
        t.probing_seed = true;
        try { t.features(box); }
        catch (...) { t.probing_seed = false; throw; }
        t.probing_seed = false;
        *count = int(t.seed_evidence.size());
        for (int i = 0; i < *count; ++i) {
            scores[i] = t.seed_evidence[i].first;
            xy[2*i] = t.seed_evidence[i].second.x;
            xy[2*i+1] = t.seed_evidence[i].second.y;
        }
        check(vpiStreamSync(t.stream));
        VPIImageData d{};
        check(vpiImageLockData(t.gray, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &d));
        const auto &plane = d.buffer.pitch.planes[0];
        for (int row = 0; row < h; ++row)
            std::memcpy(crop+row*w, static_cast<const uint8_t *>(plane.pBase)
                        + size_t(y+row)*plane.pitchBytes+x, w);
        vpiImageUnlock(t.gray);
        *crop_width = w; *crop_height = h;
        // Recreate the feature window from the refined geometry on initialization.
        vpiImageDestroy(t.feature_gray); t.feature_gray = nullptr;
        vpiImageDestroy(t.feature_host); t.feature_host = nullptr;
        vpiPayloadDestroy(t.harris); t.harris = nullptr;
        t.next_id = 1;
    });
}
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
            std::copy(seed,seed+4,t.seed_box);
            check(vpiSubmitGaussianPyramidGenerator(t.stream,VPI_BACKEND_CUDA,t.gray,t.reference,VPI_BORDER_CLAMP));
            { ArrayLock p(t.points), ref(t.reference_points,VPI_LOCK_READ_WRITE);
              ref.size(p.size());
              for (int i = 0; i < p.size(); ++i) ref.at<VPIKeypointF32>(i) = p.at<VPIKeypointF32>(i);
            }
            { ArrayLock p(t.points);
              for (int i = 0; i < out->count; ++i) {
                  auto point = p.at<VPIKeypointF32>(i);
                  out->before[2*i] = out->after[2*i] = point.x;
                  out->before[2*i+1] = out->after[2*i+1] = point.y;
                  out->point_ids[i] = t.ids[i];
              }
              out->input_count = out->forward_count = out->backward_count = out->count;
            }
            std::swap(t.previous, t.current); t.seeded = true;
        } else t.propose(*out);
        check(vpiStreamSync(t.stream));
        out->elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count();
    });
}
int vl_commit(void *handle, const float *box, int update_template, uint64_t retain_mask,
              int replenish, int32_t *stats) {
    return guarded([&] {
        if (!handle || !box) throw std::runtime_error("null handle/box");
        auto &t = *static_cast<Tracker *>(handle);
        if (!t.pending) throw std::runtime_error("no pending proposal");
        t.validate_box(box); t.box(box, false);
        if (update_template) t.learn();
        // Preserve ONLY quality-approved forward flow and identities. Never
        // replace all features with newly detected corners on every commit.
        int retained = 0;
        { ArrayLock p(t.points,VPI_LOCK_READ_WRITE);
          for (int i = 0; i < t.proposal.count; ++i) {
              if (!(retain_mask & (uint64_t(1) << i)) || t.proposal.fb[i] > 1.f) continue;
              float x = t.proposal.after[2*i], y = t.proposal.after[2*i+1];
              if (!std::isfinite(x) || !std::isfinite(y) || x < box[0] || y < box[1] ||
                  x >= box[0]+box[2] || y >= box[1]+box[3]) continue;
              bool near = false;
              for (int j = 0; j < retained; ++j) {
                  auto other = p.at<VPIKeypointF32>(j);
                  if (std::hypot(x-other.x,y-other.y) < 2.f) { near = true; break; }
              }
              if (near) continue;
              p.at<VPIKeypointF32>(retained) = {x,y};
              t.ids[retained++] = t.proposal.point_ids[i];
          }
          p.size(retained);
        }
        int total = retained;
        if (replenish && retained < 32) total = t.features(box,true);
        if (stats) { stats[0] = retained; stats[1] = total-retained; stats[2] = total; }
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
// Borrowed NVMM/EGL frames must release their wrapper BEFORE the owner unmaps
// the EGLImage or returns the GstBuffer to its pool. Internal gray/pyramids own
// their pixels. Called only after seed, commit or discard has synchronized work.
int vl_release_input(void *handle) {
    return guarded([&] {
        if (!handle) throw std::runtime_error("null handle");
        auto &t = *static_cast<Tracker *>(handle);
        if (t.pending) throw std::runtime_error("pending proposal retains input");
        check(vpiStreamSync(t.stream));
        vpiImageDestroy(t.input); t.input = nullptr; t.buffer_kind = -1;
    });
}
}
