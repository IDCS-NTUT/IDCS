// Bounded borrowed GstBuffer -> NvBufSurface -> EGLImage lifetime.
// Never maps pixel planes to CPU. GstMapInfo points at the NvBufSurface descriptor.
#include <cuda.h>
#include <cudaEGL.h>
#include <gst/gst.h>
#include <nvbufsurface.h>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string error;
struct Borrow {
    GstBuffer *buffer = nullptr;
    GstMapInfo info{};
    NvBufSurface *surface = nullptr;
    bool mapped = false, own_egl = false;
    int release() {
        int status = own_egl ? NvBufSurfaceUnMapEglImage(surface,0) : 0;
        own_egl = false;
        if (mapped) gst_buffer_unmap(buffer,&info);
        mapped = false;
        if (buffer) gst_buffer_unref(buffer);
        buffer = nullptr; surface = nullptr;
        return status;
    }
    ~Borrow() { release(); }
};
// A VPI EGLImage wrapper reads pixels only when it is created (a reused one
// returns stale frames) and costs ~2.5 ms per frame. Instead each pool buffer
// is registered with CUDA once; the registration rides on the GstBuffer as
// qdata and is undone when the buffer itself is freed (before its memory).
// Its EGL mapping is kept for that lifetime. A CUDA mapping of the same EGL
// image matched a fresh wrapper exactly after VIC writes (bench 2026-10-19).
struct Registration {
    NvBufSurface *surface = nullptr;
    void *egl = nullptr;
    CUgraphicsResource resource = nullptr;
    CUeglFrame frame{};
    bool own_egl = false;
};
GQuark registration_quark() {
    static GQuark quark = g_quark_from_static_string("idcs-visual-lock-cuda-registration");
    return quark;
}
CUcontext primary_context() {
    static CUcontext context = [] {
        CUdevice device; CUcontext c = nullptr;
        if (cuInit(0) || cuDeviceGet(&device,0) || cuDevicePrimaryCtxRetain(&c,device))
            throw std::runtime_error("CUDA primary context unavailable");
        return c;
    }();
    return context;
}
void unregister(gpointer data) {
    auto *r = static_cast<Registration *>(data);
    CUcontext c = nullptr;
    try { c = primary_context(); } catch (...) {}
    if (c && !cuCtxPushCurrent(c)) {
        cuGraphicsUnregisterResource(r->resource);
        cuCtxPopCurrent(&c);
    }
    if (r->own_egl) NvBufSurfaceUnMapEglImage(r->surface,0);
    delete r;
}
struct Clone {
    NvBufSurface *surface = nullptr;
    bool mapped = false;
    int release() {
        int unmap = mapped ? NvBufSurfaceUnMapEglImage(surface,0) : 0;
        mapped = false;
        int destroy = surface ? NvBufSurfaceDestroy(surface) : 0;
        surface = nullptr;
        return unmap || destroy ? -1 : 0;
    }
    ~Clone() { release(); }
};
}
extern "C" {
const char *vll_error() { return error.c_str(); }
void *vll_borrow(void *buffer) {
    try {
        if (!buffer) throw std::runtime_error("null GstBuffer");
        auto b = std::make_unique<Borrow>();
        b->buffer = gst_buffer_ref(static_cast<GstBuffer *>(buffer));
        if (!gst_buffer_map(b->buffer,&b->info,GST_MAP_READ)) throw std::runtime_error("GstBuffer map failed");
        b->mapped = true;
        if (b->info.size < sizeof(NvBufSurface)) throw std::runtime_error("not an NVMM descriptor");
        b->surface = reinterpret_cast<NvBufSurface *>(b->info.data);
        auto s = b->surface;
        if (s->batchSize != 1 || s->numFilled != 1 || !s->surfaceList || s->memType != NVBUF_MEM_SURFACE_ARRAY)
            throw std::runtime_error("one Jetson NVMM surface-array frame required");
        auto &p = s->surfaceList[0];
        if (p.colorFormat != NVBUF_COLOR_FORMAT_RGBA || p.width != 1280 || p.height != 720)
            throw std::runtime_error("expected 1280x720 RGBA NVMM surface");
        if (!p.mappedAddr.eglImage) {
            if (NvBufSurfaceMapEglImage(s,0)) throw std::runtime_error("NVMM EGL mapping failed");
            b->own_egl = true;
        }
        if (!p.mappedAddr.eglImage) throw std::runtime_error("null borrowed EGLImage");
        error.clear(); return b.release();
    } catch (const std::exception &e) { error = e.what(); return nullptr; }
}
void *vll_image(void *handle) {
    return handle ? static_cast<Borrow *>(handle)->surface->surfaceList[0].mappedAddr.eglImage : nullptr;
}
// Device pointer and pitch of a borrowed pool frame. The frame's EGL mapping
// must have been made by this borrow (an upstream element owning it could
// unmap it under the registration); otherwise the caller uses vll_image.
int vll_cuda(void *handle, void **pointer, int *pitch) {
    try {
        auto b = static_cast<Borrow *>(handle);
        if (!b || !b->surface || !pointer || !pitch) throw std::runtime_error("invalid borrow");
        auto &p = b->surface->surfaceList[0];
        auto *mini = GST_MINI_OBJECT_CAST(b->buffer);
        auto *r = static_cast<Registration *>(gst_mini_object_get_qdata(mini,registration_quark()));
        if (r) {
            if (r->surface != b->surface || r->egl != p.mappedAddr.eglImage)
                throw std::runtime_error("pool frame changed under its CUDA registration");
        } else {
            if (!b->own_egl) throw std::runtime_error("EGL mapping not owned by this borrow");
            CUcontext c = primary_context();
            if (cuCtxPushCurrent(c)) throw std::runtime_error("CUDA context push failed");
            auto reg = std::make_unique<Registration>();
            reg->surface = b->surface; reg->egl = p.mappedAddr.eglImage;
            CUresult status = cuGraphicsEGLRegisterImage(&reg->resource,static_cast<EGLImageKHR>(reg->egl),
                                                         CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
            if (!status) status = cuGraphicsResourceGetMappedEglFrame(&reg->frame,reg->resource,0,0);
            if (status && reg->resource) cuGraphicsUnregisterResource(reg->resource);
            cuCtxPopCurrent(&c);
            if (status) throw std::runtime_error("CUDA EGL registration failed");
            if (reg->frame.frameType != CU_EGL_FRAME_TYPE_PITCH || reg->frame.planeCount != 1) {
                unregister(reg.release());
                throw std::runtime_error("registered frame is not one pitch-linear plane");
            }
            // The registration now owns the EGL mapping; the borrow must not unmap it.
            reg->own_egl = true; b->own_egl = false;
            r = reg.release();
            gst_mini_object_set_qdata(mini,registration_quark(),r,unregister);
        }
        *pointer = r->frame.frame.pPitch[0];
        *pitch = int(r->frame.pitch);
        error.clear(); return 0;
    } catch (const std::exception &e) { error = e.what(); return -1; }
}
int vll_release(void *handle) {
    if (!handle) return 0;
    auto b = static_cast<Borrow *>(handle);
    int status = b->release(); delete b;
    if (status) { error = "borrowed EGL unmap failed"; return -1; }
    error.clear(); return 0;
}
// Keep a small, independent GPU copy of a verified input frame while the
// return video reaches the panel. Referencing the decoder's GstBuffer would
// starve its pool; copying its descriptor bytes would alias a reused surface.
void *vll_clone(void *buffer) {
    auto *borrowed = static_cast<Borrow *>(vll_borrow(buffer));
    if (!borrowed) return nullptr;
    std::unique_ptr<Borrow> source(borrowed);
    try {
        auto copy = std::make_unique<Clone>();
        NvBufSurfaceCreateParams p{};
        p.width = 1280; p.height = 720;
        p.layout = NVBUF_LAYOUT_PITCH;
        p.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
        p.memType = NVBUF_MEM_SURFACE_ARRAY;
        if (NvBufSurfaceCreate(&copy->surface,1,&p)) throw std::runtime_error("clone allocation failed");
        copy->surface->numFilled = 1;
        if (NvBufSurfaceCopy(source->surface,copy->surface)) throw std::runtime_error("NVMM frame copy failed");
        if (NvBufSurfaceMapEglImage(copy->surface,0)) throw std::runtime_error("clone EGL map failed");
        copy->mapped = true;
        if (!copy->surface->surfaceList[0].mappedAddr.eglImage)
            throw std::runtime_error("clone EGLImage missing");
        error.clear(); return copy.release();
    } catch (const std::exception &e) { error = e.what(); return nullptr; }
}
void *vll_clone_image(void *handle) {
    return handle ? static_cast<Clone *>(handle)->surface->surfaceList[0].mappedAddr.eglImage : nullptr;
}
int vll_clone_release(void *handle) {
    if (!handle) return 0;
    auto copy = static_cast<Clone *>(handle);
    int status = copy->release(); delete copy;
    if (status) { error = "clone release failed"; return -1; }
    error.clear(); return 0;
}
}
