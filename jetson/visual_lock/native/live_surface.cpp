// Bounded borrowed GstBuffer -> NvBufSurface -> EGLImage lifetime.
// Never maps pixel planes to CPU. GstMapInfo points at the NvBufSurface descriptor.
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
