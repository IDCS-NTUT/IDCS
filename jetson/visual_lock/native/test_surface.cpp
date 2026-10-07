// Test-only pixel uploader. Not a production frame cache/readback path.
// The caller owns EGL and destroys every VPI wrapper BEFORE destroying this surface.
#include <nvbufsurface.h>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string error;
struct Surface {
    NvBufSurface *surface = nullptr;
    bool mapped = false;
    ~Surface() {
        if (mapped) NvBufSurfaceUnMapEglImage(surface,0);
        if (surface) NvBufSurfaceDestroy(surface);
    }
};
}
extern "C" {
const char *vlt_error() { return error.c_str(); }
void *vlt_create(int width,int height) {
    try {
        if (width < 64 || height < 64 || width > 4096 || height > 2160)
            throw std::runtime_error("unsupported test surface dimensions");
        auto s = std::make_unique<Surface>();
        NvBufSurfaceCreateParams p{};
        p.width = width; p.height = height;
        p.layout = NVBUF_LAYOUT_PITCH;
        p.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
        p.memType = NVBUF_MEM_SURFACE_ARRAY;
        if (NvBufSurfaceCreate(&s->surface,1,&p)) throw std::runtime_error("NvBufSurfaceCreate failed");
        s->surface->numFilled = 1;
        if (NvBufSurfaceMapEglImage(s->surface,0)) throw std::runtime_error("NvBufSurfaceMapEglImage failed");
        s->mapped = true;
        if (!s->surface->surfaceList[0].mappedAddr.eglImage) throw std::runtime_error("null EGLImage");
        error.clear();
        return s.release();
    } catch (const std::exception &e) { error = e.what(); return nullptr; }
}
void *vlt_egl_image(void *handle) {
    if (!handle) return nullptr;
    return static_cast<Surface *>(handle)->surface->surfaceList[0].mappedAddr.eglImage;
}
int vlt_upload(void *handle,void *rgba,uint64_t bytes) {
    if (!handle || !rgba) { error = "null surface/pixels"; return -1; }
    auto s = static_cast<Surface *>(handle)->surface;
    auto &p = s->surfaceList[0];
    if (bytes != uint64_t(p.width)*p.height*4) { error = "wrong RGBA byte count"; return -1; }
    // Use the SDK raw-plane uploader; no mapped CPU pointer escapes this call.
    if (Raw2NvBufSurface(static_cast<unsigned char *>(rgba),0,0,p.width,p.height,s)) {
        error = "NVMM pixel upload failed"; return -1;
    }
    error.clear(); return 0;
}
int vlt_destroy(void *handle) {
    if (!handle) return 0;
    auto s = static_cast<Surface *>(handle);
    int unmapped = s->mapped ? NvBufSurfaceUnMapEglImage(s->surface,0) : 0;
    int destroyed = s->surface ? NvBufSurfaceDestroy(s->surface) : 0;
    s->mapped = false; s->surface = nullptr; delete s;
    if (unmapped || destroyed) { error = "NVMM release failed"; return -1; }
    error.clear(); return 0;
}
}
