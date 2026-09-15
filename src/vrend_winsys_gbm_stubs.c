/**************************************************************************
 *
 * Copyright (C) 2024 AOSP..
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/

#include "vrend_winsys_gbm.h"
#include "virgl_hw.h"
#include <drm_fourcc.h>
#include <stddef.h>

struct virgl_gbm *virgl_gbm_init(int fd) { return NULL; }

void virgl_gbm_fini(struct virgl_gbm *gbm) {
	return;
}

int virgl_gbm_convert_format(uint32_t *virgl_format, uint32_t *gbm_format) {
  /* EGL DMA-BUF imports need FourCC conversion even without a GBM allocator.
   * Keep the supported pairs identical to vrend_winsys_gbm.c. */
  static const struct {
    uint32_t drm;
    uint32_t virgl;
  } formats[] = {
    { DRM_FORMAT_RGB565, VIRGL_FORMAT_B5G6R5_UNORM },
    { DRM_FORMAT_ARGB8888, VIRGL_FORMAT_B8G8R8A8_UNORM },
    { DRM_FORMAT_XRGB8888, VIRGL_FORMAT_B8G8R8X8_UNORM },
    { DRM_FORMAT_ABGR2101010, VIRGL_FORMAT_R10G10B10A2_UNORM },
    { DRM_FORMAT_ABGR16161616F, VIRGL_FORMAT_R16G16B16A16_FLOAT },
    { DRM_FORMAT_NV12, VIRGL_FORMAT_NV12 },
    { DRM_FORMAT_ABGR8888, VIRGL_FORMAT_R8G8B8A8_UNORM },
    { DRM_FORMAT_XBGR8888, VIRGL_FORMAT_R8G8B8X8_UNORM },
    { DRM_FORMAT_R8, VIRGL_FORMAT_R8_UNORM },
    { DRM_FORMAT_YVU420, VIRGL_FORMAT_YV12 },
  };
  if (!virgl_format || !gbm_format || (*virgl_format && *gbm_format))
    return -1;
  for (size_t i = 0; i < sizeof(formats) / sizeof(formats[0]); i++) {
    if (formats[i].virgl == *virgl_format || formats[i].drm == *gbm_format) {
      *virgl_format = formats[i].virgl;
      *gbm_format = formats[i].drm;
      return 0;
    }
  }
  return -1;
}

int virgl_gbm_transfer(struct gbm_bo *bo, uint32_t direction,
                       const struct iovec *iovecs, uint32_t num_iovecs,
                       const struct vrend_transfer_info *info) {
  return 0;
}

uint32_t virgl_gbm_convert_flags(uint32_t virgl_bind_flags) { return 0; }

int virgl_gbm_export_fd(struct gbm_device *gbm, uint32_t handle,
                        int32_t *out_fd) {
  return 0;
}

int virgl_gbm_export_query(struct gbm_bo *bo,
                           struct virgl_renderer_export_query *query) {
  return 0;
}

int virgl_gbm_get_plane_width(struct gbm_bo *bo, int plane) { return 0; }

int virgl_gbm_get_plane_height(struct gbm_bo *bo, int plane) { return 0; }

int virgl_gbm_get_plane_bytes_per_pixel(struct gbm_bo *bo, int plane) {
  return 0;
}

bool virgl_gbm_external_allocation_preferred(uint32_t flags) { return false; }

bool virgl_gbm_gpu_import_required(uint32_t flags) { return false; }
