/*
 * Copyright 2022 Google LLC
 * Copyright 2026 DroidVM
 * SPDX-License-Identifier: MIT
 *
 * Deliberately MIT, not the GPL this project defaults to: this file began as the KGSL
 * analogue of virglrenderer's msm_renderer.c and carries Google's copyright above, and
 * MIT is what virglrenderer requires of contributions -- which is what this file is for.
 *
 * KGSL native-context backend.
 *
 * The guest runs the ordinary drm/msm (freedreno/turnip) userspace over the
 * vdrm/virtio transport, so on the wire it speaks the *msm* protocol
 * (msm_proto.h) and registers under VIRTGPU_DRM_CONTEXT_MSM.  The host GPU on
 * this device, however, is not a DRM/msm device -- it is a Qualcomm KGSL
 * character device (/dev/kgsl-3d0).  This backend is therefore an
 * msm-protocol *server* implemented on top of KGSL ioctls: it is the KGSL
 * analogue of msm_renderer.c.  The guest is unaware that the host is KGSL.
 */

#ifndef KGSL_RENDERER_H_
#define KGSL_RENDERER_H_

#include <stddef.h>

#include "drm_hw.h"

struct virgl_context;

/*
 * Unlike the DRM backends, KGSL is probed/opened via a plain character device
 * (no drmGetVersion()).  drm_renderer.c special-cases this using the
 * .char_device flag on the backend descriptor and calls these directly.
 */
#define KGSL_DEVICE_NODE "/dev/kgsl-3d0"

int drm2kgsl_renderer_probe(int fd, struct virgl_renderer_capset_drm *capset);

struct virgl_context *drm2kgsl_renderer_create(int fd, size_t debug_len,
                                           const char *debug_name);

#endif /* KGSL_RENDERER_H_ */
