/*
 * Copyright 2022 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef DRM_FENCE_H_
#define DRM_FENCE_H_

#include <stdbool.h>
#include <stdint.h>

#include "c11/threads.h"
#include "util/list.h"

/*
 * Helpers to deal with managing dma-fence fd's.  This should something that
 * can be re-used by any virtgpu native context implementation.
 */

struct drm_fence;
struct virgl_context;

/**
 * Represents a single timeline of fence-fd's.  Fences on a timeline are
 * signaled in FIFO order.
 */
struct drm_timeline {
   struct virgl_context *vctx;
   const char *name;
   int ring_idx;
   virgl_context_fence_retire fence_retire;

   int last_fence_fd;

   /* Optional blocking-wait hook.  When set, thread_sync waits by
    * (ctx_id, timestamp) via this callback instead of poll()ing a sync_file
    * fd -- lets a backend (drm2kgsl) skip the extra out-fence ioctl + fd lifecycle.
    * Returns 0 when the timestamp has retired, ETIMEDOUT to be re-waited, any
    * other value to retire (GPU fault/recovery). */
   int (*wait_timestamp)(void *arg, uint32_t ctx_id, uint32_t timestamp);
   void *wait_timestamp_arg;

   /* Pending (ctx_id, timestamp) stashed by drm_timeline_set_last_timestamp,
    * consumed by the next drm_timeline_submit_fence (mirror of last_fence_fd). */
   uint32_t last_ctx_id;
   uint32_t last_timestamp;
   bool last_ts_valid;

   struct list_head pending_fences;

   mtx_t fence_mutex;
   cnd_t fence_cond;
   thrd_t sync_thread;
   bool stop_sync_thread;
};

void drm_timeline_init(struct drm_timeline *timeline, struct virgl_context *vctx,
                       const char *name, int ring_idx,
                       virgl_context_fence_retire fence_retire);

void drm_timeline_fini(struct drm_timeline *timeline);

int drm_timeline_submit_fence(struct drm_timeline *timeline, uint32_t flags,
                              uint64_t fence_id);

void drm_timeline_set_last_fence_fd(struct drm_timeline *timeline, int fd);

/* Register the blocking-wait hook (see struct drm_timeline::wait_timestamp).
 * Once set, submit_fence uses stashed (ctx_id, timestamp) pairs. */
void drm_timeline_set_wait_timestamp_fn(
   struct drm_timeline *timeline,
   int (*fn)(void *arg, uint32_t ctx_id, uint32_t timestamp), void *arg);

/* Stash the (ctx_id, timestamp) the next submit_fence should wait on.  Mirror
 * of drm_timeline_set_last_fence_fd for the wait_timestamp path. */
void drm_timeline_set_last_timestamp(struct drm_timeline *timeline,
                                     uint32_t ctx_id, uint32_t timestamp);

/* True if a fence fd or a (ctx_id, timestamp) is stashed and awaiting
 * submit_fence. */
bool drm_timeline_has_pending(const struct drm_timeline *timeline);

#endif /* DRM_FENCE_H_ */
