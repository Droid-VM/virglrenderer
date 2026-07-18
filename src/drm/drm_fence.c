/*
 * Copyright 2022 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include <errno.h>
#include <inttypes.h>
#include <poll.h>
#include <string.h>

#include "virgl_context.h"
#include "virgl_fence.h"
#include "virgl_util.h"

#include "util/os_file.h"
#include "util/u_atomic.h"
#include "util/u_thread.h"

#include "drm_fence.h"
#include "drm_util.h"

/**
 * Tracking for a single fence on a timeline
 */
struct drm_fence {
   int fd;               /* sync_file fd, or -1 for the by_timestamp variant */
   uint32_t flags;
   uint64_t fence_id;
   /* by_timestamp variant (fd < 0): wait on (ctx_id, timestamp) via the
    * timeline's wait_timestamp hook instead of poll()ing an fd. */
   bool by_timestamp;
   uint32_t ctx_id;
   uint32_t timestamp;
   struct list_head node;
};

static void
drm_fence_destroy(struct drm_fence *fence)
{
   if (fence->fd >= 0)
      close(fence->fd);
   list_del(&fence->node);
   free(fence);
}

static struct drm_fence *
drm_fence_create_timestamp(uint32_t ctx_id, uint32_t timestamp, uint32_t flags,
                           uint64_t fence_id)
{
   struct drm_fence *fence = calloc(1, sizeof(*fence));

   if (!fence)
      return NULL;

   fence->fd = -1;
   fence->by_timestamp = true;
   fence->ctx_id = ctx_id;
   fence->timestamp = timestamp;
   fence->flags = flags;
   fence->fence_id = fence_id;

   return fence;
}

static struct drm_fence *
drm_fence_create(int fd, uint32_t flags, uint64_t fence_id)
{
   struct drm_fence *fence = calloc(1, sizeof(*fence));

   if (!fence)
      return NULL;

   fence->fd = os_dupfd_cloexec(fd);

   if (fence->fd < 0) {
      free(fence);
      return NULL;
   }

   fence->flags = flags;
   fence->fence_id = fence_id;

   return fence;
}

static int
thread_sync(void *arg)
{
   struct drm_timeline *timeline = arg;

   u_thread_setname(timeline->name);

   mtx_lock(&timeline->fence_mutex);
   while (!timeline->stop_sync_thread) {
      if (list_is_empty(&timeline->pending_fences)) {
         if (cnd_wait(&timeline->fence_cond, &timeline->fence_mutex))
            drm_err("error waiting on fence condition");
         continue;
      }

      struct drm_fence *fence =
         list_first_entry(&timeline->pending_fences, struct drm_fence, node);
      int ret;

      mtx_unlock(&timeline->fence_mutex);
      if (fence->by_timestamp) {
         /* Block until the GPU retires the timestamp (no sync_file fd). A NULL
          * hook degrades to signal-immediately. The hook uses a bounded chunk
          * timeout so shutdown stays responsive; ETIMEDOUT means still working. */
         ret = timeline->wait_timestamp
                  ? timeline->wait_timestamp(timeline->wait_timestamp_arg,
                                             fence->ctx_id, fence->timestamp)
                  : 0;
      } else {
         ret = poll(&(struct pollfd){fence->fd, POLLIN}, 1, -1); /* wait forever */
      }
      mtx_lock(&timeline->fence_mutex);

      if (fence->by_timestamp) {
         if (ret == ETIMEDOUT && !timeline->stop_sync_thread)
            continue; /* GPU still working -- keep pending and re-wait */
         drm_dbg("fence signaled: %p (%" PRIu64 ")", (void*)fence, fence->fence_id);
         timeline->fence_retire(timeline->vctx, timeline->ring_idx, fence->fence_id);
         drm_fence_destroy(fence);
      } else if (ret == 1) {
         drm_dbg("fence signaled: %p (%" PRIu64 ")", (void*)fence, fence->fence_id);
         timeline->fence_retire(timeline->vctx, timeline->ring_idx, fence->fence_id);
         drm_fence_destroy(fence);
      } else if (ret != 0) {
         drm_err("poll failed: %s", strerror(errno));
      }
   }
   mtx_unlock(&timeline->fence_mutex);

   return 0;
}

void
drm_timeline_init(struct drm_timeline *timeline, struct virgl_context *vctx,
                  const char *name, int ring_idx,
                  virgl_context_fence_retire fence_retire)
{
   timeline->vctx = vctx;
   timeline->name = name;
   timeline->ring_idx = ring_idx;
   timeline->fence_retire = fence_retire;

   timeline->last_fence_fd = -1;

   list_inithead(&timeline->pending_fences);

   mtx_init(&timeline->fence_mutex, mtx_plain);
   cnd_init(&timeline->fence_cond);

   timeline->sync_thread = u_thread_create(thread_sync, timeline);
}

void
drm_timeline_fini(struct drm_timeline *timeline)
{
   /* signal thread_sync to shutdown: */
   mtx_lock(&timeline->fence_mutex);
   timeline->stop_sync_thread = true;
   cnd_signal(&timeline->fence_cond);
   mtx_unlock(&timeline->fence_mutex);

   /* wait for thread_sync to exit: */
   thrd_join(timeline->sync_thread, NULL);

   if (timeline->last_fence_fd != -1)
      close(timeline->last_fence_fd);

   /* cleanup remaining fences: */
   list_for_each_entry_safe (struct drm_fence, fence, &timeline->pending_fences, node) {
      drm_fence_destroy(fence);
   }

   cnd_destroy(&timeline->fence_cond);
   mtx_destroy(&timeline->fence_mutex);
}

int
drm_timeline_submit_fence(struct drm_timeline *timeline, uint32_t flags,
                          uint64_t fence_id)
{
   struct drm_fence *fence;

   if (timeline->last_fence_fd != -1) {
      fence = drm_fence_create(timeline->last_fence_fd, flags, fence_id);
      if (!fence)
         return -ENOMEM;

      drm_dbg("fence: %p (%" PRIu64 ")", (void*)fence, fence->fence_id);
      virgl_fence_set_fd(fence_id, fence->fd);

      close(timeline->last_fence_fd);
      timeline->last_fence_fd = -1;
   } else if (timeline->last_ts_valid) {
      fence = drm_fence_create_timestamp(timeline->last_ctx_id,
                                         timeline->last_timestamp, flags,
                                         fence_id);
      if (!fence)
         return -ENOMEM;

      drm_dbg("fence: %p (%" PRIu64 ") ts ctx=%u ts=%u", (void*)fence,
              fence->fence_id, fence->ctx_id, fence->timestamp);

      /* No sync_file fd to export -- the wait is by (ctx, timestamp). */
      timeline->last_ts_valid = false;
   } else {
      return -EINVAL;
   }

   mtx_lock(&timeline->fence_mutex);
   list_addtail(&fence->node, &timeline->pending_fences);
   cnd_signal(&timeline->fence_cond);
   mtx_unlock(&timeline->fence_mutex);

   return 0;
}

/* takes ownership of the fd */
void
drm_timeline_set_last_fence_fd(struct drm_timeline *timeline, int fd)
{
   if (timeline->last_fence_fd != -1)
      close(timeline->last_fence_fd);
   timeline->last_fence_fd = fd;
}

void
drm_timeline_set_wait_timestamp_fn(
   struct drm_timeline *timeline,
   int (*fn)(void *arg, uint32_t ctx_id, uint32_t timestamp), void *arg)
{
   timeline->wait_timestamp = fn;
   timeline->wait_timestamp_arg = arg;
}

void
drm_timeline_set_last_timestamp(struct drm_timeline *timeline, uint32_t ctx_id,
                                uint32_t timestamp)
{
   timeline->last_ctx_id = ctx_id;
   timeline->last_timestamp = timestamp;
   timeline->last_ts_valid = true;
}

bool
drm_timeline_has_pending(const struct drm_timeline *timeline)
{
   return timeline->last_fence_fd != -1 || timeline->last_ts_valid;
}
