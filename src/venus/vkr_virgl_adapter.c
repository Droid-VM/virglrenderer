/*
 * Copyright 2026 Droid-VM
 * SPDX-License-Identifier: MIT
 *
 * In-process struct virgl_context implementation over the vendored
 * upstream-1.3 venus renderer (vkr_renderer_*, ctx_id-keyed, self-tabling).
 * Upstream only drives that API from the render server; DroidVM stays single
 * process (one debug loop, no exec/SELinux/packaging surface), so this file
 * plays the render server's role, mirroring src/server/render_context.c call
 * sequences.
 *
 * Guest-allocated blobs (the Gunyah protected-VM route) arrive exactly like
 * the drm2kgsl backend's: crosvm turns the guest pages into a udmabuf and
 * parks the fd via set_guest_blob_fd(); the next get_blob() on the same
 * context consumes it, imports it into vkr (so a later vkAllocateMemory with
 * VkImportMemoryResourceInfoMESA binds the guest pages) and hands the same fd
 * back as the blob.
 */

#include "vkr_virgl_adapter.h"

#include <errno.h>
#include <stdio.h>
#include <fcntl.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "util/hash_table.h"
#include "util/u_pointer.h"
#include "virgl_context.h"
#include "virgl_resource.h"
#include "virgl_util.h"
#include "virglrenderer.h"

#include "vkr_renderer.h"

struct vkr_adapter_ctx {
   struct virgl_context base;

   /* res_ids this context already handed to vkr (created or imported), so a
    * later ATTACH_RESOURCE of the same resource is a no-op instead of a
    * double import.
    */
   struct hash_table *known_res;

   /* Guest-allocated dma-buf parked between set_guest_blob_fd() and the
    * get_blob() that consumes it; same single-slot contract as the drm2kgsl
    * backend (the two calls are consecutive on the single command thread).
    */
   int pending_guest_fd;
};

/* ctx_id -> adapter, for the fence-retire callback coming from vkr's sync
 * thread.  Guarded: the table is mutated on the command thread.
 */
static struct hash_table *vkr_adapter_table;
static pthread_mutex_t vkr_adapter_table_mutex = PTHREAD_MUTEX_INITIALIZER;
static uint32_t vkr_adapter_virgl_flags;

static struct vkr_adapter_ctx *
to_adapter(struct virgl_context *vctx)
{
   return (struct vkr_adapter_ctx *)vctx;
}

static void
vkr_adapter_retire_fence(uint32_t ctx_id, uint32_t ring_idx, uint64_t fence_id)
{
   struct virgl_context *vctx = NULL;

   pthread_mutex_lock(&vkr_adapter_table_mutex);
   if (vkr_adapter_table) {
      const struct hash_entry *entry =
         _mesa_hash_table_search(vkr_adapter_table, uintptr_to_pointer(ctx_id));
      if (entry)
         vctx = entry->data;
   }
   pthread_mutex_unlock(&vkr_adapter_table_mutex);

   if (vctx && vctx->fence_retire)
      vctx->fence_retire(vctx, ring_idx, fence_id);
   else
      virgl_error("venus adapter: retire for unknown ctx %u\n", ctx_id);
}

bool
vkr_adapter_renderer_init(uint32_t virgl_flags)
{
   static const struct vkr_renderer_callbacks cbs = {
      /* keep the process-wide virgl log handler */
      .debug_logger = NULL,
      .retire_fence = vkr_adapter_retire_fence,
   };

   uint32_t vkr_flags = 0;
   if (virgl_flags & VIRGL_RENDERER_THREAD_SYNC)
      vkr_flags |= VKR_RENDERER_THREAD_SYNC;
   if (virgl_flags & VIRGL_RENDERER_ASYNC_FENCE_CB)
      vkr_flags |= VKR_RENDERER_ASYNC_FENCE_CB;

   vkr_adapter_virgl_flags = virgl_flags;

   return vkr_renderer_init(vkr_flags, &cbs);
}

void
vkr_adapter_renderer_fini(void)
{
   vkr_renderer_fini();
}

void
vkr_adapter_renderer_reset(void)
{
   /* upstream 1.3 has no vkr reset; the context table reset has already
    * destroyed every context through the adapter, so re-init is enough
    */
   vkr_renderer_fini();
   if (!vkr_adapter_renderer_init(vkr_adapter_virgl_flags))
      virgl_error("venus adapter: re-init after reset failed\n");
}

size_t
vkr_adapter_get_capset(void *capset)
{
   return vkr_get_capset(capset, vkr_adapter_virgl_flags);
}

static void
vkr_adapter_known_add(struct vkr_adapter_ctx *actx, uint32_t res_id)
{
   _mesa_hash_table_insert(actx->known_res, uintptr_to_pointer(res_id),
                           uintptr_to_pointer(res_id));
}

static bool
vkr_adapter_known_has(struct vkr_adapter_ctx *actx, uint32_t res_id)
{
   return _mesa_hash_table_search(actx->known_res, uintptr_to_pointer(res_id)) != NULL;
}

static void
vkr_adapter_destroy(struct virgl_context *vctx)
{
   struct vkr_adapter_ctx *actx = to_adapter(vctx);

   pthread_mutex_lock(&vkr_adapter_table_mutex);
   _mesa_hash_table_remove_key(vkr_adapter_table, uintptr_to_pointer(vctx->ctx_id));
   pthread_mutex_unlock(&vkr_adapter_table_mutex);

   /* destroys every resource the context still holds */
   vkr_renderer_destroy_context(vctx->ctx_id);

   if (actx->pending_guest_fd >= 0)
      close(actx->pending_guest_fd);
   _mesa_hash_table_destroy(actx->known_res, NULL);
   free(actx);
}

static void
vkr_adapter_attach_resource(struct virgl_context *vctx, struct virgl_resource *res)
{
   struct vkr_adapter_ctx *actx = to_adapter(vctx);

   if (vkr_adapter_known_has(actx, res->res_id))
      return;

   int fd = -1;
   const enum virgl_resource_fd_type fd_type = virgl_resource_export_fd(res, &fd);
   if (fd_type == VIRGL_RESOURCE_FD_INVALID) {
      /* iovec-only guest memory: unusable by venus (host process isolation);
       * loud because a venus guest is not supposed to produce these.
       */
      virgl_warn("venus adapter: ctx %u attach of res %u with no fd, ignored\n",
                 vctx->ctx_id, res->res_id);
      return;
   }

   uint64_t size = res->map_size;
   if (!size) {
      const off_t end = lseek(fd, 0, SEEK_END);
      if (end > 0)
         size = (uint64_t)end;
   }
   if (!size) {
      virgl_error("venus adapter: ctx %u attach of res %u with unknown size, ignored\n",
                  vctx->ctx_id, res->res_id);
      close(fd);
      return;
   }

   if (!vkr_renderer_import_resource(vctx->ctx_id, res->res_id, fd_type, fd, size)) {
      virgl_error("venus adapter: ctx %u attach import failed res=%u fd_type=%d size=%llu\n",
                  vctx->ctx_id, res->res_id, (int)fd_type, (unsigned long long)size);
      close(fd);
      return;
   }

   vkr_adapter_known_add(actx, res->res_id);
}

static void
vkr_adapter_detach_resource(struct virgl_context *vctx, struct virgl_resource *res)
{
   struct vkr_adapter_ctx *actx = to_adapter(vctx);

   if (!vkr_adapter_known_has(actx, res->res_id))
      return;

   vkr_renderer_destroy_resource(vctx->ctx_id, res->res_id);
   _mesa_hash_table_remove_key(actx->known_res, uintptr_to_pointer(res->res_id));
}

static int
vkr_adapter_transfer_3d(UNUSED struct virgl_context *vctx,
                        UNUSED struct virgl_resource *res,
                        UNUSED const struct vrend_transfer_info *info,
                        UNUSED int transfer_mode)
{
   return -1;
}

static int
vkr_adapter_get_blob(struct virgl_context *vctx,
                     uint32_t res_id,
                     uint64_t blob_id,
                     uint64_t blob_size,
                     uint32_t blob_flags,
                     struct virgl_context_blob *blob)
{
   struct vkr_adapter_ctx *actx = to_adapter(vctx);

   virgl_info("venus adapter: ctx %u get_blob res=%u blob_id=%llu size=%llu flags=0x%x "
              "pending_fd=%d\n",
              vctx->ctx_id, res_id, (unsigned long long)blob_id,
              (unsigned long long)blob_size, blob_flags, actx->pending_guest_fd);

   /* guest-allocated blob: the parked udmabuf IS the memory */
   if (actx->pending_guest_fd >= 0) {
      const int fd = actx->pending_guest_fd;
      actx->pending_guest_fd = -1;

      const int vkr_fd = fcntl(fd, F_DUPFD_CLOEXEC, 0);
      if (vkr_fd < 0 ||
          !vkr_renderer_import_resource(vctx->ctx_id, res_id, VIRGL_RESOURCE_FD_DMABUF,
                                        vkr_fd, blob_size)) {
         virgl_error("venus adapter: ctx %u guest blob import failed res=%u blob=%llu "
                     "vkr_fd=%d\n",
                     vctx->ctx_id, res_id, (unsigned long long)blob_id, vkr_fd);
         if (vkr_fd >= 0)
            close(vkr_fd);
         close(fd);
         return -ENOMEM;
      }
      vkr_adapter_known_add(actx, res_id);

      memset(blob, 0, sizeof(*blob));
      blob->type = VIRGL_RESOURCE_FD_DMABUF;
      blob->u.fd = fd;
      /* guest pool pages are mapped cacheable on both sides */
      blob->map_info = VIRGL_RENDERER_MAP_CACHE_CACHED;
      return 0;
   }

   enum virgl_resource_fd_type fd_type = VIRGL_RESOURCE_FD_INVALID;
   int res_fd = -1;
   uint32_t map_info = 0;
   struct virgl_resource_vulkan_info vulkan_info;
   void *map_ptr = NULL;
   uint64_t fd_offset = 0;
   memset(&vulkan_info, 0, sizeof(vulkan_info));

   if (!vkr_renderer_create_resource(vctx->ctx_id, res_id, blob_id, blob_size,
                                     blob_flags, &fd_type, &res_fd, &map_info,
                                     &vulkan_info, &map_ptr, &fd_offset)) {
      virgl_error("venus adapter: ctx %u create_resource failed res=%u blob_id=%llu\n",
                  vctx->ctx_id, res_id, (unsigned long long)blob_id);
      return -EINVAL;
   }

   vkr_adapter_known_add(actx, res_id);

   memset(blob, 0, sizeof(*blob));
   blob->type = fd_type;
   blob->u.fd = res_fd;
   blob->map_info = map_info;
   blob->vulkan_info = vulkan_info;
   /* venus_host pool residency: the VMM turns this into MAP_INFO_POOL */
   blob->map_ptr = map_ptr;
   blob->fd_offset = fd_offset;
   if (fd_type == VIRGL_RESOURCE_FD_OPAQUE) {
      /* same metadata under this fork's original name */
      memcpy(blob->opaque_fd_metadata.driver_uuid, vulkan_info.driver_uuid, 16);
      memcpy(blob->opaque_fd_metadata.device_uuid, vulkan_info.device_uuid, 16);
      blob->opaque_fd_metadata.allocation_size = vulkan_info.allocation_size;
      blob->opaque_fd_metadata.memory_type_index = vulkan_info.memory_type_index;
   }
   return 0;
}

static int
vkr_adapter_set_guest_blob_fd(struct virgl_context *vctx, uint64_t blob_id, int fd)
{
   struct vkr_adapter_ctx *actx = to_adapter(vctx);

   if (actx->pending_guest_fd >= 0) {
      /* two sets without an intervening get_blob: the pairing assumption is
       * wrong, report instead of silently leaking (same contract as drm2kgsl)
       */
      virgl_error("venus adapter: guest blob fd already pending (blob_id=%" PRIu64
                  "); dropping the stale one\n",
                  blob_id);
      close(actx->pending_guest_fd);
   }
   actx->pending_guest_fd = fd;
   virgl_info("venus adapter: ctx %u parked guest fd=%d blob_id=%llu\n", vctx->ctx_id, fd,
              (unsigned long long)blob_id);
   return 0;
}

static int
vkr_adapter_submit_cmd(struct virgl_context *vctx, const void *buffer, size_t size)
{
   return vkr_renderer_submit_cmd(vctx->ctx_id, (void *)buffer, (uint32_t)size) ? 0
                                                                                : -EINVAL;
}

static int
vkr_adapter_get_fencing_fd(UNUSED struct virgl_context *vctx)
{
   /* async fence callbacks only */
   return -1;
}

static void
vkr_adapter_retire_fences(UNUSED struct virgl_context *vctx)
{
   /* async fence callbacks only */
}

static int
vkr_adapter_submit_fence(struct virgl_context *vctx,
                         uint32_t flags,
                         uint32_t ring_idx,
                         uint64_t fence_id)
{
   return vkr_renderer_submit_fence(vctx->ctx_id, flags, ring_idx, fence_id) ? 0
                                                                             : -EINVAL;
}

struct virgl_context *
vkr_adapter_context_create(uint32_t ctx_id, size_t nlen, const char *name)
{
   struct vkr_adapter_ctx *actx = calloc(1, sizeof(*actx));
   if (!actx)
      return NULL;

   actx->known_res = _mesa_hash_table_create_u32_keys(NULL);
   if (!actx->known_res) {
      free(actx);
      return NULL;
   }
   actx->pending_guest_fd = -1;

   if (!vkr_renderer_create_context(ctx_id, VIRGL_RENDERER_CAPSET_VENUS, (uint32_t)nlen,
                                    name)) {
      _mesa_hash_table_destroy(actx->known_res, NULL);
      free(actx);
      return NULL;
   }

   struct virgl_context *vctx = &actx->base;
   vctx->ctx_id = ctx_id;
   vctx->capset_id = VIRGL_RENDERER_CAPSET_VENUS;
   vctx->destroy = vkr_adapter_destroy;
   vctx->attach_resource = vkr_adapter_attach_resource;
   vctx->detach_resource = vkr_adapter_detach_resource;
   vctx->export_opaque_handle = NULL;
   vctx->transfer_3d = vkr_adapter_transfer_3d;
   vctx->get_blob = vkr_adapter_get_blob;
   vctx->set_guest_blob_fd = vkr_adapter_set_guest_blob_fd;
   vctx->submit_cmd = vkr_adapter_submit_cmd;
   vctx->get_fencing_fd = vkr_adapter_get_fencing_fd;
   vctx->retire_fences = vkr_adapter_retire_fences;
   vctx->submit_fence = vkr_adapter_submit_fence;

   pthread_mutex_lock(&vkr_adapter_table_mutex);
   if (!vkr_adapter_table)
      vkr_adapter_table = _mesa_hash_table_create_u32_keys(NULL);
   if (vkr_adapter_table)
      _mesa_hash_table_insert(vkr_adapter_table, uintptr_to_pointer(ctx_id), vctx);
   pthread_mutex_unlock(&vkr_adapter_table_mutex);

   return vctx;
}
