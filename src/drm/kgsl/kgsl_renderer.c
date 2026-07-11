/*
 * Copyright 2022 Google LLC
 * Copyright 2026 DroidVM
 * SPDX-License-Identifier: MIT
 *
 * KGSL native-context backend.
 *
 * This is the KGSL analogue of msm_renderer.c.  The guest runs stock
 * drm/msm (freedreno/turnip) userspace over vdrm/virtio and therefore speaks
 * the *msm* wire protocol (msm_proto.h) and registers under
 * VIRTGPU_DRM_CONTEXT_MSM.  The host GPU on this device is a Qualcomm KGSL
 * character device (/dev/kgsl-3d0), not a DRM/msm device.  So this backend
 * implements the msm protocol server on top of KGSL ioctls.
 *
 * Three impedance mismatches between the msm protocol and KGSL are bridged
 * here:
 *
 *  1) Guest-assigned iova.  The msm protocol has the guest pick the GPU VA of
 *     every BO (util_vma_heap over [va_start, va_start+va_size)) and send it
 *     in GEM_NEW/GEM_SET_IOVA; the host must place the BO at exactly that VA.
 *     KGSL's GPUMEM_ALLOC_ID / GPUOBJ_IMPORT assign their *own* gpuaddr and
 *     provide no way to pin an arbitrary one.  We bridge this with a single
 *     giant Virtual Buffer Object (VBO) per context that reserves the whole
 *     reported VA range; each BO is bound into the VBO at
 *     target_offset = guest_iova - vbo_base via GPUMEM_BIND_RANGES.  Binding
 *     adds a *second* mapping at the guest iova -- the child keeps its own
 *     unused KGSL-assigned gpuaddr, so there is no conflict.
 *
 *  2) Exportable memory.  KGSL can only *import* dmabufs, never export them,
 *     but the guest needs the BO as a HOST3D blob (dmabuf) it can map and that
 *     the display pipeline can scan out.  So the backing is allocated from a
 *     dma-heap (exportable dmabuf) and then GPUOBJ_IMPORT'ed into KGSL.  The
 *     same dmabuf fd is what get_blob() hands back -- which flows through the
 *     existing share66 GuestAccept path exactly like gfxstream today.
 *
 *  3) Fence seqno.  The msm protocol has the guest assign the submit seqno
 *     (MSM_SUBMIT_FENCE_SN_IN).  KGSL normally assigns its own timestamp, but
 *     KGSL_CONTEXT_USER_GENERATED_TS lets us pass the guest seqno straight
 *     through as the KGSL timestamp, so the two timelines stay identical.  A
 *     sync_file fd for the vdrm ring fence is obtained from the completed
 *     submit via TIMESTAMP_EVENT(FENCE) and fed to drm_timeline.
 */

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/types.h>

#include <linux/dma-heap.h>
#include <linux/memfd.h>
#include <linux/udmabuf.h>

/* The vendored KGSL/msm uapi headers carry the kernel's __user annotation,
 * which is not defined in a pure userspace build; neutralise it.
 */
#ifndef __user
#define __user
#endif

#include "virgl_context.h"
#include "virgl_util.h"
#include "virglrenderer.h"

#include "util/hash_table.h"
#include "util/list.h"
#include "util/macros.h"
#include "util/os_file.h"
#include "util/u_atomic.h"

#include "drm_context.h"
#include "drm_fence.h"
#include "drm_hw.h"
#include "drm_util.h"

#include "drm-uapi/msm_drm.h" /* msm wire structs: drm_msm_param, submit_bo/cmd */
#include "msm/msm_proto.h"
#include "kgsl/msm_kgsl.h"
#include "kgsl/kgsl_renderer.h"

/*
 * The VBO reserves the guest's entire GPU VA range.  It is virtual-only
 * (no backing until children are bound), so a large reservation is cheap.
 * We probe the largest size the kernel accepts, walking down this ladder, and
 * report the achieved size as va_size in the capset.
 */
static const uint64_t vbo_size_ladder[] = {
   0x10000000000ull, /* 1 TiB   */
   0x04000000000ull, /* 256 GiB */
   0x01000000000ull, /* 64 GiB  */
   0x00400000000ull, /* 16 GiB  */
   0x00100000000ull, /* 4 GiB   */
};

/* Filled at probe, consumed at context create.  Global because the capset is
 * global; every context must reproduce the same VA layout.
 */
static uint64_t g_va_start;
static uint64_t g_va_size;
static uint64_t g_chip_id;
static uint32_t g_gpu_id;
static uint32_t g_gmem_size;
static uint64_t g_gmem_base;
static uint32_t g_highest_bank_bit;
static uint64_t g_uche_trap_base;
static unsigned g_nr_timelines;

/* The single VBO shared by every context (see kgsl_renderer_create).  Allocated
 * once at probe and kept alive for the renderer's lifetime by g_keeper_fd (the
 * probe fd is closed by drm_renderer_init; KGSL memory is freed when the last fd
 * of the process closes, so we hold one open).
 */
static uint32_t g_vbo_id;
static int g_keeper_fd = -1;

/* Per-context VA slicing: KGSL pagetables are per-process, so every guest
 * process's turnip shares one host VBO.  All guests allocate iovas from the
 * same capset va_start, so concurrent clients (gnome-shell + Xwayland +
 * greeter + ...) clobber each other's VBO bindings — the deterministic
 * CP opcode-0 faults and smashed desktop geometry.  Each context therefore
 * gets a disjoint slice of the VBO, handed to the guest via GET_PARAM
 * (MSM_PARAM_VA_START/VA_SIZE), which the patched guest prefers over the
 * (global) capset values. */
#define KGSL_VA_SLICE_SIZE (8ull << 30)
static uint32_t g_va_slices_used; /* bitmask; bit i = slice i in use */

static const char dma_heap_path[] = "/dev/dma_heap/system";

/*
 * KGSL object: extends drm_object with KGSL-specific bookkeeping.
 *
 *  - mem_id:     KGSL GPU memory id (from GPUOBJ_IMPORT), used to free and to
 *                identify the child in BIND_RANGES.
 *  - dmabuf_fd:  the exportable backing fd (from dma-heap), handed to get_blob
 *                and mmap'd for GEM_UPLOAD.
 *  - iova:       guest-assigned GPU VA where the BO is bound in the VBO
 *                (0 == not currently bound).
 *  - blob_size:  page-aligned size, i.e. the VA span reserved in the VBO.
 */
struct kgsl_object {
   struct drm_object base;
   uint32_t mem_id;
   int dmabuf_fd;      /* the blob-backing fd: a THP memfd for our BOs, or an
                        * imported dmabuf for external (attach_resource) BOs */
   void *host_map;     /* our own mapping of a memfd BO (kept for the KGSL
                        * useraddr import lifetime; NULL for imported dmabufs) */
   uint64_t host_map_size;
   uint64_t iova;
   uint64_t blob_size;
   uint32_t flags;
   bool exported   : 1;
   bool exportable : 1;
   bool is_shm     : 1; /* backing is a memfd (SHM blob) vs an imported dmabuf */
   /* Paged arena BO (mem_id == 0): scattered arena backing, stitched to the
    * contiguous iova via multi-entry BIND_RANGES; map is a MAP_FIXED stitch of
    * the arena memfd (owned, munmap on free). */
   struct msm_gem_new_run *runs;
   uint32_t nr_runs;
   uint8_t *map;
};
DEFINE_CAST(drm_object, kgsl_object)

struct kgsl_context {
   struct drm_context base;

   struct msm_shmem *shmem;

   /* The per-context VBO that owns the whole guest VA range. */
   uint32_t vbo_id;
   uint64_t vbo_base;

   /* dma-heap fd for exportable backing allocations. */
   int dma_heap_fd;

   /* This context's disjoint VA slice (see g_va_slices_used). */
   uint64_t va_slice_base;
   uint64_t va_slice_size;
   int va_slice_idx;

   /* The blessed arena object (gem_new covering the whole reported VA range):
    * one backing + one bind + one guest map for the context's lifetime; every
    * non-SCANOUT BO afterwards is bookkeeping aliasing into it.  NULL until
    * the guest blesses (legacy guests never do — full per-BO path remains). */
   struct kgsl_object *arena;

   /* Maps submitqueue-id (== KGSL drawctxt id) to ring_idx. */
   struct hash_table *sq_to_ring_idx_table;

   struct drm_timeline timelines[];
};
DEFINE_CAST(drm_context, kgsl_context)

/*
 * ---------------------------------------------------------------------------
 * Low-level KGSL helpers
 * ---------------------------------------------------------------------------
 */

static int
kgsl_ioctl(int fd, unsigned long request, void *arg)
{
   int ret;
   do {
      ret = ioctl(fd, request, arg);
   } while (ret == -1 && (errno == EINTR || errno == EAGAIN));
   return ret;
}

static int
kgsl_getprop(int fd, unsigned int type, void *value, size_t size)
{
   struct kgsl_device_getproperty req = {
      .type = type,
      .value = value,
      .sizebytes = size,
   };
   return kgsl_ioctl(fd, IOCTL_KGSL_DEVICE_GETPROPERTY, &req);
}

/* Allocate an exportable dmabuf from the system dma-heap. */
static int
kgsl_dma_heap_alloc(int heap_fd, uint64_t size)
{
   struct dma_heap_allocation_data alloc = {
      .len = size,
      .fd_flags = O_RDWR | O_CLOEXEC,
   };

   if (kgsl_ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc)) {
      drm_err("DMA_HEAP_IOCTL_ALLOC(%" PRIu64 ") failed: %s", size, strerror(errno));
      return -1;
   }

   return alloc.fd;
}

/*
 * Allocate BO backing that crosvm's GuestAccept share can pin.  The share does
 * pin_user_pages_fast(FOLL_LONGTERM), which fails on ZONE_MOVABLE pages; this
 * device has no unmovable zone to migrate to.  A shmem THP (2 MiB) faulted here
 * is served by the gh_hugepage_reserve pool -> non-movable -> pinnable.  We back
 * the BO with a memfd (so it has an fd for the blob), fault it as a huge page,
 * and import it into KGSL by user address.  Returns the KGSL memory id (0 on
 * failure) and, on success, *out_memfd / *out_map / *out_map_size.
 */
/* Import a dmabuf into KGSL, returning its GPU memory id (0 on failure). */
static uint32_t
kgsl_import_dmabuf(int fd, int dmabuf_fd, uint32_t kgsl_flags)
{
   struct kgsl_gpuobj_import_dma_buf priv = {
      .fd = dmabuf_fd,
   };
   struct kgsl_gpuobj_import req = {
      .priv = (uintptr_t)&priv,
      .priv_len = sizeof(priv),
      .flags = kgsl_flags,
      .type = KGSL_USER_MEM_TYPE_DMABUF,
   };

   if (kgsl_ioctl(fd, IOCTL_KGSL_GPUOBJ_IMPORT, &req)) {
      drm_err("GPUOBJ_IMPORT failed: %s", strerror(errno));
      return 0;
   }

   return req.id;
}

#ifndef MADV_HUGEPAGE
#define MADV_HUGEPAGE 14
#endif
#ifndef MADV_COLLAPSE
#define MADV_COLLAPSE 25
#endif
/*
 * Allocate BO backing store that crosvm's GuestAccept share can actually pin.
 *
 * dma-heap / plain shmem memory is ZONE_MOVABLE; crosvm shares blobs with
 * pin_user_pages_fast(FOLL_LONGTERM), which refuses movable pages (EFAULT).
 * Instead we collapse a sealed memfd into a single order-9 (2MB) folio the same
 * way gfxstream's RingBlob pool does: fault it through a PMD-aligned mapping and
 * MADV_COLLAPSE.  The in-process collapse allocation is what gh_hugepage_reserve
 * intercepts, so the folio comes from the module's non-movable reserve pool
 * (plain MADV_HUGEPAGE + fault is NOT intercepted for shmem and stays movable).
 * The memfd itself becomes the guest-shared SHM blob; a transient udmabuf view
 * of it is imported into KGSL for GPU access (see below).
 *
 * Returns the KGSL memory id (0 on failure); *out_dmabuf gets the memfd.
 */
static uint32_t
kgsl_alloc_thp_bo(int fd, uint64_t size, uint32_t kgsl_flags, int *out_dmabuf)
{
   uint64_t map_size = ALIGN_POT(size, 0x200000); /* 2 MiB for the THP path */

   int memfd = syscall(__NR_memfd_create, "kgsl-bo",
                       MFD_CLOEXEC | MFD_ALLOW_SEALING);
   if (memfd < 0) {
      drm_err("memfd_create failed: %s", strerror(errno));
      return 0;
   }
   if (ftruncate(memfd, map_size) ||
       fcntl(memfd, F_ADD_SEALS, F_SEAL_SHRINK)) {  /* udmabuf requires it */
      drm_err("ftruncate/seal(%" PRIu64 ") failed: %s", map_size, strerror(errno));
      close(memfd);
      return 0;
   }

   /* Collapse each 2MB chunk into a reserve-pool order-9 folio before the
    * udmabuf GUP-pins it.  Reserve map_size + 2MB, carve out a PMD-aligned
    * window, map the memfd there, MADV_HUGEPAGE + fault + MADV_COLLAPSE.  The
    * folio persists in the shmem page cache after the temp mapping is torn down.
    */
   void *rsv = mmap(NULL, map_size + 0x200000, PROT_NONE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
   if (rsv == MAP_FAILED) {
      drm_err("reserve mmap failed: %s", strerror(errno));
      close(memfd);
      return 0;
   }
   uintptr_t aligned_addr =
      ((uintptr_t)rsv + 0x200000 - 1) & ~(uintptr_t)(0x200000 - 1);
   void *aligned = mmap((void *)aligned_addr, map_size, PROT_READ | PROT_WRITE,
                        MAP_SHARED | MAP_FIXED, memfd, 0);
   if (aligned != MAP_FAILED) {
      madvise(aligned, map_size, MADV_HUGEPAGE);
      memset(aligned, 0, map_size);
      madvise(aligned, map_size, MADV_COLLAPSE);
   }
   munmap(rsv, map_size + 0x200000);

   /* Import a *transient* udmabuf view of the collapsed memfd into KGSL for GPU
    * access.  Split rationale: KGSL needs a dmabuf (useraddr import EINVALs
    * here), but crosvm must GuestAccept-share the *memfd* (SHM) not the udmabuf
    * — udmabuf VMAs are VM_PFNMAP, so crosvm's pin_user_pages_fast(FOLL_LONGTERM)
    * EFAULTs on them; a shmem memfd mapping pins fine.  Both fds reference the
    * same non-movable reserve folio, so the BO stays zero-copy.  KGSL's dma_buf
    * attachment holds a ref to the udmabuf, so its fd can close after import.
    */
   int udev = open("/dev/udmabuf", O_RDWR | O_CLOEXEC);
   if (udev < 0) {
      drm_err("open /dev/udmabuf failed: %s", strerror(errno));
      close(memfd);
      return 0;
   }
   struct udmabuf_create uc = {
      .memfd = memfd,
      .flags = UDMABUF_FLAGS_CLOEXEC,
      .offset = 0,
      .size = map_size,
   };
   int udmabuf = ioctl(udev, UDMABUF_CREATE, &uc);
   close(udev);
   if (udmabuf < 0) {
      drm_err("UDMABUF_CREATE failed: %s", strerror(errno));
      close(memfd);
      return 0;
   }

   uint32_t mem_id = kgsl_import_dmabuf(fd, udmabuf, kgsl_flags);
   close(udmabuf);  /* KGSL's dma_buf attachment keeps the pages */
   if (!mem_id) {
      close(memfd);
      return 0;
   }

   *out_dmabuf = memfd;  /* SHM handle crosvm hands to the guest */
   return mem_id;
}

static void
kgsl_free_gpuobj(int fd, uint32_t mem_id)
{
   struct kgsl_gpuobj_free req = { .id = mem_id };
   kgsl_ioctl(fd, IOCTL_KGSL_GPUOBJ_FREE, &req);
}

/*
 * Bind (op == KGSL_GPUMEM_RANGE_OP_BIND) or unbind a child object into this
 * context's VBO at the guest-chosen iova.  This is what makes the BO visible
 * to the GPU at exactly the address the guest baked into its cmdstreams.
 */
static int
kgsl_vbo_op(struct kgsl_context *kctx, uint32_t child_id, uint64_t iova,
            uint64_t size, uint32_t op)
{
   if (iova < kctx->vbo_base || iova + size > kctx->vbo_base + g_va_size) {
      drm_err("iova 0x%" PRIx64 " (size 0x%" PRIx64 ") outside VBO [0x%" PRIx64
              ", 0x%" PRIx64 ")", iova, size, kctx->vbo_base,
              kctx->vbo_base + g_va_size);
      return -EINVAL;
   }

   struct kgsl_gpumem_bind_range range = {
      .child_offset = 0,
      .target_offset = iova - kctx->vbo_base,
      .length = size,
      .child_id = child_id,
      .op = op,
   };
   struct kgsl_gpumem_bind_ranges req = {
      .ranges = (uintptr_t)&range,
      .ranges_nents = 1,
      .ranges_size = sizeof(range),
      .id = kctx->vbo_id,
      .flags = 0, /* synchronous */
      .fence_id = 0,
   };

   if (kgsl_ioctl(kctx->base.fd, IOCTL_KGSL_GPUMEM_BIND_RANGES, &req)) {
      drm_err("BIND_RANGES(op=%u, child=%u, off=0x%" PRIx64 ") failed: %s",
              op, child_id, iova - kctx->vbo_base, strerror(errno));
      return -errno;
   }

   return 0;
}

static inline int
kgsl_vbo_bind(struct kgsl_context *kctx, struct kgsl_object *obj, uint64_t iova)
{
   int ret = kgsl_vbo_op(kctx, obj->mem_id, iova, obj->blob_size,
                         KGSL_GPUMEM_RANGE_OP_BIND);
   if (!ret)
      obj->iova = iova;
   return ret;
}

static inline void
kgsl_vbo_unbind(UNUSED struct kgsl_context *kctx, UNUSED struct kgsl_object *obj)
{
   if (!obj->iova)
      return;
   kgsl_vbo_op(kctx, obj->mem_id, obj->iova, obj->blob_size,
               KGSL_GPUMEM_RANGE_OP_UNBIND);
   obj->iova = 0;
}

/* Stitch a paged-arena BO's scattered runs to its contiguous iova with a
 * single multi-entry BIND_RANGES against the arena child. */
static int
kgsl_vbo_bind_runs(struct kgsl_context *kctx, struct kgsl_object *obj,
                   uint64_t iova)
{
   if (iova < kctx->vbo_base ||
       iova + obj->blob_size > kctx->vbo_base + g_va_size) {
      drm_err("iova 0x%" PRIx64 " (size 0x%" PRIx64 ") outside VBO", iova,
              obj->blob_size);
      return -EINVAL;
   }

   struct kgsl_gpumem_bind_range *ranges =
      calloc(obj->nr_runs, sizeof(*ranges));
   if (!ranges)
      return -ENOMEM;

   uint64_t off = 0;
   for (uint32_t i = 0; i < obj->nr_runs; i++) {
      ranges[i] = (struct kgsl_gpumem_bind_range) {
         .child_offset = obj->runs[i].arena_off,
         .target_offset = (iova + off) - kctx->vbo_base,
         .length = obj->runs[i].len,
         .child_id = kctx->arena->mem_id,
         .op = KGSL_GPUMEM_RANGE_OP_BIND,
      };
      off += obj->runs[i].len;
   }

   struct kgsl_gpumem_bind_ranges req = {
      .ranges = (uintptr_t)ranges,
      .ranges_nents = obj->nr_runs,
      .ranges_size = sizeof(*ranges),
      .id = kctx->vbo_id,
      .flags = 0, /* synchronous */
   };

   int ret = kgsl_ioctl(kctx->base.fd, IOCTL_KGSL_GPUMEM_BIND_RANGES, &req);
   free(ranges);
   if (ret) {
      drm_err("BIND_RANGES(%u runs @ 0x%" PRIx64 ") failed: %s", obj->nr_runs,
              iova, strerror(errno));
      return -errno;
   }
   return 0;
}

/*
 * ---------------------------------------------------------------------------
 * Object lifecycle
 * ---------------------------------------------------------------------------
 */

static struct kgsl_object *
kgsl_object_create(uint32_t mem_id, int dmabuf_fd, uint32_t flags,
                   uint64_t size, uint64_t blob_size)
{
   struct kgsl_object *obj = calloc(1, sizeof(*obj));
   if (!obj)
      return NULL;

   obj->base.handle = mem_id; /* mirror for symmetry with msm's gem handle */
   obj->base.size = size;
   obj->mem_id = mem_id;
   obj->dmabuf_fd = dmabuf_fd;
   obj->flags = flags;
   obj->blob_size = blob_size;

   return obj;
}

static struct kgsl_object *
kgsl_object_from_res_id(struct kgsl_context *kctx, uint32_t res_id)
{
   struct drm_object *dobj = drm_context_get_object_from_res_id(&kctx->base, res_id);
   return dobj ? to_kgsl_object(dobj) : NULL;
}

static struct kgsl_object *
kgsl_object_from_blob_id(struct kgsl_context *kctx, uint64_t blob_id)
{
   struct drm_object *dobj =
      drm_context_retrieve_object_from_blob_id(&kctx->base, blob_id);
   return dobj ? to_kgsl_object(dobj) : NULL;
}

static void
kgsl_renderer_free_object(struct drm_context *dctx, struct drm_object *dobj)
{
   struct kgsl_context *kctx = to_kgsl_context(dctx);
   struct kgsl_object *obj = to_kgsl_object(dobj);

   /* Note: no explicit VBO unbind here (see kgsl_ccmd_gem_set_iova) — a bind
    * holds its own reference on the child, so pages stay mapped for any
    * in-flight submit until the guest reuses the VA. */

   /* Paged arena BO: bookkeeping + the stitched host view only. */
   if (!obj->mem_id) {
      if (obj->map)
         munmap(obj->map, obj->blob_size);
      free(obj->runs);
      free(obj);
      return;
   }

   if (obj == kctx->arena)
      kctx->arena = NULL;

   /* obj->map aliases host_map for our memfd BOs -- only free a distinct one. */
   if (obj->map && obj->map != obj->host_map)
      munmap(obj->map, obj->blob_size);

   kgsl_free_gpuobj(dctx->fd, obj->mem_id); /* releases the useraddr pin first */

   if (obj->host_map)
      munmap(obj->host_map, obj->host_map_size);
   if (obj->dmabuf_fd >= 0)
      close(obj->dmabuf_fd);

   free(obj);
}

/*
 * ---------------------------------------------------------------------------
 * msm ccmd handlers -> KGSL
 * ---------------------------------------------------------------------------
 */

static void *
kgsl_ctx_rsp(struct kgsl_context *kctx, const struct vdrm_ccmd_req *hdr, size_t len)
{
   return drm_context_rsp(&kctx->base, hdr, len);
}

static int
kgsl_ccmd_nop(UNUSED struct drm_context *dctx, UNUSED struct vdrm_ccmd_req *hdr)
{
   return 0;
}

/* Map an MSM_PARAM_* to the value the guest expects, from cached KGSL props. */
static uint64_t
kgsl_msm_param(struct kgsl_context *kctx, uint32_t param)
{
   switch (param) {
   case MSM_PARAM_GPU_ID:      return g_gpu_id;
   case MSM_PARAM_CHIP_ID:     return g_chip_id;
   case MSM_PARAM_GMEM_SIZE:   return g_gmem_size;
   case MSM_PARAM_GMEM_BASE:   return g_gmem_base;
   case MSM_PARAM_VA_START:    return kctx->va_slice_base;
   case MSM_PARAM_VA_SIZE:     return kctx->va_slice_size;
   case MSM_PARAM_PRIORITIES:  return g_nr_timelines; /* == MSM_PARAM_NR_RINGS */
   case MSM_PARAM_HIGHEST_BANK_BIT: return g_highest_bank_bit;
   case MSM_PARAM_UCHE_TRAP_BASE:   return g_uche_trap_base;
   case MSM_PARAM_TIMESTAMP: {
      /* Best-effort GPU timestamp; not all guests query this. */
      return 0;
   }
   case MSM_PARAM_FAULTS:      return 0;
   default:
      drm_dbg("unhandled MSM_PARAM %u", param);
      return 0;
   }
}

static int
kgsl_drawctxt_create(struct kgsl_context *kctx, UNUSED uint32_t prio, uint32_t *out_id)
{
   /* Mirror the flags turnip's own KGSL backend uses (proven on this device);
    * the guest submits its own fence seqno (MSM_SUBMIT_FENCE_SN_IN), so add
    * USER_GENERATED_TS so KGSL takes the timestamp from us.  Adding TYPE_VK /
    * PER_CONTEXT_TS / a priority band made DRAWCTXT_CREATE return EINVAL.
    */
   struct kgsl_drawctxt_create req = {
      .flags = KGSL_CONTEXT_SAVE_GMEM |
               KGSL_CONTEXT_NO_GMEM_ALLOC |
               KGSL_CONTEXT_PREAMBLE |
               KGSL_CONTEXT_USER_GENERATED_TS,
   };

   if (kgsl_ioctl(kctx->base.fd, IOCTL_KGSL_DRAWCTXT_CREATE, &req)) {
      drm_err("DRAWCTXT_CREATE failed: %s", strerror(errno));
      return -errno;
   }

   *out_id = req.drawctxt_id;
   return 0;
}

static void
kgsl_drawctxt_destroy(struct kgsl_context *kctx, uint32_t drawctxt_id)
{
   struct kgsl_drawctxt_destroy req = { .drawctxt_id = drawctxt_id };
   kgsl_ioctl(kctx->base.fd, IOCTL_KGSL_DRAWCTXT_DESTROY, &req);
}

/*
 * IOCTL_SIMPLE forwards a small allow-list of drm/msm ioctls.  On a real msm
 * host these hit the kernel directly; here we emulate them against KGSL.
 */
static int
kgsl_ccmd_ioctl_simple(struct drm_context *dctx, struct vdrm_ccmd_req *hdr)
{
   const struct msm_ccmd_ioctl_simple_req *req = to_msm_ccmd_ioctl_simple_req(hdr);
   struct kgsl_context *kctx = to_kgsl_context(dctx);
   unsigned payload_len = _IOC_SIZE(req->cmd);
   size_t req_len = size_add(sizeof(*req), payload_len);

   if (hdr->len != req_len) {
      drm_err("%u != %zu", hdr->len, req_len);
      return -EINVAL;
   }
   if (payload_len > 128) {
      drm_err("invalid ioctl payload length: %u", payload_len);
      return -EINVAL;
   }

   unsigned iocnr = _IOC_NR(req->cmd) - DRM_COMMAND_BASE;

   struct msm_ccmd_ioctl_simple_rsp *rsp;
   size_t rsp_len = sizeof(*rsp);
   if (req->cmd & IOC_OUT)
      rsp_len = size_add(rsp_len, payload_len);

   rsp = kgsl_ctx_rsp(kctx, hdr, rsp_len);
   if (!rsp)
      return -ENOMEM;

   /* Copy payload so we can write outputs back for IOC_OUT ioctls. */
   char payload[payload_len];
   memcpy(payload, req->payload, payload_len);

   switch (iocnr) {
   case DRM_MSM_GET_PARAM: {
      struct drm_msm_param *p = (void *)payload;
      p->value = kgsl_msm_param(kctx, p->param);
      rsp->ret = 0;
      break;
   }
   case DRM_MSM_SET_PARAM:
      /* comm/cmdline debug params -- accept and ignore. */
      rsp->ret = 0;
      break;
   case DRM_MSM_SUBMITQUEUE_NEW: {
      struct drm_msm_submitqueue *q = (void *)payload;
      uint32_t id;
      rsp->ret = kgsl_drawctxt_create(kctx, q->prio, &id);
      if (!rsp->ret) {
         q->id = id;
         unsigned ring_idx = MIN2(q->prio, g_nr_timelines - 1) + 1;
         _mesa_hash_table_insert(kctx->sq_to_ring_idx_table,
                                 (void *)(uintptr_t)id,
                                 (void *)(uintptr_t)ring_idx);
      }
      break;
   }
   case DRM_MSM_SUBMITQUEUE_CLOSE: {
      uint32_t id = *(uint32_t *)payload;
      kgsl_drawctxt_destroy(kctx, id);
      _mesa_hash_table_remove_key(kctx->sq_to_ring_idx_table,
                                  (void *)(uintptr_t)id);
      rsp->ret = 0;
      break;
   }
   default:
      drm_err("unsupported IOCTL_SIMPLE: %08x (nr %u)", req->cmd, iocnr);
      rsp->ret = -EINVAL;
      break;
   }

   if (req->cmd & IOC_OUT)
      memcpy(rsp->payload, payload, payload_len);

   return 0;
}

/*
 * GEM_NEW: allocate exportable backing from the dma-heap, import into KGSL,
 * and bind into the VBO at the guest-chosen iova.
 */
static int
kgsl_ccmd_gem_new(struct drm_context *dctx, struct vdrm_ccmd_req *hdr)
{
   const struct msm_ccmd_gem_new_req *req = to_msm_ccmd_gem_new_req(hdr);
   struct kgsl_context *kctx = to_kgsl_context(dctx);
   uint64_t blob_size = ALIGN_POT(req->size, getpagesize());
   int ret = 0;

   if (!drm_context_blob_id_valid(dctx, req->blob_id)) {
      drm_err("Invalid blob_id %u", req->blob_id);
      ret = -EINVAL;
      goto out_error;
   }

   /* Arena v2 (see ARENA_V2_PLAN.md and msm_proto.h):
    *  - iova == 0                  -> arena blessing (allocate the page pool)
    *  - trailing run list present  -> paged arena BO (stitch runs, bookkeeping)
    *  - otherwise                  -> full legacy per-BO path
    */
   const bool is_arena = req->iova == 0;

   uint32_t nr_runs = 0;
   const struct msm_gem_new_run *runs = NULL;
   if (hdr->len > sizeof(*req)) {
      if (hdr->len < sizeof(*req) + 2 * sizeof(uint32_t)) {
         drm_err("malformed gem_new run list");
         ret = -EINVAL;
         goto out_error;
      }
      const uint32_t *tail = (const uint32_t *)((const uint8_t *)req + sizeof(*req));
      nr_runs = tail[0];
      runs = (const struct msm_gem_new_run *)(tail + 2);
      if (hdr->len <
          sizeof(*req) + 2 * sizeof(uint32_t) + (uint64_t)nr_runs * sizeof(*runs)) {
         drm_err("gem_new run list overflows ccmd (nr_runs=%u)", nr_runs);
         ret = -EINVAL;
         goto out_error;
      }
   }

   /* Paged arena BO: validate the runs, stitch them to the contiguous guest
    * iova with one multi-entry BIND_RANGES against the arena child, and keep
    * only bookkeeping.  No memfd, no import, no GuestAccept share.
    */
   if (nr_runs && !is_arena) {
      if (!kctx->arena) {
         drm_err("run-list gem_new before arena blessing");
         ret = -EINVAL;
         goto out_error;
      }
      uint64_t total = 0;
      for (uint32_t i = 0; i < nr_runs; i++) {
         if (!runs[i].len || (runs[i].arena_off | runs[i].len) & (getpagesize() - 1) ||
             runs[i].arena_off + runs[i].len > kctx->arena->blob_size) {
            drm_err("bad arena run %u: off=0x%" PRIx64 " len=0x%" PRIx64,
                    i, runs[i].arena_off, runs[i].len);
            ret = -EINVAL;
            goto out_error;
         }
         total += runs[i].len;
      }
      if (total < blob_size) {
         drm_err("runs cover 0x%" PRIx64 " < blob 0x%" PRIx64, total, blob_size);
         ret = -EINVAL;
         goto out_error;
      }

      struct kgsl_object *obj =
         kgsl_object_create(0 /* no own kgsl object */, -1, req->flags,
                            req->size, blob_size);
      if (!obj) {
         ret = -ENOMEM;
         goto out_error;
      }
      obj->runs = malloc(nr_runs * sizeof(*obj->runs));
      if (!obj->runs) {
         free(obj);
         ret = -ENOMEM;
         goto out_error;
      }
      memcpy(obj->runs, runs, nr_runs * sizeof(*obj->runs));
      obj->nr_runs = nr_runs;
      obj->is_shm = true;

      ret = kgsl_vbo_bind_runs(kctx, obj, req->iova);
      if (ret) {
         free(obj->runs);
         free(obj);
         goto out_error;
      }
      obj->iova = req->iova;
      drm_context_object_set_blob_id(dctx, &obj->base, req->blob_id);
      return 0;
   }

   /* Map msm cache flags to KGSL import flags.
    *
    * Every BO is imported IO-coherent regardless of the guest's cache mode:
    * the guest CPU writes BOs through its own stage-1 mapping (often cached)
    * and nobody on this path issues KGSL cache-maintenance ops (real turnip's
    * tu_knl_kgsl does its own GPUOBJ_SYNC; the vdrm guest cannot).  A
    * non-snooping GPU read then sees stale DRAM — deterministic smashed
    * geometry on the desktop.  Adreno 830 supports full IO coherency, so let
    * the GPU snoop the (shared) CPU caches instead.
    */
   uint32_t kgsl_flags = KGSL_MEMFLAGS_IOCOHERENT;
   if (req->flags & (MSM_BO_CACHED_COHERENT | MSM_BO_CACHED))
      kgsl_flags |= KGSL_CACHEMODE_WRITEBACK << KGSL_CACHEMODE_SHIFT;
   else
      kgsl_flags |= KGSL_CACHEMODE_WRITECOMBINE << KGSL_CACHEMODE_SHIFT;

   /* Back the BO with a reserve-pool collapsed memfd (see kgsl_alloc_thp_bo):
    * its pages are non-movable, so crosvm's FOLL_LONGTERM GuestAccept share can
    * pin them.  The memfd is shared to the guest as SHM; KGSL gets the same
    * pages via a transient udmabuf import.  dma-heap / plain shmem memory is
    * movable and fails that pin (EFAULT).
    */
   int dmabuf_fd = -1;  /* actually the memfd backing the SHM blob */
   uint32_t mem_id = kgsl_alloc_thp_bo(dctx->fd, blob_size, kgsl_flags, &dmabuf_fd);
   if (!mem_id) {
      ret = -ENOMEM;
      goto out_error;
   }

   struct kgsl_object *obj =
      kgsl_object_create(mem_id, dmabuf_fd, req->flags, req->size, blob_size);
   if (!obj) {
      kgsl_free_gpuobj(dctx->fd, mem_id);
      close(dmabuf_fd);
      ret = -ENOMEM;
      goto out_error;
   }
   obj->is_shm = true;  /* backing fd is a memfd, shared to the guest as SHM */

   /* The arena is a pure page pool: it has no GPU VA of its own; its pages
    * reach the GPU through per-BO run stitching. */
   if (!is_arena) {
      ret = kgsl_vbo_bind(kctx, obj, req->iova);
      if (ret) {
         kgsl_renderer_free_object(dctx, &obj->base);
         goto out_error;
      }
   }

   drm_context_object_set_blob_id(dctx, &obj->base, req->blob_id);

   if (is_arena) {
      kctx->arena = obj;
      drm_log("ARENA blessed: pool=0x%" PRIx64 " bytes mem_id=%u", blob_size,
              mem_id);
   }

   drm_dbg("obj=%p blob_id=%u mem_id=%u iova=0x%" PRIx64 " size=0x%" PRIx64,
           (void *)obj, req->blob_id, mem_id, req->iova, blob_size);

   return 0;

out_error:
   if (kctx->shmem)
      kctx->shmem->async_error++;
   return ret;
}

/*
 * GEM_SET_IOVA: (re)bind an existing object at a new iova, or unbind when the
 * guest passes iova == 0 (releasing an imported BO's address).
 */
static int
kgsl_ccmd_gem_set_iova(struct drm_context *dctx, struct vdrm_ccmd_req *hdr)
{
   const struct msm_ccmd_gem_set_iova_req *req = to_msm_ccmd_gem_set_iova_req(hdr);
   struct kgsl_context *kctx = to_kgsl_context(dctx);
   struct kgsl_object *obj = kgsl_object_from_res_id(kctx, req->res_id);

   if (!obj) {
      drm_err("Could not lookup obj: res_id=%u", req->res_id);
      goto out_error;
   }

   /* Never explicitly unbind: a submit referencing this BO may still be on
    * the GPU (drm/msm pins submit BOs until retire; guest turnip relies on
    * that — eager unbinds made the CP read zeros mid-flight, deterministic
    * "opcode error | opcode=0x00000000" hard faults).  KGSL's VBO machinery
    * makes stale ranges safe: a bind holds a reference on the child (pages
    * outlive GPUOBJ_FREE) and a later bind over the same range removes or
    * splits the stale entries and drops their references (kgsl_vbo.c,
    * kgsl_memdesc_add_range).  The range simply stays mapped until the guest
    * reuses the VA — exactly the msm lifetime the guest assumes.
    */
   if (!req->iova) {
      obj->iova = 0;
      return 0;
   }

   /* Paged arena BOs: re-stitch the same runs at the new iova (old ranges
    * stay until overwritten, per the no-unbind rule). */
   if (!obj->mem_id) {
      if (obj->nr_runs && kgsl_vbo_bind_runs(kctx, obj, req->iova))
         goto out_error;
      obj->iova = req->iova;
      return 0;
   }

   if (obj->iova != req->iova) {
      int ret = kgsl_vbo_bind(kctx, obj, req->iova);
      if (ret)
         goto out_error;
   }

   return 0;

out_error:
   if (kctx->shmem)
      kctx->shmem->async_error++;
   return 0;
}

static int
kgsl_map_object(struct kgsl_context *kctx, struct kgsl_object *obj)
{
   if (obj->map)
      return 0;

   /* Paged arena BOs have no backing fd of their own: stitch their runs of
    * the arena memfd into one linear view (mirror of what the guest does with
    * its stage-1 tables). */
   if (!obj->mem_id) {
      if (!kctx->arena || !obj->nr_runs)
         return -ENOENT;
      uint8_t *span = mmap(NULL, obj->blob_size, PROT_NONE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (span == MAP_FAILED)
         return -ENOMEM;
      uint64_t off = 0;
      for (uint32_t i = 0; i < obj->nr_runs && off < obj->blob_size; i++) {
         uint64_t len = MIN2(obj->runs[i].len, obj->blob_size - off);
         if (mmap(span + off, len, PROT_READ | PROT_WRITE,
                  MAP_SHARED | MAP_FIXED, kctx->arena->dmabuf_fd,
                  obj->runs[i].arena_off) == MAP_FAILED) {
            drm_err("stitch mmap failed: %s", strerror(errno));
            munmap(span, obj->blob_size);
            return -ENOMEM;
         }
         off += len;
      }
      obj->map = span;
      return 0;
   }

   /* Our own BOs already keep a host mapping of the memfd for the KGSL useraddr
    * import -- reuse it rather than mapping twice.
    */
   if (obj->host_map) {
      obj->map = obj->host_map;
      return 0;
   }

   uint8_t *map = mmap(NULL, obj->blob_size, PROT_READ | PROT_WRITE,
                       MAP_SHARED, obj->dmabuf_fd, 0);
   if (map == MAP_FAILED) {
      drm_err("mmap failed: %s", strerror(errno));
      return -ENOMEM;
   }

   obj->map = map;
   return 0;
}

static int
kgsl_ccmd_gem_cpu_prep(struct drm_context *dctx, struct vdrm_ccmd_req *hdr)
{
   const struct msm_ccmd_gem_cpu_prep_req *req = to_msm_ccmd_gem_cpu_prep_req(hdr);
   struct kgsl_context *kctx = to_kgsl_context(dctx);
   struct msm_ccmd_gem_cpu_prep_rsp *rsp = kgsl_ctx_rsp(kctx, hdr, sizeof(*rsp));

   if (!rsp)
      return -ENOMEM;

   /* The dmabuf is shared coherently with the guest through the same pages, so
    * once the referencing submit has retired there is nothing extra to do.
    * The guest polls CPU_PREP, so returning ready (0) is correct for coherent
    * memory; non-coherent cache maintenance is handled by the guest's own
    * mapping attributes.
    */
   rsp->ret = 0;
   return 0;
}

static int
kgsl_ccmd_gem_set_name(UNUSED struct drm_context *dctx, UNUSED struct vdrm_ccmd_req *hdr)
{
   /* KGSL has no per-BO debug name; accept and ignore. */
   return 0;
}

static int
kgsl_ccmd_gem_upload(struct drm_context *dctx, struct vdrm_ccmd_req *hdr)
{
   const struct msm_ccmd_gem_upload_req *req = to_msm_ccmd_gem_upload_req(hdr);
   struct kgsl_context *kctx = to_kgsl_context(dctx);

   if (req->pad || req->len > (hdr->len - sizeof(*req))) {
      drm_err("Invalid upload ccmd");
      return -EINVAL;
   }

   struct kgsl_object *obj = kgsl_object_from_res_id(kctx, req->res_id);
   if (!obj) {
      drm_err("No obj: res_id=%u", req->res_id);
      return -ENOENT;
   }

   if (size_add(req->off, req->len) > obj->base.size)
      return -EFAULT;

   int ret = kgsl_map_object(kctx, obj);
   if (ret)
      return ret;

   memcpy(&obj->map[req->off], req->payload, req->len);
   return 0;
}

/*
 * GEM_SUBMIT: translate the msm submit (bo table + cmd table) into a KGSL
 * GPU_COMMAND.  Each IB cmd references the VBO (which holds every BO mapping)
 * at the guest iova; the guest-assigned seqno is passed through as the KGSL
 * user-generated timestamp.
 */
static int
kgsl_ccmd_gem_submit(struct drm_context *dctx, struct vdrm_ccmd_req *hdr)
{
   const struct msm_ccmd_gem_submit_req *req = to_msm_ccmd_gem_submit_req(hdr);
   struct kgsl_context *kctx = to_kgsl_context(dctx);

   size_t sz = sizeof(*req);
   sz = size_add(sz, size_mul(req->nr_bos, sizeof(struct drm_msm_gem_submit_bo)));
   sz = size_add(sz, size_mul(req->nr_cmds, sizeof(struct drm_msm_gem_submit_cmd)));
   if (sz > hdr->len) {
      drm_err("out of bounds: nr_bos=%u, nr_cmds=%u", req->nr_bos, req->nr_cmds);
      return -ENOSPC;
   }

   const struct drm_msm_gem_submit_bo *bos = (void *)req->payload;
   const struct drm_msm_gem_submit_cmd *cmds =
      (void *)&req->payload[req->nr_bos * sizeof(struct drm_msm_gem_submit_bo)];

   if (req->nr_cmds == 0)
      return 0;

   struct kgsl_command_object *kcmds =
      calloc(req->nr_cmds, sizeof(*kcmds));
   if (!kcmds)
      return -ENOMEM;

   unsigned n = 0;
   for (uint32_t i = 0; i < req->nr_cmds; i++) {
      const struct drm_msm_gem_submit_cmd *c = &cmds[i];

      if (c->type != MSM_SUBMIT_CMD_BUF &&
          c->type != MSM_SUBMIT_CMD_IB_TARGET_BUF)
         continue;

      if (c->submit_idx >= req->nr_bos) {
         drm_err("submit_idx %u out of range", c->submit_idx);
         free(kcmds);
         return -EINVAL;
      }

      struct kgsl_object *obj =
         kgsl_object_from_res_id(kctx, bos[c->submit_idx].handle);
      if (!obj || !obj->iova) {
         drm_err("cmd bo res_id=%u has no iova", bos[c->submit_idx].handle);
         free(kcmds);
         return -EINVAL;
      }

      kcmds[n++] = (struct kgsl_command_object) {
         .gpuaddr = obj->iova + c->submit_offset,
         .size = c->size,
         .flags = KGSL_CMDLIST_IB,
         .id = kctx->vbo_id, /* address lives in the VBO's mapping */
      };

      /* NCTX_IB_CHECK: host-side visibility probe for the CP-reads-zeros
       * fault — dump the IB's first/last dwords as the CPU sees them at
       * submit time.  Valid cmdstreams start with type4/type7 headers
       * (0x4xxxxxxx/0x7xxxxxxx); zeros here mean the guest's writes never
       * reached these pages, zeros only at the GPU mean a bad VBO binding. */
      if (getenv("NCTX_IB_CHECK") && !kgsl_map_object(kctx, obj)) {
         const uint32_t *ib = (uint32_t *)(obj->map + c->submit_offset);
         uint32_t nd = c->size / 4;
         drm_log("IB-CHECK iova=0x%" PRIx64 " size=0x%x head=%08x %08x %08x %08x tail=%08x %08x %08x %08x",
                 obj->iova + c->submit_offset, c->size,
                 nd > 0 ? ib[0] : 0, nd > 1 ? ib[1] : 0,
                 nd > 2 ? ib[2] : 0, nd > 3 ? ib[3] : 0,
                 nd > 3 ? ib[nd - 4] : 0, nd > 2 ? ib[nd - 3] : 0,
                 nd > 1 ? ib[nd - 2] : 0, nd > 0 ? ib[nd - 1] : 0);
      }

      /* NCTX_IB_SCAN: walk the IB1 for CP_INDIRECT_BUFFER (type7, opcode
       * 0x3f) calls and verify each IB2 target against our object table:
       * is the target iova inside a bound BO, and is its content non-zero
       * host-side?  Directly answers "unbound target" vs "unwritten pages"
       * for the recurring CP opcode-0 hard fault at ib1+0xD968. */
      if (getenv("NCTX_IB_SCAN") && !kgsl_map_object(kctx, obj)) {
         const uint32_t *ib = (uint32_t *)(obj->map + c->submit_offset);
         uint32_t nd = c->size / 4;
         for (uint32_t w = 0; w + 3 < nd; w++) {
            if ((ib[w] >> 28) != 0x7 || ((ib[w] >> 16) & 0x7f) != 0x3f)
               continue;
            uint64_t tgt = (uint64_t)ib[w + 1] | ((uint64_t)ib[w + 2] << 32);
            uint32_t tsz = ib[w + 3] & 0xfffff; /* dwords */
            /* find a BO covering tgt */
            struct kgsl_object *tobj = NULL;
            hash_table_foreach(dctx->resource_table, ent) {
               struct kgsl_object *o = (struct kgsl_object *)ent->data;
               if (o->iova && tgt >= o->iova && tgt < o->iova + o->blob_size) {
                  tobj = o;
                  break;
               }
            }
            uint32_t thead = 0;
            if (tobj && !kgsl_map_object(kctx, tobj))
               thead = *(uint32_t *)(tobj->map + (tgt - tobj->iova));
            drm_log("IB-SCAN ib1=0x%" PRIx64 "+0x%x -> ib2=0x%" PRIx64 " sz=0x%x %s head=%08x",
                    obj->iova + c->submit_offset, w * 4, tgt, tsz,
                    tobj ? "BOUND" : "UNBOUND!", thead);
            w += 3;
         }
      }
   }

   const struct hash_entry *entry = _mesa_hash_table_search(
      kctx->sq_to_ring_idx_table, (void *)(uintptr_t)req->queue_id);
   if (!entry) {
      drm_err("unknown submitqueue: %u", req->queue_id);
      free(kcmds);
      return -EINVAL;
   }
   unsigned ring_idx = (uintptr_t)entry->data;

   struct kgsl_gpu_command cmd = {
      .flags = KGSL_CMDBATCH_SUBMIT_IB_LIST,
      .cmdlist = (uintptr_t)kcmds,
      .cmdsize = sizeof(struct kgsl_command_object),
      .numcmds = n,
      .context_id = req->queue_id,
      .timestamp = req->fence, /* guest-assigned seqno (USER_GENERATED_TS) */
   };

   int ret = kgsl_ioctl(dctx->fd, IOCTL_KGSL_GPU_COMMAND, &cmd);
   free(kcmds);

   if (ret) {
      drm_err("GPU_COMMAND failed: %s", strerror(errno));
      if (kctx->shmem)
         kctx->shmem->async_error++;
      return 0; /* async error, not a protocol error */
   }


   /* Obtain a sync_file for this submit's completion and feed the ring
    * timeline, so the vdrm ring fence signals when the GPU passes the seqno.
    * NCTX_NO_FENCE: skip the out-fence entirely (submit_fence then retires the
    * ring immediately) -- diagnostic A/B for the post-submit crosvm exit(1).
    */
   if (!getenv("NCTX_NO_FENCE")) {
      struct kgsl_timestamp_event_fence fence_priv = { .fence_fd = -1 };
      struct kgsl_timestamp_event event = {
         .type = KGSL_TIMESTAMP_EVENT_FENCE,
         .timestamp = req->fence,
         .context_id = req->queue_id,
         .priv = &fence_priv, /* void __user *; kernel writes fence_fd back */
         .len = sizeof(fence_priv),
      };

      if (!kgsl_ioctl(dctx->fd, IOCTL_KGSL_TIMESTAMP_EVENT, &event) &&
          fence_priv.fence_fd >= 0) {
         drm_timeline_set_last_fence_fd(&kctx->timelines[ring_idx - 1],
                                        fence_priv.fence_fd);
      }
   }

   return 0;
}

static int
kgsl_ccmd_submitqueue_query(struct drm_context *dctx, struct vdrm_ccmd_req *hdr)
{
   const struct msm_ccmd_submitqueue_query_req *req =
      to_msm_ccmd_submitqueue_query_req(hdr);
   struct kgsl_context *kctx = to_kgsl_context(dctx);
   struct msm_ccmd_submitqueue_query_rsp *rsp =
      kgsl_ctx_rsp(kctx, hdr, size_add(sizeof(*rsp), req->len));

   if (!rsp)
      return -ENOMEM;

   /* KGSL exposes no submitqueue introspection; report empty. */
   rsp->ret = 0;
   rsp->out_len = 0;
   return 0;
}

static int
kgsl_ccmd_wait_fence(struct drm_context *dctx, struct vdrm_ccmd_req *hdr)
{
   const struct msm_ccmd_wait_fence_req *req = to_msm_ccmd_wait_fence_req(hdr);
   struct kgsl_context *kctx = to_kgsl_context(dctx);
   struct msm_ccmd_wait_fence_rsp *rsp = kgsl_ctx_rsp(kctx, hdr, sizeof(*rsp));

   if (!rsp)
      return -ENOMEM;

   /* Non-blocking poll (timeout 0): the guest polls if not yet signaled, so we
    * never stall the single-threaded host.  KGSL returns 0 when the GPU has
    * passed the timestamp, or -ETIMEDOUT / -EDEADLK otherwise; both are
    * reported verbatim.
    */
   struct kgsl_device_waittimestamp_ctxtid wait = {
      .context_id = req->queue_id,
      .timestamp = req->fence,
      .timeout = 0,
   };

   rsp->ret = kgsl_ioctl(dctx->fd, IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID, &wait)
                 ? -errno
                 : 0;
   return 0;
}

static int
kgsl_ccmd_set_debuginfo(UNUSED struct drm_context *dctx, UNUSED struct vdrm_ccmd_req *hdr)
{
   /* Guest process comm/cmdline -- no KGSL equivalent, accept and ignore. */
   return 0;
}

static const struct drm_ccmd ccmd_dispatch[] = {
#define HANDLER(N, n)                                                            \
   [MSM_CCMD_##N] = { #N, kgsl_ccmd_##n, sizeof(struct msm_ccmd_##n##_req) }
   HANDLER(NOP, nop),
   HANDLER(IOCTL_SIMPLE, ioctl_simple),
   HANDLER(GEM_NEW, gem_new),
   HANDLER(GEM_SET_IOVA, gem_set_iova),
   HANDLER(GEM_CPU_PREP, gem_cpu_prep),
   HANDLER(GEM_SET_NAME, gem_set_name),
   HANDLER(GEM_SUBMIT, gem_submit),
   HANDLER(GEM_UPLOAD, gem_upload),
   HANDLER(SUBMITQUEUE_QUERY, submitqueue_query),
   HANDLER(WAIT_FENCE, wait_fence),
   HANDLER(SET_DEBUGINFO, set_debuginfo),
#undef HANDLER
};

/*
 * ---------------------------------------------------------------------------
 * virgl_context vtable
 * ---------------------------------------------------------------------------
 */

static void
kgsl_renderer_attach_resource(struct virgl_context *vctx, struct virgl_resource *res)
{
   struct drm_context *dctx = to_drm_context(vctx);
   struct kgsl_context *kctx = to_kgsl_context(dctx);
   struct kgsl_object *obj = kgsl_object_from_res_id(kctx, res->res_id);

   if (obj)
      return;

   int fd;
   enum virgl_resource_fd_type fd_type = virgl_resource_export_fd(res, &fd);
   if (fd_type != VIRGL_RESOURCE_FD_DMABUF) {
      if (fd_type != VIRGL_RESOURCE_FD_INVALID)
         close(fd);
      return;
   }

   /* Import an externally-created dmabuf (another context, or the display) so
    * this context can reference it.  Its iova arrives later via GEM_SET_IOVA.
    */
   uint32_t mem_id = kgsl_import_dmabuf(dctx->fd, fd, 0);
   if (!mem_id) {
      close(fd);
      return;
   }

   off_t size = lseek(fd, 0, SEEK_END);
   if (size < 0) {
      drm_err("lseek failed: %s", strerror(errno));
      kgsl_free_gpuobj(dctx->fd, mem_id);
      close(fd);
      return;
   }

   struct kgsl_object *nobj =
      kgsl_object_create(mem_id, fd, 0, size, ALIGN_POT(size, getpagesize()));
   if (!nobj) {
      kgsl_free_gpuobj(dctx->fd, mem_id);
      close(fd);
      return;
   }

   drm_context_object_set_res_id(dctx, &nobj->base, res->res_id);
}

static enum virgl_resource_fd_type
kgsl_renderer_export_opaque_handle(struct virgl_context *vctx,
                                   struct virgl_resource *res, int *out_fd)
{
   struct drm_context *dctx = to_drm_context(vctx);
   struct kgsl_context *kctx = to_kgsl_context(dctx);
   struct kgsl_object *obj = kgsl_object_from_res_id(kctx, res->res_id);

   if (!obj) {
      drm_err("invalid res_id %u", res->res_id);
      return VIRGL_RESOURCE_FD_INVALID;
   }

   if (!obj->exportable)
      return VIRGL_RESOURCE_FD_INVALID;

   int fd = os_dupfd_cloexec(obj->dmabuf_fd);
   if (fd < 0) {
      drm_err("dup failed: %s", strerror(errno));
      return VIRGL_RESOURCE_FD_INVALID;
   }

   *out_fd = fd;
   return VIRGL_RESOURCE_FD_DMABUF;
}

static int
kgsl_renderer_get_blob(struct virgl_context *vctx, uint32_t res_id, uint64_t blob_id,
                       uint64_t blob_size, uint32_t blob_flags,
                       struct virgl_context_blob *blob)
{
   struct drm_context *dctx = to_drm_context(vctx);
   struct kgsl_context *kctx = to_kgsl_context(dctx);

   if ((blob_id >> 32) != 0) {
      drm_err("invalid blob_id: %" PRIu64, blob_id);
      return -EINVAL;
   }

   /* blob_id 0 is the msm shmem control buffer. */
   if (blob_id == 0) {
      int ret = drm_context_get_shmem_blob(dctx, "kgsl-shmem", sizeof(*kctx->shmem),
                                           blob_size, blob_flags, blob);
      if (ret)
         return ret;
      kctx->shmem = to_msm_shmem(dctx->shmem);
      return 0;
   }

   if (!drm_context_res_id_unused(dctx, res_id)) {
      drm_err("Invalid res_id %u", res_id);
      return -EINVAL;
   }

   struct kgsl_object *obj = kgsl_object_from_blob_id(kctx, blob_id);
   if (!obj) {
      drm_err("No object for blob_id %" PRIu64, blob_id);
      return -ENOENT;
   }

   if (obj->exported) {
      drm_err("Already exported!");
      return -EINVAL;
   }

   if (obj->blob_size != blob_size) {
      drm_err("Invalid blob size 0x%" PRIx64 " != 0x%" PRIx64, obj->blob_size, blob_size);
      return -EINVAL;
   }

   drm_context_object_set_res_id(dctx, &obj->base, res_id);

   /* Paged arena BOs have no backing fd of their own, but the guest still
    * creates a virtio resource for each one (submits reference BOs by res_id).
    * Hand back a dup of the arena memfd so the resource plumbing is satisfied;
    * the guest never maps these (CPU access goes through its own stage-1
    * stitch of the arena) and the fd dies with the resource. */
   if (!obj->mem_id) {
      if (!kctx->arena)
         return -EINVAL;
      int afd = os_dupfd_cloexec(kctx->arena->dmabuf_fd);
      if (afd < 0)
         return -EINVAL;
      blob->type = VIRGL_RESOURCE_FD_SHM;
      blob->u.fd = afd;
      blob->map_info = VIRGL_RENDERER_MAP_CACHE_WC;
      obj->exported = true;
      obj->exportable = false;
      return 0;
   }

   /* Hand back the backing fd for the guest to map through GuestAccept.  Our own
    * BOs are THP memfds (SHM); externally-imported BOs are dmabufs.
    */
   int fd = os_dupfd_cloexec(obj->dmabuf_fd);
   if (fd < 0) {
      drm_err("dup failed: %s", strerror(errno));
      return -EINVAL;
   }

   blob->type = obj->is_shm ? VIRGL_RESOURCE_FD_SHM : VIRGL_RESOURCE_FD_DMABUF;
   blob->u.fd = fd;

   if (obj->flags & (MSM_BO_CACHED | MSM_BO_CACHED_COHERENT))
      blob->map_info = VIRGL_RENDERER_MAP_CACHE_CACHED;
   else
      blob->map_info = VIRGL_RENDERER_MAP_CACHE_WC;

   obj->exported = true;
   obj->exportable = !!(blob_flags & VIRGL_RENDERER_BLOB_FLAG_USE_MAPPABLE);

   return 0;
}

static int
kgsl_renderer_submit_fence(struct virgl_context *vctx, uint32_t flags,
                           uint32_t ring_idx, uint64_t fence_id)
{
   struct drm_context *dctx = to_drm_context(vctx);
   struct kgsl_context *kctx = to_kgsl_context(dctx);

   if (ring_idx > g_nr_timelines) {
      drm_err("invalid ring_idx: %u", ring_idx);
      return -EINVAL;
   }

   /* ring_idx 0 is the host-CPU timeline: already passed by the time we are
    * called.  Likewise if the queue produced no fence fd, signal immediately.
    */
   if (ring_idx == 0 || kctx->timelines[ring_idx - 1].last_fence_fd < 0) {
      vctx->fence_retire(vctx, ring_idx, fence_id);
      return 0;
   }

   return drm_timeline_submit_fence(&kctx->timelines[ring_idx - 1], flags, fence_id);
}

static void
kgsl_renderer_fence_retire(struct virgl_context *vctx, uint32_t ring_idx,
                           uint64_t fence_id)
{
   vctx->fence_retire(vctx, ring_idx, fence_id);
}

static void
kgsl_renderer_destroy(struct virgl_context *vctx)
{
   struct drm_context *dctx = to_drm_context(vctx);
   struct kgsl_context *kctx = to_kgsl_context(dctx);

   for (unsigned i = 0; i < g_nr_timelines; i++)
      drm_timeline_fini(&kctx->timelines[i]);

   drm_context_deinit(dctx); /* frees remaining objects via free_object */


   /* The VBO is global (shared by all contexts) -- do not free it here. */
   g_va_slices_used &= ~(1u << kctx->va_slice_idx);
   if (kctx->dma_heap_fd >= 0)
      close(kctx->dma_heap_fd);

   _mesa_hash_table_destroy(kctx->sq_to_ring_idx_table, NULL);

   free(kctx);
}

/*
 * Reserve the whole guest VA range as a single VBO, walking the size ladder
 * down until the kernel accepts it.  Returns 0 and fills *out_id / *out_base /
 * *out_size on success.
 */
static int
kgsl_alloc_vbo(int fd, uint32_t *out_id, uint64_t *out_base, uint64_t *out_size)
{
   /* The VBO flag lives in bit 34, so it can only be expressed through the
    * 64-bit-flags GPUOBJ_ALLOC ioctl (GPUMEM_ALLOC_ID's flags are 32-bit).
    * For a VBO, `size` is the virtual address range to reserve; the VBO flag
    * makes it virtual-only (no physical backing).  The alloc path does not use
    * va_len.  GPUOBJ_ALLOC does not return the gpuaddr, so query it separately
    * with GPUOBJ_INFO.
    */
   for (unsigned i = 0; i < ARRAY_SIZE(vbo_size_ladder); i++) {
      struct kgsl_gpuobj_alloc req = {
         .size = vbo_size_ladder[i],
         .flags = KGSL_MEMFLAGS_VBO,
      };

      if (kgsl_ioctl(fd, IOCTL_KGSL_GPUOBJ_ALLOC, &req)) {
         drm_dbg("GPUOBJ_ALLOC(VBO, size=0x%" PRIx64 ") failed: %s",
                 vbo_size_ladder[i], strerror(errno));
         continue;
      }

      struct kgsl_gpuobj_info info = { .id = req.id };
      if (kgsl_ioctl(fd, IOCTL_KGSL_GPUOBJ_INFO, &info)) {
         drm_dbg("GPUOBJ_INFO(id=%u) failed: %s", req.id, strerror(errno));
         kgsl_free_gpuobj(fd, req.id);
         continue;
      }

      drm_log("VBO ok: id=%u gpuaddr=0x%" PRIx64 " size=0x%" PRIx64,
              req.id, (uint64_t)info.gpuaddr, vbo_size_ladder[i]);
      *out_id = req.id;
      *out_base = info.gpuaddr;
      *out_size = vbo_size_ladder[i];
      return 0;
   }

   drm_err("VBO allocation failed at every size: %s", strerror(errno));
   return -ENOMEM;
}

struct virgl_context *
kgsl_renderer_create(int fd, UNUSED size_t debug_len, UNUSED const char *debug_name)
{
   drm_log("");

   struct kgsl_context *kctx =
      calloc(1, sizeof(*kctx) + g_nr_timelines * sizeof(kctx->timelines[0]));
   if (!kctx)
      return NULL;

   kctx->dma_heap_fd = open(dma_heap_path, O_RDONLY | O_CLOEXEC);
   if (kctx->dma_heap_fd < 0) {
      drm_err("open %s failed: %s", dma_heap_path, strerror(errno));
      goto fail_early;
   }

   /* All contexts share the single global VBO allocated at probe time.  KGSL
    * memory ids are per-process, so g_vbo_id is valid on this context's fd too.
    * Sharing one VBO keeps va_start stable regardless of what else (e.g. the
    * virgl 2D path) allocates GPU VA between probe and context creation -- a
    * per-context VBO would drift and no longer match the reported va_start.
    */
   kctx->vbo_id = g_vbo_id;
   kctx->vbo_base = g_va_start;

   /* Claim a disjoint VA slice for this context. */
   {
      unsigned nslices = MIN2(32, g_va_size / KGSL_VA_SLICE_SIZE);
      int idx = -1;
      for (unsigned i = 0; i < nslices; i++) {
         if (!(g_va_slices_used & (1u << i))) {
            idx = i;
            break;
         }
      }
      if (idx < 0) {
         drm_err("out of VA slices (%u contexts)", nslices);
         goto fail_early;
      }
      g_va_slices_used |= 1u << idx;
      kctx->va_slice_idx = idx;
      kctx->va_slice_base = g_va_start + (uint64_t)idx * KGSL_VA_SLICE_SIZE;
      kctx->va_slice_size = KGSL_VA_SLICE_SIZE;
      drm_log("VA slice %d: [0x%" PRIx64 ", +8GB)", idx, kctx->va_slice_base);
   }

   if (!drm_context_init(&kctx->base, fd, ccmd_dispatch, ARRAY_SIZE(ccmd_dispatch)))
      goto fail_early;

   kctx->sq_to_ring_idx_table = _mesa_hash_table_create_u32_keys(NULL);

   for (unsigned i = 0; i < g_nr_timelines; i++)
      drm_timeline_init(&kctx->timelines[i], &kctx->base.base, "kgsl-sync",
                        i + 1, kgsl_renderer_fence_retire);

   kctx->base.base.destroy = kgsl_renderer_destroy;
   kctx->base.base.attach_resource = kgsl_renderer_attach_resource;
   kctx->base.base.export_opaque_handle = kgsl_renderer_export_opaque_handle;
   kctx->base.base.get_blob = kgsl_renderer_get_blob;
   kctx->base.base.submit_fence = kgsl_renderer_submit_fence;
   kctx->base.free_object = kgsl_renderer_free_object;

   /* msm wire protocol only requires 4-byte alignment. */
   kctx->base.ccmd_alignment = 4;

   return &kctx->base.base;

fail_early:
   if (kctx->dma_heap_fd >= 0)
      close(kctx->dma_heap_fd);
   close(fd);
   free(kctx);
   return NULL;
}

/*
 * ---------------------------------------------------------------------------
 * Probe
 * ---------------------------------------------------------------------------
 */

int
kgsl_renderer_probe(int fd, struct virgl_renderer_capset_drm *capset)
{
   drm_log("");

   struct kgsl_devinfo devinfo = { 0 };
   if (kgsl_getprop(fd, KGSL_PROP_DEVICE_INFO, &devinfo, sizeof(devinfo))) {
      drm_err("KGSL_PROP_DEVICE_INFO failed: %s", strerror(errno));
      return -ENOTSUP;
   }

   /* freedreno's device table keys chip_id with the low (fuse/patch) byte 0 --
    * e.g. Adreno 830 is FD830 = 0x44050000 -- but KGSL reports the actual fuse
    * revision (e.g. 0x44050001).  fd_dev_info() needs an exact match (or a 0xff
    * patch wildcard the table doesn't carry), so strip the fuse byte to hit the
    * canonical entry; otherwise the guest enumerates zero devices.
    */
   g_chip_id = devinfo.chip_id & ~UINT64_C(0xff);
   g_gpu_id = ((devinfo.chip_id >> 24) & 0xff) * 100 +
              ((devinfo.chip_id >> 16) & 0xff) * 10 +
              ((devinfo.chip_id >>  8) & 0xff);
   drm_log("KGSL chip_id=0x%" PRIx64 " -> reported 0x%" PRIx64 " gpu_id=%u",
           (uint64_t)devinfo.chip_id, g_chip_id, g_gpu_id);
   g_gmem_size = devinfo.gmem_sizebytes;

   uint64_t gmem_iova = 0;
   kgsl_getprop(fd, KGSL_PROP_UCHE_GMEM_VADDR, &gmem_iova, sizeof(gmem_iova));
   g_gmem_base = gmem_iova;

   g_highest_bank_bit = 0;
   kgsl_getprop(fd, KGSL_PROP_HIGHEST_BANK_BIT, &g_highest_bank_bit,
                sizeof(g_highest_bank_bit));

   if (kgsl_getprop(fd, KGSL_PROP_UCHE_TRAP_BASE, &g_uche_trap_base,
                    sizeof(g_uche_trap_base)))
      g_uche_trap_base = 0x1fffffffff000ull; /* known hardcoded fallback */

   /* Reserve the one global VBO covering the guest VA range and report its
    * base/size as va_start/va_size.  Keep it (and an fd to keep KGSL's
    * per-process memory alive) for the whole renderer lifetime; every context
    * binds into this same VBO.
    */
   if (kgsl_alloc_vbo(fd, &g_vbo_id, &g_va_start, &g_va_size))
      return -ENOTSUP;
   g_keeper_fd = dup(fd);
   if (g_keeper_fd < 0) {
      drm_err("dup keeper fd failed: %s", strerror(errno));
      return -ENOMEM;
   }

   /* KGSL supports a single priority band per drawctxt here; expose one ring.
    * (Priorities are encoded in the drawctxt flags but map to one timeline.)
    */
   g_nr_timelines = 1;

   /* Advertise a recent drm/msm uabi so the guest turnip enables the modern
    * code paths (userspace IOVA, FENCE_SN_IN).  There is no real msm kernel
    * behind us, but the guest gates features on these numbers.
    */
   capset->version_major = 1;
   capset->version_minor = 12;
   capset->version_patchlevel = 0;

   capset->wire_format_version = 2;
   capset->u.msm.has_cached_coherent = 1;
   capset->u.msm.priorities = g_nr_timelines;
   capset->u.msm.va_start = g_va_start;
   capset->u.msm.va_size = g_va_size;
   capset->u.msm.gpu_id = g_gpu_id;
   capset->u.msm.gmem_size = g_gmem_size;
   capset->u.msm.gmem_base = g_gmem_base;
   capset->u.msm.chip_id = g_chip_id;
   capset->u.msm.highest_bank_bit = g_highest_bank_bit;
   capset->u.msm.uche_trap_base = g_uche_trap_base;
   capset->u.msm.has_preemption = VIRTGPU_CAP_BOOL_FALSE;
   capset->u.msm.has_raytracing = VIRTGPU_CAP_BOOL_FALSE;

   drm_log("gpu_id=%u chip_id=0x%" PRIx64 " va_start=0x%" PRIx64
           " va_size=0x%" PRIx64 " gmem_size=%u hbb=%u",
           g_gpu_id, g_chip_id, g_va_start, g_va_size, g_gmem_size,
           g_highest_bank_bit);

   if (!g_va_size) {
      drm_log("no usable VA range");
      return -ENOTSUP;
   }

   return 0;
}
