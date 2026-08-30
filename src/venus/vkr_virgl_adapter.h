/*
 * Copyright 2026 Droid-VM
 * SPDX-License-Identifier: MIT
 */

#ifndef VKR_VIRGL_ADAPTER_H
#define VKR_VIRGL_ADAPTER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct virgl_context;

#ifdef ENABLE_VENUS

/* In-process bridge between this fork's struct virgl_context world and the
 * vendored upstream-1.3 venus renderer, which is only reachable through the
 * ctx_id-keyed vkr_renderer_*() API (upstream drives it from the render
 * server; DroidVM keeps everything in one process).
 */

bool
vkr_adapter_renderer_init(uint32_t virgl_flags);

void
vkr_adapter_renderer_fini(void);

void
vkr_adapter_renderer_reset(void);

size_t
vkr_adapter_get_capset(void *capset);

struct virgl_context *
vkr_adapter_context_create(uint32_t ctx_id, size_t nlen, const char *name);

#else /* ENABLE_VENUS */

#include "virgl_util.h"

static inline bool
vkr_adapter_renderer_init(UNUSED uint32_t virgl_flags)
{
   virgl_error("Vulkan support was not enabled in virglrenderer\n");
   return false;
}

static inline void
vkr_adapter_renderer_fini(void)
{
}

static inline void
vkr_adapter_renderer_reset(void)
{
}

static inline size_t
vkr_adapter_get_capset(UNUSED void *capset)
{
   return 0;
}

static inline struct virgl_context *
vkr_adapter_context_create(UNUSED uint32_t ctx_id,
                           UNUSED size_t nlen,
                           UNUSED const char *name)
{
   return NULL;
}

#endif /* ENABLE_VENUS */

#endif /* VKR_VIRGL_ADAPTER_H */
