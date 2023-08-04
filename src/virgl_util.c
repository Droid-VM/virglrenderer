/**************************************************************************
 *
 * Copyright (C) 2019 Chromium.
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "virgl_util.h"

#include <errno.h>
#ifdef HAVE_EVENTFD_H
#include <sys/eventfd.h>
#endif
#include <unistd.h>

#include "util/os_misc.h"
#include "util/u_pointer.h"

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#if ENABLE_TRACING == TRACE_WITH_PERFETTO
#include <vperfetto-min.h>
#endif

#if ENABLE_TRACING == TRACE_WITH_STDERR
#include <stdio.h>
#endif

uint32_t hash_func_u32(const void *key)
{
   intptr_t ip = pointer_to_intptr(key);
   return (uint32_t)(ip & 0xffffffff);
}

bool equal_func(const void *key1, const void *key2)
{
   return key1 == key2;
}

bool has_eventfd(void)
{
#ifdef HAVE_EVENTFD_H
   return true;
#else
   return false;
#endif
}

int create_eventfd(unsigned int initval)
{
#ifdef HAVE_EVENTFD_H
   return eventfd(initval, EFD_CLOEXEC | EFD_NONBLOCK);
#else
   (void)initval;
   return -1;
#endif
}

int write_eventfd(int fd, uint64_t val)
{
   const char *buf = (const char *)&val;
   size_t count = sizeof(val);
   ssize_t ret = 0;

   while (count) {
      ret = write(fd, buf, count);
      if (ret < 0) {
         if (errno == EINTR)
            continue;
         break;
      }
      count -= ret;
      buf += ret;
   }

   return count ? -1 : 0;
}

void flush_eventfd(int fd)
{
    ssize_t len;
    uint64_t value;
    do {
       len = read(fd, &value, sizeof(value));
    } while ((len == -1 && errno == EINTR) || len == sizeof(value));
}

const struct log_levels_lut {
   char *name;
   enum virgl_log_level_flags log_level;
} log_levels_table[] = {
   {"debug", VIRGL_LOG_LEVEL_DEBUG},
   {"info", VIRGL_LOG_LEVEL_INFO},
   {"warning", VIRGL_LOG_LEVEL_WARNING},
   {"error", VIRGL_LOG_LEVEL_ERROR},
   { NULL, 0 },
};

#ifndef NDEBUG
static enum virgl_log_level_flags virgl_log_level = VIRGL_LOG_LEVEL_WARNING;
#else
static enum virgl_log_level_flags virgl_log_level = VIRGL_LOG_LEVEL_ERROR;
#endif
static bool virgl_log_level_initialized = false;

static
void virgl_default_logger(UNUSED enum virgl_log_level_flags log_level,
                          const char *message,
                          UNUSED void* user_data)
{
   static FILE* fp = NULL;
   if (NULL == fp) {
      const char* log = getenv("VIRGL_LOG_FILE");
      if (log) {
         char *log_prefix = strdup(log);
         char *log_suffix = strstr(log_prefix, "%PID%");
         if (log_suffix) {
            *log_suffix = 0;
            log_suffix += 5;
            int len = strlen(log) + 32;
            char *name = malloc(len);
            snprintf(name, len, "%s%d%s", log_prefix, getpid(), log_suffix);
            fp = fopen(name, "a");
            free(name);
         } else {
            fp = fopen(log, "a");
         }
         free(log_prefix);
         if (NULL == fp) {
            fprintf(stderr, "Can't open %s\n", log);
            fp = stderr;
         }
      } else {
            fp = stderr;
      }
   }

   if (!virgl_log_level_initialized) {
      const char* log_level_env = getenv("VIRGL_LOG_LEVEL");
      if (log_level_env != NULL && log_level_env[0] != '\0') {
         const struct log_levels_lut *lut = log_levels_table;
         while (lut->name) {
            if (!strcmp(lut->name, log_level_env)) {
               virgl_log_level = lut->log_level;
               break;
            }
         }

         if (!lut->name)
            fprintf(fp, "Unknown log level %s requested\n", log_level_env);
      }

      virgl_log_level_initialized = true;
   }

   if (log_level < virgl_log_level)
      return;

   fprintf(fp, "%s", message);
   fflush(fp);
}

static struct {
   virgl_log_callback_type log_cb;
   virgl_free_data_callback_type free_data_cb;
   void *user_data;
} virgl_log_data = { virgl_default_logger, NULL, NULL };

void virgl_log_set_handler(virgl_log_callback_type log_cb,
                           void *user_data,
                           virgl_free_data_callback_type free_data_cb)
{
   if (virgl_log_data.free_data_cb)
      virgl_log_data.free_data_cb(virgl_log_data.user_data);

   virgl_log_data.log_cb = log_cb;
   virgl_log_data.free_data_cb = free_data_cb;
   virgl_log_data.user_data = user_data;
}

void virgl_logv(const char *fmt, va_list va)
{
   char *str = NULL;

   if (!virgl_log_data.log_cb)
      return;

   if (vasprintf(&str, fmt, va) < 0)
      return;

   virgl_log_data.log_cb(VIRGL_LOG_LEVEL_INFO, str, virgl_log_data.user_data);
   free (str);
}

#if ENABLE_TRACING == TRACE_WITH_PERCETTO
PERCETTO_CATEGORY_DEFINE(VIRGL_PERCETTO_CATEGORIES)

void trace_init(void)
{
  PERCETTO_INIT(PERCETTO_CLOCK_DONT_CARE);
}
#endif

#if ENABLE_TRACING == TRACE_WITH_PERFETTO
static void on_tracing_state_change(bool enabled) {
    virgl_log("%s: tracing state change: %d\n", __func__, enabled);
}

void trace_init(void)
{
   struct vperfetto_min_config config = {
      .on_tracing_state_change = on_tracing_state_change,
      .init_flags = VPERFETTO_INIT_FLAG_USE_SYSTEM_BACKEND,
            .filename = NULL,
            .shmem_size_hint_kb = 32 * 1024,
   };

   vperfetto_min_startTracing(&config);
}

const char *trace_begin(const char *scope)
{
   vperfetto_min_beginTrackEvent_VMM(scope);
   return scope;
}

void trace_end(const char **dummy)
{
   (void)dummy;
   vperfetto_min_endTrackEvent_VMM();
}
#endif

#if ENABLE_TRACING == TRACE_WITH_STDERR
static int nesting_depth = 0;
void trace_init(void)
{
}

const char *trace_begin(const char *scope)
{
   for (int i = 0; i < nesting_depth; ++i)
      fprintf(stderr, "  ");

   fprintf(stderr, "ENTER:%s\n", scope);
   nesting_depth++;

   return scope;
}

void trace_end(const char **func_name)
{
   --nesting_depth;
   for (int i = 0; i < nesting_depth; ++i)
      fprintf(stderr, "  ");
   fprintf(stderr, "LEAVE %s\n", *func_name);
}
#endif
