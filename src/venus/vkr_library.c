/*
 * Copyright 2021 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "vkr_common.h"

#include <dlfcn.h>

#include "vkr_library.h"

void
vkr_library_preload_icd(void)
{
#ifdef ENABLE_VULKAN_PRELOAD
   struct vulkan_library lib = { 0 };

   if (!vkr_library_load(&lib))
      return;

   /* Get vkGetInstanceProcAddr from libvulkan */
   PFN_vkGetInstanceProcAddr get_proc_addr = lib.GetInstanceProcAddr;

   PFN_vkEnumerateInstanceExtensionProperties enumerate_inst_ext_props =
      (PFN_vkEnumerateInstanceExtensionProperties)get_proc_addr(
         VK_NULL_HANDLE, "vkEnumerateInstanceExtensionProperties");
   if (enumerate_inst_ext_props) {
      /* this makes the Vulkan loader loads ICDs */
      uint32_t unused_count;
      enumerate_inst_ext_props(NULL, &unused_count, NULL);
   }

   vkr_library_unload(&lib);
#endif
}

#if defined(ENABLE_VULKAN_DLOAD)

/* DroidVM: the driver named by ANDROID_EMU_VK_LOADER_PATH takes precedence
 * over any system loader -- crosvm must run on the app-bundled host turnip,
 * never on the vendor ICD (same contract as gfxstream's VulkanDispatch).
 * That .so is an Android Vulkan HAL exporting only the "HMI" hw_module
 * symbol, so vkGetInstanceProcAddr must be obtained by opening its hwvulkan
 * device ("vk0").  The frozen libhardware ABI is mirrored inline, exactly as
 * gfxstream host/vulkan/VulkanDispatch.cpp does (that layout is proven
 * against the bundled turnip on this device).
 *
 * The resolved driver is cached for the process lifetime: contexts load and
 * unload the library per guest instance, and re-opening the HAL device on
 * every context create buys nothing.
 */

struct vkr_hw_module_methods {
   int (*open)(const void *module, const char *id, void **device);
};
struct vkr_hw_module {
   uint32_t tag;
   uint16_t module_api_version;
   uint16_t hal_api_version;
   const char *id;
   const char *name;
   const char *author;
   struct vkr_hw_module_methods *methods;
   void *dso;
#ifdef __LP64__
   uint64_t reserved[32 - 7];
#else
   uint32_t reserved[32 - 7];
#endif
};
struct vkr_hw_device {
   uint32_t tag;
   uint32_t version;
   void *module;
#ifdef __LP64__
   uint64_t reserved[12];
#else
   uint32_t reserved[12];
#endif
   int (*close)(void *device);
};
struct vkr_hwvulkan_device {
   struct vkr_hw_device common;
   PFN_vkEnumerateInstanceExtensionProperties EnumerateInstanceExtensionProperties;
   PFN_vkCreateInstance CreateInstance;
   PFN_vkGetInstanceProcAddr GetInstanceProcAddr;
};

static struct vulkan_library vkr_cached_library;

static PFN_vkGetInstanceProcAddr
vkr_library_resolve_gipa(void *handle)
{
   /* Clear any existing error */
   dlerror();

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
   /* ISO C forbids conversion of object pointer to function pointer type */
   PFN_vkGetInstanceProcAddr gipa =
      (PFN_vkGetInstanceProcAddr)dlsym(handle, "vkGetInstanceProcAddr");
   if (!gipa)
      gipa = (PFN_vkGetInstanceProcAddr)dlsym(handle, "vk_icdGetInstanceProcAddr");
#pragma GCC diagnostic pop
   if (gipa)
      return gipa;

   /* Android Vulkan HAL (turnip): bridge through the hwvulkan device */
   struct vkr_hw_module *mod = (struct vkr_hw_module *)dlsym(handle, "HMI");
   if (mod && mod->methods && mod->methods->open) {
      struct vkr_hwvulkan_device *dev = NULL;
      int ret = mod->methods->open(mod, "vk0", (void **)&dev);
      if (ret == 0 && dev) {
         vkr_log("bridged Android Vulkan HAL via HMI, gipa=%p",
                 (void *)dev->GetInstanceProcAddr);
         return dev->GetInstanceProcAddr;
      }
      vkr_log("hwvulkan open() failed: %d", ret);
   }
   return NULL;
}

bool
vkr_library_load(struct vulkan_library *lib)
{
   if (lib->handle)
      return true;

   if (vkr_cached_library.handle) {
      *lib = vkr_cached_library;
      return true;
   }

   const char *env_path = getenv("ANDROID_EMU_VK_LOADER_PATH");
   if (env_path && env_path[0]) {
      lib->handle = dlopen(env_path, RTLD_NOW | RTLD_LOCAL);
      vkr_log("dlopen('%s') -> %p", env_path, lib->handle);
   }
#ifdef __APPLE__
   if (lib->handle == NULL)
      lib->handle = dlopen("libvulkan.dylib", RTLD_NOW | RTLD_LOCAL);
   if (lib->handle == NULL)
      lib->handle = dlopen("libvulkan.1.dylib", RTLD_NOW | RTLD_LOCAL);
   if (lib->handle == NULL)
      lib->handle = dlopen("libMoltenVK.dylib", RTLD_NOW | RTLD_LOCAL);
#else
   if (lib->handle == NULL)
      lib->handle = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
   if (lib->handle == NULL)
      lib->handle = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
#endif
   if (lib->handle == NULL) {
      vkr_log("failed to open libvulkan: %s", dlerror());
      return false;
   }

   lib->GetInstanceProcAddr = vkr_library_resolve_gipa(lib->handle);
   if (lib->GetInstanceProcAddr == NULL) {
      vkr_log("failed to resolve vkGetInstanceProcAddr");
      dlclose(lib->handle);
      lib->handle = NULL;
      return false;
   }

   vkr_cached_library = *lib;
   return true;
}

#if 0 /* DroidVM: replaced by the cached loader above */
bool
vkr_library_load_orig(struct vulkan_library *lib)
{
   if (lib->handle)
      return true;

#ifdef __APPLE__
   lib->handle = dlopen("libvulkan.dylib", RTLD_NOW | RTLD_LOCAL);
   if (lib->handle == NULL)
      lib->handle = dlopen("libvulkan.1.dylib", RTLD_NOW | RTLD_LOCAL);
   if (lib->handle == NULL)
      lib->handle = dlopen("libMoltenVK.dylib", RTLD_NOW | RTLD_LOCAL);
#else
   lib->handle = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
   if (lib->handle == NULL)
      lib->handle = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
#endif
   if (lib->handle == NULL) {
      vkr_log("failed to open libvulkan: %s", dlerror());
      return false;
   }

   /* Clear any existing error */
   dlerror();

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
   /* ISO C forbids conversion of object pointer to function pointer type */
   lib->GetInstanceProcAddr =
      (PFN_vkGetInstanceProcAddr)dlsym(lib->handle, "vkGetInstanceProcAddr");
#pragma GCC diagnostic pop

   char *error = dlerror();
   if (error != NULL) {
      vkr_log("dlerror: %s", error);
      goto fail;
   }

   if (lib->GetInstanceProcAddr == NULL) {
      vkr_log("failed to load vkGetInstanceProcAddr: %s", dlerror());
      goto fail;
   }

   return true;

fail:
   dlclose(lib->handle);
   lib->handle = NULL;
   return false;
}
#endif /* DroidVM: replaced by the cached loader above */

void
vkr_library_unload(struct vulkan_library *lib)
{
   /* DroidVM: the process-lifetime cache owns the dlopen reference; a
    * per-context unload must not drop it (the cached GetInstanceProcAddr
    * would dangle for every later context).
    */
   if (lib->handle && lib->handle != vkr_cached_library.handle)
      dlclose(lib->handle);
   lib->GetInstanceProcAddr = NULL;
   lib->handle = NULL;
}

#endif /* ENABLE_VULKAN_DLOAD */

bool
vkr_library_has_portability_enumeration(
   PFN_vkEnumerateInstanceExtensionProperties enum_inst_ext_props)
{
   uint32_t property_count = 0;
   VkExtensionProperties *properties;
   bool has_portability_enumeration = false;

   VkResult ret = enum_inst_ext_props(NULL, &property_count, NULL);
   if (ret != VK_SUCCESS)
      return false;

   properties = calloc(property_count, sizeof(*properties));
   if (!properties)
      return false;

   ret = enum_inst_ext_props(NULL, &property_count, properties);
   if (ret != VK_SUCCESS) {
      free(properties);
      return false;
   }

   for (uint32_t i = 0; i < property_count; i++) {
      if (!strcmp(properties[i].extensionName,
                  VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
         has_portability_enumeration = true;
         break;
      }
   }
   free(properties);
   return has_portability_enumeration;
}
