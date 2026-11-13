# drm2kgsl fence coalescing and remaining fence overhead

## Goal

Reduce the CPU cost of correct virtio-gpu fence completion without reporting a
fence before the corresponding KGSL end-of-pipeline timestamp has retired.
`NCTX_NO_FENCE=1` is only a diagnostic ceiling: it reports completion without
waiting for the GPU and is not a valid production mode.

This document records the implemented timestamp-fence coalescing, the deployed
context-zero fix, the host scheduling A/B, and the remaining optimization
candidates.  The latest tests show that benchmark results are not comparable
unless the guest vCPU scheduling policy is also held constant.

## Required semantics

`VIRGL_RENDERER_FENCE_FLAG_MERGEABLE` permits notifications for fences on one
timeline to be coalesced into a notification for a newer fence.  The flag that
matters is the flag on each older fence whose individual notification is
omitted.

Virtio-gpu defines `fence_id` as a sequence number on the synchronization
timeline selected by the context and ring.  When a command associated with a
fence completes, the device completes every outstanding command with
`fence_id` less than or equal to that fence on the same timeline.  Crosvm
implements this by draining matching descriptors with
`fence_id <= completed_fence.fence_id`.

Consequently, a correct optimization may reduce waits and notifications, but
it may not signal a fence before the GPU timestamp associated with it has
retired.  This is the fundamental difference from `NCTX_NO_FENCE=1`.

## Implemented behavior

The DRM timeline always waits for the head fence.  After that fence completes,
the worker walks forward over already-retired fences.  It omits a fence's
callback only when that fence is mergeable and retires the newest completed
fence in the run.  This opportunistic merge does not deliberately add latency.

The merge obeys these rules:

1. Only fences on the same timeline, using the same fence representation and
   KGSL context, are merged.
2. Every fence whose individual notification is omitted must be mergeable.
3. A non-mergeable fence cannot be crossed.
4. A fence is retired only after its real sync file or KGSL timestamp signals.

The timestamp probe uses
`IOCTL_KGSL_CMDSTREAM_READTIMESTAMP_CTXTID` with
`KGSL_TIMESTAMP_RETIRED`.  It is nonblocking and therefore safe for the
opportunistic walk under `fence_mutex`.  The blocking
`WAITTIMESTAMP_CTXTID` hook is never used as a probe because its timeout is
three seconds.

The timestamp path is enabled with:

```sh
NCTX_WAITTS=1
```

All hot `NCTX_*` environment settings are parsed once with `pthread_once`.
This removes the repeated `getenv()` cost previously visible in every submit.

## Measurements

### Original comparison

The initial comparison used the same crosvm and virglrenderer binaries and an
Adreno 830 at a 900 MHz maximum clock:

| Mode | FPS | CPU events/fence | GPU worker/fence | kgsl-sync/fence |
| --- | ---: | ---: | ---: | ---: |
| sync_file | 10062 | 12680.5 | 9898.8 | 1819.9 |
| timestamp wait | 10787 | 10987.6 | 8531.9 | 1536.0 |
| no fence, invalid ceiling | 16385 | 6488.2 | 5793.9 | 0 |

The timestamp run queued 107883 fences, issued 62477 blocking waits, and
opportunistically merged 45406 already-retired fences.  Thus 42.09% of the
fences were already covered by a newer completion notification.  The remaining
gap was not evidence that merging was disabled.

### Removed proactive wait-target experiment

An earlier build allowed the worker to skip the head and deliberately wait for
a bounded newer timestamp.  Representative results were:

| Wait-target limit | vkmark score |
| ---: | ---: |
| 8 | 10111 |
| 1 | 10428, 10833 |

Subsequent repetitions were approximately equal.  There is no stable evidence
that deliberately waiting for a newer target improves this workload.  The
extra coalescing is offset by completion latency or other serialized work.  The
proactive mechanism and its runtime configuration were therefore removed.  The
worker now always waits for the head before performing the opportunistic merge.

### Pre-context-fix simpleperf profile

The pre-fix build was profiled with exactly this guest command:

```sh
env \
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/freedreno_icd.aarch64.json \
  VK_DRIVER_FILES=/usr/share/vulkan/icd.d/freedreno_icd.aarch64.json \
  MESA_LOADER_DRIVER_OVERRIDE=zink \
  vkmark --winsys headless --size 800x600 -b clear
```

Omitting these environment variables changes the rendered stack and produced
only roughly 4000 to 6000 FPS, so such runs are not comparable.

The unprofiled warm-up scored 10984.  The profiled run scored 10481, captured
21186 samples with zero lost samples, and recorded 1,153,824,495 CPU cycles.
Artifacts are in:

```text
out/simpleperf/timestamp-batch1-current-20260814/perf.data
out/simpleperf/timestamp-batch1-current-20260814/callgraph.txt
out/simpleperf/timestamp-batch1-current-20260814/vkmark.txt
```

Event distribution by thread was:

| Thread | CPU cycles | Share |
| --- | ---: | ---: |
| GPU worker (`binder:18168_2`) | 850,970,808 | 73.75% |
| combined `kgsl-sync` workers | 193,897,559 | 16.80% |
| `crosvm_vcpu0` | 107,778,295 | 9.34% |

Important inclusive costs in this profile were:

| Path | Share of total cycles |
| --- | ---: |
| `vrend_renderer_force_ctx_0` | 23.93% |
| `thread_sync` | 16.36% |
| `write_context_fence` | 10.29% |
| `create_fence_handler` | 8.97% |
| `trigger_interrupt` | 5.20% |
| `Rutabaga::create_fence` | 4.53% |
| `context_create_fence` | 4.17% |
| `kgsl_renderer_submit_fence` | 2.90% |
| `drm_timeline_submit_fence` | 2.49% |
| `kgsl_timeline_wait_timestamp` | 1.90% |
| `kgsl_timeline_poll_timestamp` | 1.70% |

These are inclusive call-graph percentages and must not be added together.

### Deployed context-zero fix

The pre-fix largest hotspot was outside the fence wait loop.  Crosvm called
`VirtioGpu::force_ctx_0()` before every decoded virtio-gpu command.  With a
virglrenderer default component, this reaches:

```text
vrend_renderer_force_ctx_0
  -> vrend_winsys_make_context_current
  -> virgl_egl_make_context_current
  -> eglMakeCurrent
```

That path accounted for 23.93% of total sampled cycles in the DRM-native
workload.  It also pulls Android EGL, vendor GLES, and `getuid()` work into the
GPU worker even though a DRM capset `SUBMIT_3D` is handled by the native DRM
context and does not use vrend's OpenGL context.

The call is legacy behavior: it predates DRM native contexts and protects
operations that may use virgl context zero, GL resources, or display/cursor
state.  It should not simply be removed globally.  In particular, `ctx_id != 0`
is not a sufficient test because virgl/virgl2 rendering contexts also have
nonzero IDs and still require correct GL context management.

The deployed implementation makes the decision context/capset aware:

1. Preserve `force_ctx_0()` for global, 2D, display, cursor, and virgl/virgl2
   operations.
2. Skip it only for `SUBMIT_3D` commands whose context is known to use
   `RUTABAGA_CAPSET_DRM`.
3. Retain the context's capset ID and expose a query, because
   `RutabagaComponentType::VirglRenderer` alone cannot distinguish virgl2 from
   the DRM capset: both are implemented by the virglrenderer component.
4. Preserve the old behavior for an unknown context and for every command that
   is not a confirmed DRM submit.

After deployment, a 20-second profile captured 16548 samples with no loss.
`vrend_renderer_force_ctx_0`, `virgl_egl_make_context_current`, and
`eglMakeCurrent` fell from 23.93% to 0.04% inclusive.  The new important
inclusive paths were `thread_sync` at 20.66%, `write_context_fence` at 14.59%,
and `drm_context_submit_cmd` at 11.14%.  This confirms that the capset test is
matching the DRM submit path and that the EGL work was removed as intended.

### vCPU scheduling A/B

The first post-fix runs varied from roughly 5000 to 6000 FPS even though the
EGL hotspot had disappeared.  This was not a regression in the capset logic.
The fence-driven workload alternates short guest execution bursts with waits;
without a utilization floor, the Android WALT governor kept the vCPU cluster
near its minimum frequency.

The following read-modify-restore A/B tests used the same running VM and the
same vkmark command.  Every temporary clamp and frequency setting was restored
after its run:

| Host scheduling variant | vkmark score |
| --- | ---: |
| Default after context-zero fix | 5130 |
| Lock CPU6/7 policy at 4.32 GHz | 9327 |
| `uclamp.min=1024` on GPU worker only | 6309 |
| `uclamp.min=1024` on all crosvm threads | 8395 |
| WALT latency-sensitive policy only | 5852 |
| `uclamp.min=1024` on vCPU threads only | 10113 |

The vCPU-only result identifies the relevant policy.  Boosting every crosvm
thread is worse because it creates avoidable CPU contention.  Crosvm already
implements the required per-vCPU `sched_setattr` behavior behind:

```text
--boost-uclamp
```

DroidVM passes each `extra_options` array entry directly to `crosvm run`, so
the VM configuration contains:

```json
"extra_options": [
  "--boost-uclamp"
]
```

After a normal VM restart, the crosvm command line contained the option and
`crosvm_vcpu0` reported `util_clamp min/max = 1024/1024`.  Three consecutive
runs scored 10249, 10818, and 10920 FPS (mean 10662).  Thermal status remained
0 and no GPU fault, IOMMU fault, device loss, or hang was observed.  The known
guest `CtxDetachResource: InvalidContextId` messages remain unrelated to this
change.

### Activity-scoped KGSL power constraint

The dynamic governor A/B showed that the GPU can remain near 222 MHz during
the shallow, fence-serialized headless workload even though the same build
reaches roughly 16K FPS with the GPU and CPU held at their maximum clocks.
Changing global `min_freq` is useful as a diagnostic but is not an acceptable
runtime policy.

The drm2kgsl backend therefore has an opt-in, per-KGSL-context activity boost:

```sh
NCTX_PWR_BOOST=1
NCTX_PWR_IDLE_MS=100
```

`NCTX_PWR_BOOST` is disabled when absent.  `NCTX_PWR_IDLE_MS` accepts 1 through
10000 and defaults to 100.  Its behavior is:

1. The first submit in an idle period sets `KGSL_PROP_PWR_CONSTRAINT` to
   `KGSL_CONSTRAINT_PWRLEVEL/KGSL_CONSTRAINT_PWR_MAX` for that KGSL draw
   context.  Boost-enabled draw contexts are created with
   `KGSL_CONTEXT_PWR_CONSTRAINT`, so a GMU-based driver also forwards a hint
   configured before the context's first firmware registration.
2. Submits during the active period carry `KGSL_CMDBATCH_PWR_CONSTRAINT` and
   only move the idle deadline forward.  They do not issue another property
   ioctl or wake the idle monitor once per frame.
3. When the deadline expires, the monitor reads the context's retired
   timestamp.  It does not release the hint while the most recently submitted
   timestamp is still executing.
4. Once both the idle interval and GPU retirement conditions hold, it sends
   `KGSL_CONSTRAINT_NONE`.  Submitqueue close and renderer destruction also
   release any remaining hint.

This covers both current Qualcomm driver paths.  GMU-based DCVS consumes the
context property as a firmware performance hint, while host-based DCVS applies
the command-batch flag to the submitted timestamp and keeps the kernel's own
retirement-based expiry protection.  Thermal and platform maximum limits still
win; this feature does not alter global sysfs min/max frequencies.

With `CROSVM_DRM2KGSL_DIAG=1`, the context-destroy report includes
`pwr_boost`, `pwr_release`, and `pwr_error`.  A healthy run should have zero
errors and, after the benchmark becomes idle, a release count matching the
number of activated submitqueues.  Measure this option as an A/B rather than
enabling it by default until temperature, interactive workloads, and idle
power have also been checked.

## Remaining fence optimizations

After the context-zero and scheduling fixes, the next candidates are:

### Diagnostics state

`CROSVM_DRM2KGSL_DIAG` now defaults to disabled.  It adds process-wide atomic
accounting on the submit, wait, and retire paths and should be enabled only for
diagnostic captures:

```sh
CROSVM_DRM2KGSL_DIAG=1
```

Diagnostic-on and diagnostic-off results must not be mixed in one comparison.

### 1. Read the retired timestamp once per merge pass

The opportunistic merge currently issues one
`READTIMESTAMP_CTXTID(KGSL_TIMESTAMP_RETIRED)` ioctl for each candidate fence.
The cumulative diagnostics from a long run were approximately:

```text
fence_queued=2388242
fence_callback=1444809
fence_merged=895330
wait=1469000
```

Many successful probes therefore read the same monotonically advancing retired
timestamp repeatedly.  A backend hook that returns the current retired
timestamp once would let the generic loop compare every following timestamp in
memory, including the normal signed wraparound comparison.  This can reduce
the `kgsl_timeline_poll_timestamp` cost without changing completion semantics.

### 2. Poll before enqueue

At fence creation, query whether the last submitted KGSL timestamp has already
retired.  If it has, invoke the normal retire callback synchronously and avoid
allocating a `drm_fence`, taking the timeline mutex, waking `kgsl-sync`, and
performing a later blocking wait.

This is compatible with current crosvm callback ordering: the callback records
the completed fence before `create_fence` returns, and descriptor processing
checks the completed-fence map before placing the descriptor on the pending
list.  It still requires an A/B because an unconditional extra ioctl can lose
when most timestamps are not yet retired.  A queue-depth or recent-completion
heuristic may be needed.

### 3. Revisit the guest wait path only after host fixed costs

Mesa's `tu_virtio_sync_wait_many()` takes an ioctl path when
`VK_SYNC_WAIT_PENDING` is present.  Historical measurements showed roughly one
such ioctl per frame at about 20.4 us/frame, but much of that duration is real
waiting for GPU submission rather than removable syscall overhead.  It remains
lower priority than removing redundant host completion work.

## Validation plan

Use the exact guest environment shown above, identical clocks, a warm-up, and
at least three runs per variant.  An A/B/B/A order helps expose thermal and
frequency drift.  Keep `--boost-uclamp` identical across compared variants;
otherwise WALT frequency selection can dominate the result.  Record:

- vkmark score and CPU cycles per submitted fence
- queued fences, waits, callbacks, merged fences, and maximum retire batch
- GPU worker, `kgsl-sync`, and vCPU event counts
- wait duration, timeout/error counts, and guest-visible failures
- submit-to-completion p50/p95/p99 latency when instrumentation is available

For a context-zero change, re-profile and confirm that
`vrend_renderer_force_ctx_0`, `eglMakeCurrent`, Android `getuid()`, and vendor
GLES symbols disappear from the DRM `SUBMIT_3D` hot path.  Also test a virgl2
workload and a real display/cursor path; a headless DRM benchmark alone cannot
establish compatibility for those paths.

For fence changes, success means fewer ioctls, callbacks, thread wakeups, or
CPU cycles without an early completion, GPU fault, hang, or unacceptable tail
latency regression.  Correct fence handling will not equal `NCTX_NO_FENCE`
because at least one real observation of GPU completion and the guest
completion path remain necessary.

## Current recommendation

The optimization order supported by the measurements is:

1. Keep `--boost-uclamp` enabled for this VM and do not boost all crosvm
   threads.
2. Retain the deployed context/capset-aware `force_ctx_0()` bypass, diagnostics
   default-off behavior, and opportunistic merge correctness fixes.
3. Replace per-candidate retired-timestamp ioctls with one read per merge pass.
4. Evaluate the poll-before-enqueue fast path.
5. Re-profile before changing guest `VK_SYNC_WAIT_PENDING` behavior.
6. A/B the opt-in activity-scoped KGSL power constraint against the unchanged
   dynamic-governor baseline; verify that it releases after the benchmark and
   does not change global frequency limits.

## References

- VirtIO 1.3 GPU command lifecycle and fence timeline requirements:
  https://docs.oasis-open.org/virtio/virtio/v1.3/csd01/virtio-v1.3-csd01.html
- KGSL UAPI, including retired timestamp query and waits:
  https://android.googlesource.com/kernel/msm/+/android-7.1.0_r0.2/include/uapi/linux/msm_kgsl.h
- Current Qualcomm KGSL driver, including GMU and host-based constraint paths:
  https://github.com/qualcomm-linux/kgsl
- Crosvm upstream source and history:
  https://chromium.googlesource.com/crosvm/crosvm/
- Virglrenderer Android source:
  https://android.googlesource.com/platform/external/virglrenderer/
