# Licensing

This repository holds two kinds of material and they are licensed differently.

## Material inherited from upstream

- **virglrenderer** (MIT) — https://gitlab.freedesktop.org/virgl/virglrenderer

Every file that came from an upstream project stays under that project's
license. Nothing here relicenses it, and modifications to those files do not
relicense them either — a patched upstream file is still an upstream file.

## Material written for DroidVM

Files carrying `SPDX-License-Identifier: GPL-3.0-or-later` are DroidVM work
and are licensed under the GNU GPL, version 3 or later, **with the
additional permissions in `ADDITIONAL-PERMISSIONS`**.

Those permissions exist so this work can go upstream. They let anyone
relicense it under the terms an upstream project requires, for the purpose of
getting it merged there — and only for that purpose. Once upstream publishes
it, upstream's license governs that copy.

## Third-party material that is neither

`src/drm/drm2kgsl/msm_kgsl.h` is Qualcomm's KGSL kernel UAPI header, copied in
verbatim. It keeps its own terms and is not DroidVM work.

`src/drm/drm2kgsl/drm2kgsl_renderer.{c,h}` are **MIT, not GPL**, and that is
deliberate. They began as the KGSL analogue of virglrenderer's own
`msm_renderer.c` and carry Google's copyright alongside DroidVM's, so DroidVM
cannot relicense them alone. MIT is also what virglrenderer requires of
contributions — which is what these files exist to become. Nothing is lost:
MIT already permits everything the additional permissions would have granted.

## Contributing

See `CONTRIBUTING.md`. Sign-off is required; there is no CLA.
