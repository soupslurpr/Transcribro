# Transcribro app icon

`transcribro-launcher.blend` is the editable source for the microphone relief,
materials, camera, and lighting. `foreground.png` is its transparent 1080 px
master. The separate background is indigo (`#34395E`). `monochrome.svg` and
`monochrome.xml` provide the themed silhouette and text cutouts.

From the repository root, with Blender 5.2 and Inkscape installed:

```sh
blender -b -t 4 --python-exit-code 1 \
  --python artwork/launcher-icon/render.py
```

The exporter uses OptiX when available and otherwise four CPU threads. It
updates the master, five Android foreground densities, monochrome vector,
background color, root `ic_launcher.svg`, and 512 px store PNG. It also writes
`artifact-hashes.json` for the sources and exported assets. The foreground
uses Android's 108-to-72 adaptive viewport ratio.

This original artwork and its source use the repository's MIT license.
