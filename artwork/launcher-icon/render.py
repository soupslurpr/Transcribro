"""Export the approved Blender scene into Transcribro's app identity assets.

The editable scene is the source for physical geometry and lighting. The
monochrome vector flattens the same silhouette without painted shading.
"""

import base64
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import bpy

DIRECTORY = Path(__file__).resolve().parent
PROJECT = DIRECTORY.parents[1]
RESOURCES = PROJECT / 'app/src/main/res'
SCENE = DIRECTORY / 'transcribro-launcher.blend'
BACKGROUND = '#34395E'
DENSITIES = (('mdpi', 108), ('hdpi', 162), ('xhdpi', 216),
             ('xxhdpi', 324), ('xxxhdpi', 432))


def main():
    bpy.ops.wm.open_mainfile(filepath=str(SCENE))
    scene = bpy.context.scene
    assert scene.camera.data.type == 'ORTHO'
    assert abs(scene.camera.data.ortho_scale - .15) < 1e-7
    assert scene.render.film_transparent
    assert scene.render.image_settings.color_mode == 'RGBA'
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = 4
    scene.cycles.samples = 1024
    preferences = bpy.context.preferences.addons['cycles'].preferences
    try:
        preferences.compute_device_type = 'OPTIX'
        preferences.get_devices()
        for device in preferences.devices:
            device.use = device.type == 'OPTIX'
        scene.cycles.device = 'GPU' if any(d.use for d in preferences.devices) else 'CPU'
    except (RuntimeError, TypeError):
        scene.cycles.device = 'CPU'

    exports = [(DIRECTORY / 'foreground.png', 1080)]
    exports += [(RESOURCES / f'drawable-{density}/ic_launcher_foreground.png', size)
                for density, size in DENSITIES]
    for target, size in exports:
        target.parent.mkdir(parents=True, exist_ok=True)
        scene.render.resolution_x = scene.render.resolution_y = size
        scene.render.filepath = str(target)
        bpy.ops.render.render(write_still=True)

    shutil.copyfile(DIRECTORY / 'monochrome.xml', RESOURCES / 'drawable/ic_launcher_monochrome.xml')
    (RESOURCES / 'values/ic_launcher_background.xml').write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<resources>\n'
        f'    <color name="ic_launcher_background">{BACKGROUND}</color>\n</resources>\n')
    encoded = base64.b64encode((DIRECTORY / 'foreground.png').read_bytes()).decode()
    standalone = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 100 100">'
        '<title>Transcribro</title><desc>Ivory text microphone with an apricot stand on indigo.</desc>'
        f'<rect width="100" height="100" fill="{BACKGROUND}"/>'
        f'<image x="-25" y="-25" width="150" height="150" href="data:image/png;base64,{encoded}"/>'
        '</svg>\n'
    )
    svg = PROJECT / 'ic_launcher.svg'
    svg.write_text(standalone)
    store_icon = PROJECT / 'app/src/main/ic_launcher-accrescent.png'
    subprocess.run(['inkscape', str(svg), f'--export-filename={store_icon}',
                    '--export-width=512', '--export-height=512'], check=True, capture_output=True)
    files = [target for target, _ in exports]
    files += [RESOURCES / 'drawable/ic_launcher_monochrome.xml',
              RESOURCES / 'values/ic_launcher_background.xml', svg, store_icon,
              SCENE, DIRECTORY / 'monochrome.svg', DIRECTORY / 'monochrome.xml', Path(__file__)]
    (DIRECTORY / 'artifact-hashes.json').write_text(json.dumps({
        str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }, indent=2) + '\n')
    print('Exported the approved icon, five Android densities, and matching identity assets.')


if __name__ == '__main__':
    main()
