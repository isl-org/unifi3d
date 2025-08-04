from typing import Optional
import os
import sys
from pathlib import Path

sys.path.append(Path(__file__).parent.resolve().as_posix())
import argparse
import numpy as np
import tempfile


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script for converting a 3D asset to a numpy dict",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("input", type=Path, help="The input asset file path")
    parser.add_argument("output", type=Path, help="The output file path")
    parser.add_argument(
        "--target_size",
        type=float,
        default=0.9,
        help="""If > 0 the scene will be scaled and shifted such that the longest edge of the bounding box is the target_size and the center of the bounding box is in the origin""",
    )
    parser.add_argument(
        "--up_axis",
        type=str,
        default=None,
        choices=("X", "Y", "Z", "NEGATIVE_X", "NEGATIVE_Y", "NEGATIVE_Z"),
        help="Parameter directly passed to the import function in blender",
    )
    parser.add_argument(
        "--forward_axis",
        type=str,
        default=None,
        choices=("X", "Y", "Z", "NEGATIVE_X", "NEGATIVE_Y", "NEGATIVE_Z"),
        help="Parameter directly passed to the import function in blender",
    )
    parser.add_argument(
        "--blender",
        type=str,
        default="blender",
        help="blender command. script was tested with 4.1",
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if "--" in sys.argv:
        args, _ = parser.parse_known_args(sys.argv[sys.argv.index("--") + 1 :])
    else:
        args = parser.parse_args()
    print(args)
    return args


def convert_to_numpy(
    input: Path,
    output: Optional[Path] = None,
    target_size: float = 0.9,
    up_axis: Optional[str] = None,
    forward_axis: Optional[str] = None,
    blender: str = "blender",
):
    """Reads a 3D asset with Blender and converts it to a numpy representation

    To avoid any coordinate axis conversion for .obj files use
    up_axis='Z', forward_axis='Y'

    Args:
        input: The input asset file path
        output: The optional output file path to write the numpy dict as .npz
        target_size: If > 0 the scene will be scaled and shifted such that the
            longest edge of the bounding box is the target_size and the center
            of the bounding box is in the origin
        up_axis: Defines the axis that Blender should treat as the up-axis.
            Blender will make this the +Z axis.
        forward_axis: Defines the axis that Blender should treat as the forward-axis.
            Blender will make this the +Y axis.
        blender: Command to invoke blender
    Returns:
        Returns the numpy dict representation
    """
    assert up_axis in (None, "X", "Y", "Z", "NEGATIVE_X", "NEGATIVE_Y", "NEGATIVE_Z")
    assert forward_axis in (
        None,
        "X",
        "Y",
        "Z",
        "NEGATIVE_X",
        "NEGATIVE_Y",
        "NEGATIVE_Z",
    )
    import subprocess

    if output is None:
        with tempfile.TemporaryDirectory() as tmpdirname:
            output = Path(tmpdirname) / "scene.npz"
            return convert_to_numpy(
                input=input,
                output=output,
                target_size=target_size,
                up_axis=up_axis,
                forward_axis=forward_axis,
                blender=blender,
            )

    cmd = [blender, "-b", "-P", __file__]
    cmd.append("--")
    cmd += [str(input), str(output), "--target_size", str(target_size)]
    if up_axis is not None:
        cmd += ["--up_axis", up_axis]
    if forward_axis is not None:
        cmd += ["--forward_axis", forward_axis]

    print("==================================================================")
    print("Calling blender with")
    print(" ".join(cmd))
    print("==================================================================")
    status = subprocess.check_call(cmd, shell=False)
    print(status)
    scene = np.load(output)
    return scene


def main_python():
    args = parse_args()
    convert_to_numpy(**vars(args))


def main_blender():
    """This is the main function if the script is called with blender"""
    print("main_blender", flush=True)
    import bpy
    import blender_utils as bu

    bpy.context.preferences.view.show_splash = False

    if bpy.app.version < (3, 4, 1):
        raise Exception("This script requires at least Blender 3.4.1")

    args = parse_args()

    bu.delete_all_objects()

    mesh = bu.import_mesh(
        args.input, up_axis=args.up_axis, forward_axis=args.forward_axis
    )

    for obj in bpy.context.scene.objects.values():
        if obj.type == "MESH":
            if not bu.all_faces_are_triangles(obj):
                bu.triangulate_mesh(obj)

    if args.target_size > 0:
        bu.normalize_scene(target_size=args.target_size)

    for obj in bpy.context.scene.objects.values():
        bu.setup_vertex_color_shading(obj)

    # export the whole scene as numpy file
    scene_np = bu.scene_to_numpy(bpy.context.scene.objects.values())
    np.savez(args.output, **scene_np)


if __name__ == "__main__":
    try:
        import bpy

        is_blender = True
    except:
        is_blender = False

    if is_blender:
        main_blender()
    else:
        sys.exit(main_python())
