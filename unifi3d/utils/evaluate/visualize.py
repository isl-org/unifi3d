import os
import numpy as np
import open3d as o3d
import trimesh
import numpy as np
from PIL import Image
from unifi3d.utils.visualization_utils import PyrenderRenderer
import matplotlib.pyplot as plt


class SimpleMeshRender:
    def __init__(self, num_views=1, shape=(480, 480)):
        self.num_views = num_views
        angle = 2 * np.pi / num_views
        jaw_list = [np.pi / 6.0 + angle * (i + 1) for i in range(num_views)]

        render_config = {
            "cam_angle_yaw": jaw_list,
            "cam_angle_pitch": [np.pi / 6.0 for _ in range(self.num_views)],
            "cam_dist": [2.5 for _ in range(self.num_views)],
            "cam_height": [0 for _ in range(self.num_views)],
            "cam_angle_z": [0 for _ in range(self.num_views)],
            "yfov": np.pi / 6.0,
            "shape": shape,
        }
        self.renderer = PyrenderRenderer(render_config)

    def render_and_save_mesh(self, image_output_path, mesh):
        """
        Render image from mesh and save to file
        """
        if (not os.path.isdir(image_output_path)) and self.num_views > 1:
            raise RuntimeError(
                "Output path must be directory when multiple views should be rendered"
            )

        mesh = trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles)
        )

        rendered_images = self.renderer.render(meshes=[mesh])
        # need to add first dimension if only one image is rendered
        if rendered_images.ndim == 3:
            rendered_images = np.expand_dims(rendered_images, 0)

        for i, rendered in enumerate(rendered_images):
            if os.path.isdir(image_output_path):
                out_path = os.path.join(image_output_path, f"view_{i}.png")
            else:
                out_path = image_output_path
            img = Image.fromarray(rendered)
            img.save(out_path)


def save_individual_meshes(path: str, meshes: list):
    """
    Saves individual .obj files for each mesh in the list
    Args:
        path (str): path to save the .obj files, e.g. outputs/
        meshes: list of lists, each list contains two meshes (gt and pred) for a single object
    """
    labels = {1: "gt", 0: "pred"}
    if not os.path.exists(path):
        os.makedirs(path)
    for row, gt_and_pred_mesh in enumerate(meshes):
        # first row: pred, second row: gt
        for r, mesh in enumerate(gt_and_pred_mesh):
            o3d.io.write_triangle_mesh(f"{path}/mesh_{row}_{labels[r]}.obj", mesh)
    print("All meshes saved")


def save_combined_mesh_with_rows(path: str, meshes: list):
    """
    Saves a .obj file with all meshes in the list, each row is a different object
    Args:
        path: path to save the .obj file, e.g. outputs/test.obj
        meshes: list of lists, each list contains two meshes (gt and pred) for a single object
    """
    vertex_offset = 0
    translation_distance = 0.5

    if os.path.exists(path):
        os.remove(path)
    out_file = open(path, "w")

    # iterate over meshes
    for row, gt_and_pred_mesh in enumerate(meshes):
        # first row: pred, second row: gt
        for r, mesh in enumerate(gt_and_pred_mesh):
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)

            vertices[:, 0] += translation_distance * (r / 0.2 - 1)
            vertices[:, 2] += translation_distance * (row / 0.2 - 1)

            for vertex in vertices:
                out_file.write(
                    "v {0} {1} {2}\n".format(
                        str(vertex[0].round(6)),
                        str(vertex[1].round(6)),
                        str(vertex[2].round(6)),
                    )
                )

            for face in faces:
                item = face + vertex_offset + 1
                out_file.write("f {0} {1} {2}\n".format(item[0], item[1], item[2]))

            vertex_offset += len(vertices)

    out_file.close()


if __name__ == "__main__":
    # simple rendering of one mesh -> used to tune the pitch / yaw config
    mesh = o3d.io.read_triangle_mesh("mesh.obj")
    render = SimpleMeshRender()
    render.render_and_save_mesh("test.png", mesh)
