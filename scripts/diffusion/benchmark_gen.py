"""
Sample new images from a pre-trained DiT.
"""

import os
import pandas as pd
import shutil
import time
import numpy as np
from omegaconf import DictConfig
from typing import Optional
import torch
import hydra
import warnings
import open3d as o3d
from PIL import Image
from safetensors.torch import load_file
import rootutils
from unifi3d.utils.evaluate.unconditional_gen import evaluate_unconditional_generation
from unifi3d.utils import rich_utils
from unifi3d.utils.evaluate.visualize import save_combined_mesh_with_rows
from unifi3d.utils.data.mesh_utils import load_generated_dataset
from unifi3d.utils.logger.memory_logger import MemoryLogger
from unifi3d.utils.model.model_utils import overwrite_cfg_from_ckpt
from unifi3d.utils.evaluate.visualize import SimpleMeshRender


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def sample_diffusion(model, dataset, cfg, info_to_save={}):
    """Sample from diffusion model and evaluate results"""
    model.eval()
    # intiiate metrics
    metrics_3d = {
        metric.name: hydra.utils.instantiate(metric.constructor)
        for metric in cfg.metrics
    }
    metrics_types = {
        metric.name: metric.get("cond_type", None) for metric in cfg.metrics
    }
    # initialize renderer
    num_views_render = cfg.get("num_views_render", 1)
    render = SimpleMeshRender(num_views=num_views_render, shape=(256, 256))
    # make out dir
    os.makedirs(cfg.output_path, exist_ok=True)
    if cfg.plot_mesh and not cfg.save_mesh:
        warnings.warn("Cannot plot meshes because not saved, set cfg.save_mesh=True")
    out_path_meshes = os.path.join(cfg.output_path, "generated_samples")
    out_path_images = os.path.join(cfg.output_path, "img_renders")
    if cfg.save_mesh:
        if os.path.exists(out_path_meshes):
            warnings.warn(f"Output path {out_path_meshes} already exists, overwriting")
        os.makedirs(out_path_meshes, exist_ok=True)
        os.makedirs(out_path_images, exist_ok=True)

    shape = tuple(cfg.shape)
    num_samples_to_generate = cfg.num_samples
    bs = cfg.get("batch_size_generation", 1)

    # get context for conditional generation
    context = {}
    if "context" in cfg:
        # context in config is a list of dictionaries
        for cond_dict in cfg.context:
            context.update(cond_dict)
        context = {k: list(v) for k, v in context.items()}  # cast ListConfigs to lists
        # make sure that the lengths are the same, e.g. as many images as texts
        cond_lens = [len(context[cond_type]) for cond_type in context]
        assert len(np.unique(cond_lens)) == 1, "Context lengths are not the same"
        bs = cond_lens[0]  # need to adjust bs to context size

    results_log_list = []
    num_batches = num_samples_to_generate // bs

    # Generate samples in batches
    # create noise, denoise, and decode with vqvae
    mesh_counter, empty_mesh_counter = 0, 0
    for _ in range(num_batches):
        # call inference function of diffusion trainer
        tic = time.time()
        model_output = model._inference(
            size=(bs, *shape),
            num_sampling_steps=cfg.num_sampling_steps,
            context=context,
        )
        time_gen = (time.time() - tic) / bs
        decoded = model_output["reconstructed"]
        if isinstance(decoded, dict):
            num_samples = 1
        elif decoded is None:
            # this is triplane
            num_samples = bs
        else:
            num_samples = len(decoded)
        for i in range(num_samples):

            if isinstance(decoded, dict):
                dec = decoded
            else:
                dec = decoded[i]
            # reconstruct mesh
            if (
                hasattr(model.encoder_decoder, "is_transmitter")
                and model.encoder_decoder.is_transmitter
            ):
                reconstructed = model.encoder_decoder.representation_to_mesh(
                    dec
                ).remove_unreferenced_vertices()
            else:
                reconstructed = dataset.representation_to_mesh(
                    dec
                ).remove_unreferenced_vertices()

            # for empty meshes, cannot compute metrics -> raise Error
            if len(reconstructed.vertices) == 0 or len(reconstructed.triangles) == 0:
                warnings.warn("Empty mesh generated. Cannot compute metrics")
                empty_mesh_counter += 1
                if empty_mesh_counter > 5:
                    raise ValueError("More than 5 empty meshes generated. Stopping.")
                continue

            # save mesh to obj file
            if cfg.save_mesh:
                o3d.io.write_triangle_mesh(
                    os.path.join(out_path_meshes, f"mesh_{mesh_counter}.obj"),
                    reconstructed,
                )
                print(f"Saved mesh to: {out_path_meshes}")

            render_path = None
            if cfg.render_mesh:
                # render image and save to file
                if num_views_render == 1:
                    render_path = os.path.join(
                        out_path_images, f"img_{mesh_counter}.png"
                    )
                else:
                    render_path = os.path.join(
                        out_path_images, f"render_{mesh_counter}"
                    )
                    os.makedirs(render_path, exist_ok=True)
                render.render_and_save_mesh(render_path, reconstructed)

            results_sample = {"sample_id": mesh_counter, "time_gen": time_gen}

            # compute metrics that can be computed for mesh without gt
            mesh = o3d.t.geometry.TriangleMesh.from_legacy(reconstructed)
            for metric_name, metric_func in metrics_3d.items():
                # check whether conditional gen metric
                is_conditional_metric = metrics_types[metric_name] is not None
                # compute metric
                if is_conditional_metric:
                    metric_cond_type = metrics_types[metric_name]  # image or text
                    # call function with condition type and render path
                    if metric_cond_type == "text":
                        prompt = context[metric_cond_type][i]
                    elif metric_cond_type == "image":
                        prompt = Image.open(context[metric_cond_type][i])
                    else:
                        raise NotImplementedError
                    metric_res = metric_func(prompt, render_path)
                else:
                    metric_res = metric_func(mesh)
                # average if output is a list
                if isinstance(metric_res, np.ndarray):
                    metric_res = np.mean(metric_res)
                results_sample[metric_name] = metric_res

            results_log_list.append(results_sample)

            # Remove renders that were only used for evaluation
            if render_path is not None and i > 10:
                shutil.rmtree(render_path, ignore_errors=True)

            mesh_counter += 1

        # free up space
        del decoded

    if cfg.plot_mesh and cfg.save_mesh:
        # Load generated meshes and combine 10 of them in one file for visualization
        meshes_to_plot = load_generated_dataset(
            out_path_meshes, limit=10, from_legacy=False
        )
        # scale the meshes to make them better visible
        meshes_to_plot = [[mesh.scale(3, mesh.get_center())] for mesh in meshes_to_plot]
        save_combined_mesh_with_rows(
            os.path.join(cfg.output_path, "mesh_plot.obj"), meshes_to_plot
        )
        print(
            "Plotted some meshes in one file, see",
            os.path.join(cfg.output_path, "mesh_plot.obj"),
        )

    # save mesh metrics
    results = pd.DataFrame(results_log_list)
    results.to_csv(os.path.join(cfg.output_path, "results.csv"), index=False)
    # print(results.head(10).to_markdown())

    # Compute unconditional generation metrics
    if cfg.get("compute_metrics", True):
        print("Computing Cov, MMD and 1-NNA metrics ...")
        evaluate_unconditional_generation(cfg, out_path_meshes, info_to_save)


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="sample_diffusion.yaml",
)
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # use parameters from saved config.yaml if possible
    if cfg.get("overwrite_model_cfg", False):
        cfg = overwrite_cfg_from_ckpt(cfg)

    if "categories" in cfg:
        cfg.reference_dataset.categories = cfg.categories
    elif "cat_id" in cfg:
        cfg.reference_dataset.categories = [cfg.cat_id.lower()]
    else:
        raise ValueError(
            "No categories provided in config. Need category for reference dataset"
        )

    # print config tree after potentially overwriting:
    rich_utils.print_config_tree(
        cfg,
        print_order=(
            "sample",
            "data",
            "model",
        ),
        resolve=True,
        is_main_process=True,
    )

    memory_logger = MemoryLogger("diffusion")

    # Load dataset
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    dataset = datamodule.data_train
    memory_logger.log_memory("dataset")

    # load diffusion model
    diffusion_model = hydra.utils.instantiate(cfg.model)
    if cfg.ckpt:
        path_to_ckpt = (
            os.path.join(cfg.ckpt, "model.safetensors")
            if "safetensors" not in cfg.ckpt
            else cfg.ckpt
        )
        diff_model_statedict = load_file(path_to_ckpt)
        # strict=False necessary due to changes in huggingface CLIP model
        diffusion_model.load_state_dict(diff_model_statedict, strict=False)
    else:
        print("======================= Warning ==============================")
        print("No checkpoint path provided, using random weights.")
        print(
            "This option should only be true if we want to log the "
            + "memory only and not care about the generation quality"
        )
        print("======================= Warning ==============================")
    diffusion_model.cuda()

    memory_logger.log_memory("diffusion model")
    memory_logger.print_logs()

    sample_diffusion(diffusion_model, dataset, cfg, info_to_save=memory_logger.logs)


if __name__ == "__main__":
    main()
