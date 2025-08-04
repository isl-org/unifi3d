import json
import hydra
import numpy as np
import sys
import warnings
import os
import open3d as o3d
from omegaconf import DictConfig
import pandas as pd
import rootutils
import time
import torch
from unifi3d.utils.evaluate.visualize import (
    save_combined_mesh_with_rows,
    save_individual_meshes,
)
from unifi3d.utils.logger.memory_logger import MemoryLogger
from unifi3d.utils.data.mesh_utils import (
    scale_mesh_to_gt,
    load_npz_mesh,
    load_normalize_mesh,
)
from unifi3d.utils import rich_utils
from unifi3d.utils.model.model_utils import overwrite_cfg_from_ckpt
from unifi3d.utils.evaluate.visualize import SimpleMeshRender


# suppress info logs from o3d
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="benchmark")
def main(cfg: DictConfig):

    # use parameters from saved config.yaml if possible
    if cfg.get("overwrite_model_cfg", False):
        cfg = overwrite_cfg_from_ckpt(cfg)

    # apply extra utilities
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

    # Check if the user has provided a config name
    if not cfg:
        raise ValueError(
            "You must provide a configuration file with benchmark=<config_name>"
        )
    # create output dir
    os.makedirs(cfg.output_path, exist_ok=True)

    # initialize renderer and img dir
    render = SimpleMeshRender()
    out_path_images = os.path.join(cfg.output_path, "img_renders")
    os.makedirs(out_path_images, exist_ok=True)

    metrics_3d = {
        metric.name: (metric.requires_gt, hydra.utils.instantiate(metric.constructor))
        for metric in cfg.metrics
    }
    memory_logger = MemoryLogger("reconstruction")
    if cfg.net_encode:

        print(f"Found net_encode, instantiating model <{cfg.model._target_}>")
        autoencoder = hydra.utils.instantiate(cfg.net_encode).cuda()
        if hasattr(autoencoder, "is_transmitter") and autoencoder.is_transmitter:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            autoencoder.build(device)
        autoencoder.eval()
        if cfg.ckpt_path:
            autoencoder.load_checkpoint(cfg.ckpt_path)
        else:
            print("======================= Warning ==============================")
            print("No checkpoint path provided, using random weights.")
            print(
                "This option should only be true if we want to log the "
                + "memory only and not care about the reconstruction quality"
            )
            print("======================= Warning ==============================")

        memory_logger.log_memory("autoencoder")
        computed_model_size = autoencoder.total_size / (1024**2)

        memory_info = {
            "Allocated": memory_logger.logs["autoencoder"]["allocated_diff"],
            "Reserved": memory_logger.logs["autoencoder"]["reserved_diff"],
            "Computed model size": computed_model_size,
        }
        # Save memory info
        with open(
            os.path.join(cfg.output_path, "memory_info.json"), "w", encoding="latin-1"
        ) as f:
            json.dump(memory_info, f)

    print(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    data_loader = datamodule.test_dataloader()
    # create iterator
    data_loader_iter = iter(data_loader)
    memory_logger.log_memory("dataloader")

    if cfg["representation"] == "triplane":
        pass
    # Iterate over batches
    # list to collect generated samples and results
    all_meshes, results_log_list, mesh_counter, empty_counter = [], [], 0, 0
    for batch_idx in range(len(data_loader_iter)):
        # 1) create batch: load mesh and transform into the representation
        start_time = time.time()
        batch = next(data_loader_iter)
        time_convert = time.time() - start_time
        bs = len(batch["mesh_path"])

        if cfg.net_encode:
            # 2) encode batch: transform representation into latent space
            start_time = time.time()
            with torch.no_grad():
                encoded = autoencoder.encode_wrapper(batch)
                memory_logger.log_memory("autoencoder encoder")

            time_encode = time.time() - start_time
            size_encode = encoded.numel()

            # 3) decode batch: transform latent space into representation
            start_time = time.time()
            with torch.no_grad():
                decoded = autoencoder.decode_wrapper(encoded)
            memory_logger.log_memory("autoencoder decoder")

            time_decode = time.time() - start_time
        else:
            time_encode, time_decode, size_encode = (pd.NA, pd.NA, pd.NA)
            # helper function to get the representation from the batch
            decoded = datamodule.data_test.get_representation_from_batch(batch)

        memory_logger.print_logs()

        # evaluate per mesh in batch
        for i, fp in enumerate(batch["mesh_path"]):
            print(f"{batch_idx}: batch_idx, {i}: i, file path: {fp}")
            # 4) convert back to mesh
            start_time = time.time()
            if isinstance(decoded, dict):
                # TODO: Dualoctree evaluates one tree at a time. Batch dummy full
                # dualoctree generation is not impelemented by the original authors.
                # Please more checks if others are also returning dictionary.
                mesh_pred = datamodule.data_test.representation_to_mesh(decoded)
            else:
                if (
                    hasattr(autoencoder, "is_transmitter")
                    and autoencoder.is_transmitter
                ):
                    mesh_pred = autoencoder.representation_to_mesh(decoded[i])
                else:
                    mesh_pred = datamodule.data_test.representation_to_mesh(decoded[i])

            time_reconstruct = time.time() - start_time
            if decoded is None:
                # Triplane: correct runtime to part of the reconstrution time
                time_decode = (time_reconstruct * bs) - time_encode

            # Logging and evaluation
            results_log_sample = {
                "batch_id": batch_idx,
                "file": fp,
                "time_convert": time_convert / bs,
                "time_encode": time_encode / bs,
                "time_decode": time_decode / bs,
                "time_reconstruct": time_reconstruct,
                "size_encoded": size_encode / bs,
            }

            # for empty meshes, cannot compute metrics -> raise Error
            if len(mesh_pred.vertices) == 0 or len(mesh_pred.triangles) == 0:
                if empty_counter > 10:
                    raise RuntimeError("Too many empty meshes, stopping")
                warnings.warn(
                    f"Empty mesh for {fp}. Likely, reconstruction didn't work"
                )
                empty_counter += 1
                print("Empty mesh")
                continue

            # load real mesh: use preprocessed npz file if avail
            if "mesh_path" in batch.keys():
                mesh_real = load_npz_mesh(batch["mesh_path"][i])
            else:
                mesh_real = load_normalize_mesh(batch["file_path"][i], cfg.mesh_scale)

            if np.min(mesh_pred.vertices) < -0.5 or np.max(mesh_pred.vertices) > 0.5:
                warnings.warn(
                    "GT mesh is scaled to box [-0.4, 0.4], but pred mesh is larger"
                )
            print("Loaded mesh real")
            print(mesh_real)
            # optional: scale reconstructed mesh to size of real mesh
            if cfg.align_scale:
                print("align scale.")
                mesh_pred = scale_mesh_to_gt(mesh_pred, mesh_real)

            if cfg.plot_mesh and len(all_meshes) < 10:
                print("save mesh.")
                mesh_counter += 1
                all_meshes.append([mesh_pred, mesh_real])
                # render image and save to file
                out_gt = os.path.join(out_path_images, f"img_gt_{mesh_counter}.png")
                out_pred = os.path.join(out_path_images, f"img_pred_{mesh_counter}.png")
                render.render_and_save_mesh(out_gt, mesh_real)
                render.render_and_save_mesh(out_pred, mesh_pred)

            mesh_pred = o3d.t.geometry.TriangleMesh.from_legacy(mesh_pred)
            mesh_real = o3d.t.geometry.TriangleMesh.from_legacy(mesh_real)
            # calculate metrics
            print("Calculating metrics")
            for metric_name, (requires_gt, metric_obj) in metrics_3d.items():
                # check if we need to input both meshes or only pred
                try:
                    if requires_gt:
                        metric_res = metric_obj(mesh_pred, mesh_real)
                    else:
                        metric_res = metric_obj(mesh_pred)
                except Exception as e:
                    # metrics sometimes fail -> set to NA instead of throwing error
                    # to avoid loosing all results collected so far.
                    warnings.warn(f"Error {e} in metric {metric_name}, sample {fp}")
                    metric_res = pd.NA
                # average if output is a list
                if isinstance(metric_res, np.ndarray):
                    metric_res = np.mean(metric_res)
                results_log_sample[metric_name] = metric_res

            results_log_list.append(results_log_sample)

        del decoded  # to save space (necessary in 3dshape2vec)

        # stop if limit reached
        if cfg.limit > 0 and len(results_log_list) >= cfg.limit:
            break
    if cfg.plot_mesh:
        print("Saving mesh")
        # # uncomment to save outputs in individual files
        # save_individual_meshes("outputs/test_folder", all_random_samples)
        save_combined_mesh_with_rows(
            os.path.join(cfg.output_path, "mesh_comp.obj"), all_meshes
        )
        print("Saved mesh plot as", os.path.join(cfg.output_path, "mesh_comp.obj"))

    results = pd.DataFrame(results_log_list)
    results.to_csv(os.path.join(cfg.output_path, "results.csv"))

    # print(results.head(10).drop("file", axis=1).to_markdown())
    numerical_df = results.select_dtypes(include="number")
    averages = numerical_df.mean()
    print(averages)


if __name__ == "__main__":
    main()
