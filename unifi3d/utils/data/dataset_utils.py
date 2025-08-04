import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from functools import partial


def custom_collate(data, pad_id: int = -1):
    """
    Collates a list of data samples into a batch.
    Args:
        pad_id (int): Padding for combining sequences of different length
    """
    is_dict = isinstance(data[0], dict)

    if is_dict:
        keys = sorted(data[0].keys())
        data = [[d[k] for k in keys] for d in data]

    output = []

    for datum in zip(*data):
        if isinstance(datum[0], torch.Tensor):
            datum = pad_sequence(datum, batch_first=True, padding_value=pad_id)
        else:
            datum = list(datum)

        output.append(datum)

    output = tuple(output)

    if is_dict:
        output = dict(zip(keys, output))

    return output


def sdf_sphere(point, radius=0.25):
    """
    Returns the signed distance value of a point against a sphere of radius 0.25 centered at (0,0,0)
    """
    return np.linalg.norm(point) - radius


def sdf_cube(point, side=0.25):
    """
    Returns the signed distance value of a point against a cube of side 0.25 centered at (0,0,0)
    """
    # Convert the point to its absolute coordinates (cube is symmetric)
    p = np.abs(point)

    # Distance from the point to the cube's faces, find the minimum dimension
    d = p - side / 2.0

    # Signed distance calculation
    # Distance outside of the cube
    outside_distance = np.linalg.norm(np.maximum(d, 0))

    # Distance inside of the cube
    inside_distance = np.minimum(np.maximum(d[0], np.maximum(d[1], d[2])), 0)
    return outside_distance + inside_distance


def sdf_cylinder(point, radius=0.25, height=0.5):
    """
    Returns the signed distance value of a point against a cylinder of radius 0.25, height 0.5 centered at (0,0,0)
    """
    # Compute the distance to the infinite cylinder (projected onto the XY plane)
    x, y, z = point
    d_xy = np.sqrt(x**2 + y**2) - radius
    # Compute the distance to the caps (top and bottom planes)
    d_z = abs(z) - height / 2
    if d_xy <= 0 and d_z <= 0:
        return np.maximum(d_xy, d_z)
    elif d_xy > 0 and d_z > 0:
        return np.sqrt(d_xy**2 + d_z**2)
    elif d_xy > 0:
        return d_xy
    else:
        return d_z


def sdf_cone(point, radius=0.25, height=0.5):
    """
    Returns the signed distance value of a point against a cone of radius 0.25, height 0.5 centered at (0,0,0)
    """
    x, y, z = point

    # Define the tip and base position
    tip_z = height / 2.0
    base_z = -height / 2.0

    # Cone's tip is at (0, 0, height / 2) and the base is at z = -height/2
    k = radius / height  # slope of the cone's side

    # radius on the x-y plane
    d_to_axis = np.sqrt(x**2.0 + y**2.0)
    vertical_dist = z - base_z

    # Check if the point is outside or inside the cone
    if z > height / 2.0:  # Above the tip
        return np.linalg.norm([d_to_axis, z - tip_z])
    elif z < -height / 2:  # Below the base
        base_dist = d_to_axis - radius
        if base_dist > 0:
            return np.sqrt(base_dist**2 + vertical_dist**2.0)
        else:
            return abs(vertical_dist)
    else:  # in between tip and base
        dz_to_tip = tip_z - z
        projected_radius = k * dz_to_tip  # local radius at z
        d_local_r_to_tip = np.sqrt(projected_radius**2 + dz_to_tip**2.0)

        d_to_surface = dz_to_tip / d_local_r_to_tip * (d_to_axis - projected_radius)
        if d_to_axis <= projected_radius and vertical_dist < np.abs(d_to_surface):
            return -vertical_dist
        else:
            return d_to_surface


def generate_sample_points(num_points=1000, range_min=-0.5, range_max=0.5, dim=3):
    """
    Generate uniformly sampled points of dimension $dim with range between range_min and range_max.
    """
    return np.random.uniform(range_min, range_max, size=(num_points, dim))


def get_near_fraction(abs_sdf, threshold_near):
    near_point_occupancy = abs_sdf < threshold_near
    occ_sum = near_point_occupancy.sum()
    fraction = occ_sum / len(near_point_occupancy)
    return fraction


def update_data(query_data, uniform_sample_data, sample_inds, start_ind, end_ind):
    uniform_sampled_points = uniform_sample_data["points"]
    uniform_sdf = uniform_sample_data["sdf"]
    uniform_occupancy = uniform_sample_data["occupancy"]

    new_near_query_points = uniform_sampled_points[sample_inds]
    end_ind = min(len(query_data["points"]), end_ind)
    num_new = end_ind - start_ind
    max_ind = len(query_data["points"])
    query_data["points"][start_ind : min(end_ind, max_ind)] = new_near_query_points[
        :num_new
    ]
    new_near_sdf = uniform_sdf[sample_inds]
    query_data["sdf"][start_ind : min(end_ind, max_ind)] = new_near_sdf[:num_new]
    new_near_query_occupancy = uniform_occupancy[sample_inds]
    query_data["occupancy"][start_ind : min(end_ind, max_ind)] = (
        new_near_query_occupancy[:num_new]
    )
    return query_data


def generate_near_far_sample_points(
    sdf_fn,
    num_points=1000,
    range_min=-0.5,
    range_max=0.5,
    dim=3,
    fraction_near=0.2,
    threshold_near=0.01,
):
    """
    Near far sampler of the geometry
    """
    print("Generate near far samples...")

    assert (
        threshold_near < range_max
    ), f"threshold_near: {threshold_near} cannot be larger than half of the smapling cube {range_max}."

    num_near = 0
    num_far = 0
    target_near = int(num_points * fraction_near)
    target_far = num_points - target_near
    query_near_data = {
        "points": np.empty((target_near, 3)),
        "sdf": np.empty((target_near,)),
        "occupancy": np.empty((target_near,)),
    }
    query_far_data = {
        "points": np.empty((target_far, 3)),
        "sdf": np.empty((target_far,)),
        "occupancy": np.empty((target_far,)),
    }
    print(f"Need {target_near} near points, {target_far} far points.")

    while num_near < target_near or num_far < target_far:
        uniform_sampled_points = generate_sample_points(
            num_points * 10, range_min=range_min, range_max=range_max, dim=dim
        )
        uniform_sdf, uniform_occupancy = compute_occupancy_and_sdf(
            uniform_sampled_points, sdf_fn
        )
        uniform_sample_data = {
            "points": uniform_sampled_points,
            "sdf": uniform_sdf,
            "occupancy": uniform_occupancy,
        }

        abs_sdf = np.abs(uniform_sdf)
        fraction = get_near_fraction(abs_sdf, threshold_near)
        near_inds = np.where(abs_sdf < threshold_near)[0]
        far_inds = np.where(abs_sdf >= threshold_near)[0]

        required_near = max(target_near - num_near, 0)
        required_far = max(target_far - num_far, 0)

        if fraction <= fraction_near:
            near_inds_sample = near_inds[:required_near]
            far_inds_sample = np.random.choice(far_inds, required_far, replace=False)
        else:
            near_inds_sample = np.random.choice(near_inds, required_near, replace=False)
            far_inds_sample = far_inds[:required_far]

        start_near_ind = num_near
        start_far_ind = num_far
        num_near += len(near_inds_sample)
        num_far += len(far_inds_sample)

        if len(near_inds_sample) > 0:
            query_near_data = update_data(
                query_data=query_near_data,
                uniform_sample_data=uniform_sample_data,
                sample_inds=near_inds_sample,
                start_ind=start_near_ind,
                end_ind=num_near,
            )
        if len(far_inds_sample) > 0:
            query_far_data = update_data(
                query_data=query_far_data,
                uniform_sample_data=uniform_sample_data,
                sample_inds=far_inds_sample,
                start_ind=start_far_ind,
                end_ind=num_far,
            )

        # Update collected num_near, and collected_num_far

        new_num_near = max(target_near - num_near, 0)
        new_num_far = max(target_far - num_far, 0)
        new_total = new_num_near + new_num_far
        print(f"Sampled {num_near} near points, {num_far} far points.")
        if new_total > 0:
            fraction_near = new_num_near / new_total
            print(f"fraction near is updated to {fraction_near}")
        else:
            break
    query_points = np.concatenate(
        [
            query_near_data["points"][:target_near],
            query_far_data["points"][:target_far],
        ]
    )
    query_sdf = np.concatenate(
        [query_near_data["sdf"][:target_near], query_far_data["sdf"][:target_far]]
    )
    query_occupancy = np.concatenate(
        [
            query_near_data["occupancy"][:target_near],
            query_far_data["occupancy"][:target_far],
        ]
    )

    return query_points, query_sdf, query_occupancy


def compute_occupancy_and_sdf(points, sdf_func, threshold=0.01):
    """
    Return the occupancy and sdf of points based on the sdf_func per point.
    """
    sdf_values = np.apply_along_axis(sdf_func, 1, points)
    occupancy = (sdf_values < threshold).astype(
        float
    )  # 1 for on surface and within, 0 otherwise
    return sdf_values, occupancy


def sample_surface_sphere(num_points, radius=0.25):
    """
    Sample $num_points of surface point of a sphere of given radius centered at (0,0,0)
    """
    phi = np.random.uniform(0, 2 * np.pi, num_points)  # azimuthal angle
    theta = np.arccos(np.random.uniform(-1, 1, num_points))  # Polar angle
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return np.vstack((x, y, z)).T


def sample_surface_cube(num_points, side_length=0.5):
    """
    Sample $num_points of surface point of a cube of a given side length centered at (0,0,0)
    """
    face_indices = np.random.choice(6, num_points)
    half_edge = side_length / 2.0
    random_coords = np.random.uniform(-half_edge, half_edge, (num_points, 2))

    surface_points = np.zeros((num_points, 3))

    for i in range(6):
        mask = face_indices == i
        if i == 0:  # +x face
            surface_points[mask] = np.column_stack(
                (np.full(mask.sum(), half_edge), random_coords[mask])
            )
        elif i == 1:  # -x face
            surface_points[mask] = np.column_stack(
                (np.full(mask.sum(), -half_edge), random_coords[mask])
            )
        elif i == 2:  # +y face
            surface_points[mask] = np.column_stack(
                (
                    random_coords[mask][:, 0],
                    np.full(mask.sum(), half_edge),
                    random_coords[mask][:, 1],
                )
            )
        elif i == 3:  # -y face
            surface_points[mask] = np.column_stack(
                (
                    random_coords[mask][:, 0],
                    np.full(mask.sum(), -half_edge),
                    random_coords[mask][:, 1],
                )
            )
        elif i == 4:  # +z face
            surface_points[mask] = np.column_stack(
                (random_coords[mask], np.full(mask.sum(), half_edge))
            )
        elif i == 5:  # -z face
            surface_points[mask] = np.column_stack(
                (random_coords[mask], np.full(mask.sum(), -half_edge))
            )
        else:
            assert False
    return surface_points


def sample_surface_cone(num_points, height=0.5, radius=0.25):
    """
    Sample $num_points of surface point of a cone of a given height and radius centered at (0,0,0)
    """
    points = []
    for _ in range(num_points):
        phi = np.random.uniform(0, 2 * np.pi)
        if np.random.rand() < 0.5:
            # Sample points on the slanted surface
            z = np.random.uniform(
                -0.5 * height, 0.5 * height
            )  # uniform sampling along the height range
            r = (0.5 * height - z) * (radius / height)  # radius decreases linearly.
            x = r * np.cos(phi)
            y = r * np.sin(phi)
        else:
            # Sample points on the circular base
            z = -0.5 * height  # base is at z = -.5 * height
            r = (
                np.sqrt(np.random.uniform(0, 1)) * radius
            )  # Uniform sampling on the disk
            x = r * np.cos(phi)
            y = r * np.sin(phi)
        points.append([x, y, z])

    return np.array(points)


def sample_surface_cylinder(num_points, height=0.5, radius=0.25):
    """
    Sample $num_points of surface point of a cylinder of a given height and radius centered at (0,0,0)
    """

    surface_types = np.random.choice(["side", "top", "bottom"], size=num_points)

    x = np.zeros(num_points)
    y = np.zeros(num_points)
    z = np.zeros(num_points)

    # side surface
    mask_side = surface_types == "side"
    num_side = np.sum(mask_side)
    phi_side = np.random.uniform(0, 2 * np.pi, num_side)
    z[mask_side] = np.random.uniform(-height / 2, height / 2, num_side)
    x[mask_side] = radius * np.cos(phi_side)
    y[mask_side] = radius * np.sin(phi_side)

    # Top cap
    mask_top = surface_types == "top"
    num_top = np.sum(mask_top)
    phi_top = np.random.uniform(0, 2 * np.pi, num_top)
    r_top = np.sqrt(np.random.uniform(0, radius**2, num_top))
    x[mask_top] = r_top * np.cos(phi_top)
    y[mask_top] = r_top * np.sin(phi_top)
    z[mask_top] = height / 2

    # Bottom cap
    mask_bottom = surface_types == "bottom"
    num_bottom = np.sum(mask_bottom)
    phi_bottom = np.random.uniform(0, 2 * np.pi, num_bottom)
    r_bottom = np.sqrt(np.random.uniform(0, radius**2, num_bottom))
    z[mask_bottom] = -height / 2
    x[mask_bottom] = r_bottom * np.cos(phi_bottom)
    y[mask_bottom] = r_bottom * np.sin(phi_bottom)
    sampled_points = np.column_stack((x, y, z))
    return sampled_points


def sample_points_from_simple_geometries(
    num_sample_points=10000,
    num_surface_points=10000,
    pt_range=0.5,
    fraction_near=0.5,
    threshold_near=0.01,
    shape_names=[],
    mode="train",
):
    """
    Sample surface points and random query points from four known shapes: sphere, cube, cylinder and cone.

    Args:
        num_sample_points: Number of points sampled from 3D dimension of range -pt_range to pt_range.
        num_surface_points: Number of points sampled from the surface of shapes specified in shape_names.
        pt_range: range of points on each direction of the dimension. For query points, it samples uniformly
            from -pt_range to pt_range. For known shapes, it defines the height as 3/4 pt_range, and the
            radius as pt_range/2.
        fraction_near: when mode is train, near_far sampler is used, where the fraction_near indicates how
            much percentage of the sampled points are close the surface as indicated by threshold_near thresholding
            on absolute sdf values.
        threshold_near: defines what counts as near points via measuring how close a point is to the surface
        shape_names: A list of text, with names chosen from ["sphere", "cube", "cylinder", "cone"]
        mode: "train", "test", "val"
    Returns:
        A list of dictionaries, where each item has keys of ["surface_points", "sample_points", "sample_sdf",
        "sample_occupancy", and "label"]
            - surface_points: numpy array of float in the shape of [num_surface_points, 3]
            - sample_points: numpy array of float in the shape of [num_sample_points, 3]
            - sample_sdf: numpy array of float in the shape of [num_sample_points,]
            - sample_occupancy: numpy array of float in the shape of [num_sample_points, 3]
            - label: text with content as "sphere" or "cube" or "cylinder" or "cone".
    """
    R = pt_range
    val_scale = 1.0  # Different scale has not been enabled yet.
    test_scale = 1.0
    if mode == "val":
        R = R * val_scale
    elif mode == "test":
        R = R * test_scale
    else:
        assert (
            mode == "train"
        ), f"mode needs to be chosen from train, test, or val, {mode} is not recognized!"
    sampled_data = []

    side = R * 3.0 / 4.0
    radius = R / 2.0
    shapes = {
        "sphere": {
            "sdf_fn": partial(sdf_sphere, radius=radius),
            "surface_fn": partial(sample_surface_sphere, radius=radius),
        },
        "cube": {
            "sdf_fn": partial(sdf_cube, side=side),
            "surface_fn": partial(sample_surface_cube, side_length=side),
        },
        "cylinder": {
            "sdf_fn": partial(sdf_cylinder, height=side, radius=radius),
            "surface_fn": partial(sample_surface_cylinder, height=side, radius=radius),
        },
        "cone": {
            "sdf_fn": partial(sdf_cone, radius=radius, height=side),
            "surface_fn": partial(sample_surface_cone, height=side, radius=radius),
        },
    }
    if len(shape_names) == 0:
        shape_names = shapes.keys()
    else:
        for shape_name in shape_names:
            assert (
                shape_name in shapes
            ), f"{shape_name} is not supported for function sampling."

    for shape_name in shape_names:
        sdf_func = shapes[shape_name]["sdf_fn"]
        surface_fnc = shapes[shape_name]["surface_fn"]
        surface_points = surface_fnc(num_surface_points)

        if mode == "train":
            sample_points, sample_sdf, sample_occupancy = (
                generate_near_far_sample_points(
                    sdf_fn=sdf_func,
                    num_points=num_sample_points,
                    range_min=-R,
                    range_max=R,
                    dim=3,
                    fraction_near=fraction_near,
                    threshold_near=threshold_near,
                )
            )
        else:
            sample_points = generate_sample_points(num_sample_points, -R, R)

            sample_sdf, sample_occupancy = compute_occupancy_and_sdf(
                sample_points, sdf_func
            )
        sampled_data.append(
            {
                "surface_points": surface_points,
                "sample_points": sample_points,
                "sample_sdf": sample_sdf,
                "sample_occupancy": sample_occupancy,
                "label": shape_name,
            }
        )

    return sampled_data
