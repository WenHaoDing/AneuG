import open3d as o3d
import itertools
import numpy as np
import numpy
import matplotlib.pyplot as plt
# import skspatial
import copy
import trimesh.transformations


def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()

def pick_points_v2(pcd, branch):
    # pick points but give you one branch to see
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(branch)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()



def pick_points_with_mesh(pcd, mesh):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(mesh)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()

def load_npy_dict(npy_file):
    op_dict = np.load(npy_file, allow_pickle=True).item()
    op_vidx = op_dict['op_vidx']
    vertices = op_dict['vertices']
    normals = op_dict['normals']
    return op_vidx, vertices, normals


def get_average_cross_product(normals):
    sequence = np.arange(normals.shape[0])
    comb = itertools.combinations(sequence, 2)
    comb = np.asarray(list(comb)).reshape(-1, 2)
    vec1 = normals[comb[:, 0], :]
    vec2 = normals[comb[:, 1], :]
    cross = np.cross(vec1, vec2)
    cross = normalize_v(cross)
    # flip
    direction = np.sum(np.repeat(cross[0].reshape(-1, 3), cross.shape[0], axis=0) * cross, axis=1, keepdims=True)
    flip = np.ones((cross.shape[0], 1))
    flip[np.where(direction < 0)[0], :] = -1.0
    cross_flip = cross * flip
    # viz_cross_products(cross_mean)
    return np.mean(cross_flip, axis=0)  # average normal vector for the opening (direction not guaranteed)


def get_average_cross_product_from_center(vertices):
    # this works better than using og normals~
    normals = vertices - np.mean(vertices, axis=0).reshape(-1, 3)
    normals = normalize_v(normals)
    sequence = np.arange(vertices.shape[0])
    comb = itertools.combinations(sequence, 2)
    comb = np.asarray(list(comb)).reshape(-1, 2)
    vec1 = normals[comb[:, 0], :]
    vec2 = normals[comb[:, 1], :]
    cross = np.cross(vec1, vec2)
    cross = normalize_v(cross)
    # flip
    direction = np.sum(np.repeat(cross[0].reshape(-1, 3), cross.shape[0], axis=0) * cross, axis=1, keepdims=True)
    flip = np.ones((cross.shape[0], 1))
    flip[np.where(direction < 0)[0], :] = -1.0
    cross_flip = cross * flip
    # viz_cross_products(cross_mean)
    return np.mean(cross_flip, axis=0)  # average normal vector for the opening (direction not guaranteed)


def normalize_v(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors_scaled = vectors / norms
    return vectors_scaled


def viz_cross_products(cross_flip):
    cross_mean = np.mean(cross_flip, axis=0)
    fig = plt.figure(dpi=400)
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, z coordinates of vectors
    x = cross_flip[:, 0]
    y = cross_flip[:, 1]
    z = cross_flip[:, 2]

    # Plot vectors
    ax.quiver(0, 0, 0, x, y, z, arrow_length_ratio=0.1, color='b', zorder=0)
    ax.quiver(0, 0, 0, cross_mean[0], cross_mean[1], cross_mean[2], arrow_length_ratio=0.1, color='r', zorder=2)
    # Set plot labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    plt.show()
    return None


def pcd_to_approx_plane(pcd, cross_mean, viz=False):
    centroid = np.mean(pcd, axis=0)
    from skspatial.objects import Plane, Point, Vector
    from skspatial.plotting import plot_3d
    approx_plane = Plane(point=centroid, normal=cross_mean)
    points = [Point(single_point) for single_point in list(pcd)]
    points_projected = [approx_plane.project_point(single_point) for single_point in points]
    vector_projection = [Vector.from_points(point, point_projected) for point, point_projected in zip(points, points_projected)]
    if viz:
        plot_3d(
            approx_plane.plotter(lims_x=(-0.05, 0.05),
                                 lims_y=(-0.05, 0.05), alpha=0.3),
            *[point.plotter(s=1, c='k') for point in points],  # Plot original points
            *[point_projected.plotter(s=1, c='r', zorder=3) for point_projected in points_projected],
            *[vector_projection.plotter(point=point, c='b', linestyle='--') for vector_projection, point in zip(vector_projection, points)],
        )
        plt.show()
    return np.array(points_projected)


def trimesh_points_2d_to_3d(vertices_2d: np.ndarray, to_3D: np.ndarray):
    vertices_3d = np.column_stack(
            (copy.deepcopy(vertices_2d), np.zeros(len(vertices_2d)))
        )
    vertices_3d = trimesh.transformations.transform_points(vertices_3d, to_3D)
    return vertices_3d


def check_pcd_sequence(pcd: np.ndarray):
    fig = plt.figure()
    if pcd.shape[-1] == 3:
        ax = fig.add_subplot(111, projection='3d')
        for i in range(pcd.shape[0]):
            ax.scatter(pcd[i, 0], pcd[i, 1], pcd[i, 2], s=5, c=i, cmap='gray', vmin=0,
                       vmax=pcd.shape[0])
            ax.text(pcd[i, 0], pcd[i, 1], pcd[i, 2], str(i), fontsize=8)
    elif pcd.shape[-1] == 2:
        ax = fig.add_subplot(111)
        for i in range(pcd.shape[0]):
            ax.scatter(pcd[i, 0], pcd[i, 1], s=5, c=i, cmap='gray', vmin=0,
                       vmax=pcd.shape[0])
            ax.text(pcd[i, 0], pcd[i, 1], str(i), fontsize=8)

    plt.show()
    return None


def clock_sort_2D_points(pcd: np.ndarray):
    # sort the pcd clock wise
    centroid = np.mean(pcd, axis=0)
    pcd_normalized = pcd - centroid
    angles = np.arctan2(pcd_normalized[:, 0], pcd_normalized[:, 1])
    sort_indice = np.argsort(angles)
    pcd = pcd[sort_indice, :]
    return pcd


def lazy_viz_mesh(vertices, faces):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, edgecolor=[[0, 0, 0]], linewidth=0.1,
                    alpha=0.2, color='mediumpurple')
    plt.show()


def get_mapped_sequence_and_faces(op_v_indices, points_projected, vertices_3d, faces, process=False):
    if process:
        faces -= 1  # shapely index vertices from 1 instead of 0 *^*
        points_projected = points_projected[:-1, :]
        vertices_3d = vertices_3d[:-1, :]
    assert points_projected.shape == vertices_3d.shape, 'getting mapping, shape not matched'
    order = []
    N = vertices_3d.shape[0]
    for idx in range(N):
        distances = np.linalg.norm(vertices_3d[idx,:]-points_projected, axis=1)
        closest_index = np.argmin(distances)
        order.append(op_v_indices[closest_index])
    mapping = dict(zip(list(np.arange(N)), order))
    op_rec_f_map = np.vectorize(lambda x: mapping.get(x))(faces)
    return np.array(order), op_rec_f_map

