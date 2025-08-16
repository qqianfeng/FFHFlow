import numpy as np
import open3d as o3d
from urdfpy import URDF
# import transforms3d as tf
import trimesh
import copy
import pyrender

HAND_CFG = {
    'Right_Index_0': 0.2,
    'Right_Index_1': 0.2,
    'Right_Index_2': 0.2,
    'Right_Index_3': 0.2,
    'Right_Little_0': 0.2,
    'Right_Little_1': 0.2,
    'Right_Little_2': 0.2,
    'Right_Little_3': 0.2,
    'Right_Middle_0': 0.2,
    'Right_Middle_1': 0.2,
    'Right_Middle_2': 0.2,
    'Right_Middle_3': 0.2,
    'Right_Ring_0': 0.2,
    'Right_Ring_1': 0.2,
    'Right_Ring_2': 0.2,
    'Right_Ring_3': 0.2,
    'Right_Thumb_0': 0.2,
    'Right_Thumb_1': 0.2,
    'Right_Thumb_2': 0.2,
    'Right_Thumb_3': 0.2,
}

def get_hand_cfg_map(cfg_arr):
    cfg_map = HAND_CFG
    keys = sorted(HAND_CFG.keys())
    for idx, k in enumerate(keys):
        cfg_map[k] = cfg_arr[idx]
    return cfg_map

def full_joint_conf_from_partial_joint_conf(partial_joint_conf):
    """Takes in the 15 dimensional joint conf output from VAE and repeats the 3*N-th dimension to turn dim 15 into dim 20.

    Args:
        partial_joint_conf (np.array): Output from vae with dim(partial_joint_conf.position) = 15

    Returns:
        full_joint_conf (np.array): Full joint state with dim(full_joint_conf.position) = 20
    """
    full_joint_pos = 20 * [0]
    ix_full_joint_pos = 0
    for i, val in enumerate(partial_joint_conf):
        if (i + 1) % 3 == 0:
            full_joint_pos[ix_full_joint_pos] = val
            full_joint_pos[ix_full_joint_pos + 1] = val
            ix_full_joint_pos += 2
        else:
            full_joint_pos[ix_full_joint_pos] = val
            ix_full_joint_pos += 1

    full_joint_conf = full_joint_pos
    return full_joint_conf


def get_robot_fk(robot, joint_conf):
    # get the full joint config
    if joint_conf.shape[0] == 15:
        joint_conf_full = full_joint_conf_from_partial_joint_conf(joint_conf)
    elif joint_conf.shape[0] == 20:
        joint_conf_full = joint_conf
    else:
        raise Exception('Joint_conf has the wrong size in dimension one: %d. Should be 15 or 20' %
                        joint_conf.shape[0])
    cfg_map = get_hand_cfg_map(joint_conf_full)
    fk = robot.visual_trimesh_fk(cfg=cfg_map)
    return fk

def hand_pcd_viewer(pcd, grasp, mesh_robot_total, base_T_palm):
    """
    pcd and grasp both in world frame
    """
    world_T_base = np.matmul(grasp, np.linalg.inv(base_T_palm))

    # Visualization: robots downsampled pointcloud
    mesh_robot_total_tmp = copy.deepcopy(mesh_robot_total).transform(world_T_base)
    robot_points = np.asarray(mesh_robot_total_tmp.vertices)
    pcd_robot = mesh_robot_total_tmp.sample_points_uniformly(number_of_points=int(len(robot_points)/10))
    pcd_robot_points = np.asarray(pcd_robot.points)
    pcd_robot_colors = np.zeros(pcd_robot_points.shape)
    pcd_robot_colors[:] = (0.0, 0.0, 128.0/255.0)
    pcd_robot.colors = o3d.utility.Vector3dVector(pcd_robot_colors)
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    grasp_frame = origin.transform(world_T_base)

    o3d.visualization.draw_geometries([origin, pcd_robot, pcd, grasp_frame])
    
def all_grasps_pcd_viewer(grasps, robot_meshes):
    """
    pcd and grasp both in world frame
    """
    o3d_meshes = []
    for idx, grasp in enumerate(grasps.shape[0]):
        robot_mesh, base_T_palm, grasp_score = robot_meshes[idx]

        hand_pose_world_T_base = np.matmul(grasp, np.linalg.inv(base_T_palm))
        mesh_robot_total_tmp = copy.deepcopy(robot_mesh).transform(hand_pose_world_T_base)
        robot_points = np.asarray(mesh_robot_total_tmp.vertices)
        n_pcd_robot = int(len(robot_points)/100)

        # downsample robot pointcloud
        pcd_robot = mesh_robot_total_tmp.sample_points_uniformly(number_of_points=n_pcd_robot)
        pcd_robot_points = np.asarray(pcd_robot.points)
        pcd_robot_colors = np.zeros(pcd_robot_points.shape)

        # set color for each grasp
        pcd_robot_colors[:] = (255.0*grasp_score/255.0, 0.0, 0.0)
        pcd_robot.colors = o3d.utility.Vector3dVector(pcd_robot_colors)

        # create coordinate frame
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        grasp_frame = origin.transform(hand_pose_world_T_base)
        o3d_meshes.append(pcd_robot)
        
    # o3d.visualization.draw_geometries(o3d_meshes)
    return o3d_meshes

def create_robot_mesh_from_joints(joints, hand_pose):
    # load robot urdf
    robot_urdf = URDF.load('/home/jianxiang.feng/Projects/FFHFlow/meshes/hithand.urdf')
    fk = get_robot_fk(robot_urdf, joints)

    # compute robot link
    fk_link = robot_urdf.link_fk()
    assert robot_urdf.links[2].name == 'palm_link_hithand'  # link 2 must be palm
    base_T_palm = fk_link[robot_urdf.links[2]]  # get the transform from base to palm

    # create robot mesh
    mesh_robot_total = o3d.geometry.TriangleMesh()
    for tm in fk:
        pose = fk[tm] 
        mesh_robot = o3d.geometry.TriangleMesh()
        # mesh_robot.vertices = o3d.cuda.pybind.utility.Vector3dVector(np.asarray(tm.vertices.copy(),dtype=np.float64))
        # mesh_robot.triangles = o3d.cuda.pybind.utility.Vector3iVector(np.asarray(tm.faces.copy(),dtype=np.int32))
        mesh_robot.vertices = o3d.open3d_pybind.utility.Vector3dVector(np.asarray(tm.vertices.copy(),dtype=np.float64))
        mesh_robot.triangles = o3d.open3d_pybind.utility.Vector3iVector(np.asarray(tm.faces.copy(),dtype=np.int32))

        mesh_robot.transform(pose)
        mesh_robot_total += mesh_robot

    hand_pose_world_T_base = np.matmul(hand_pose, np.linalg.inv(base_T_palm))
    robot_mesh = copy.deepcopy(mesh_robot_total).transform(hand_pose_world_T_base)
    return robot_mesh

def create_robot_mesh_from_joints_pyrender(joints, hand_pose, score):
    # load robot urdf
    robot_urdf = URDF.load('/home/jianxiang.feng/Projects/FFHFlow/meshes/hithand.urdf')
    fk = get_robot_fk(robot_urdf, joints)

    # compute robot link
    fk_link = robot_urdf.link_fk()
    assert robot_urdf.links[2].name == 'palm_link_hithand'  # link 2 must be palm
    base_T_palm = fk_link[robot_urdf.links[2]]  # get the transform from base to palm

    # Add the robot to the scene
    palm_T_centr = np.linalg.inv(hand_pose)
    hand_base_T_centr = np.matmul(base_T_palm, palm_T_centr)
    centr_T_hand_base = np.linalg.inv(hand_base_T_centr)
    mesh_robot_total = []
    for tm in fk:
        pose = fk[tm]
        pose = np.matmul(centr_T_hand_base, pose)
        # from points
        # robot_mesh = o3d.geometry.TriangleMesh()
        # robot_mesh.vertices = o3d.open3d_pybind.utility.Vector3dVector(np.asarray(tm.vertices.copy(),dtype=np.float64))
        # robot_mesh.triangles = o3d.open3d_pybind.utility.Vector3iVector(np.asarray(tm.faces.copy(),dtype=np.int32))
        # robot_points = np.asarray(robot_mesh.vertices)
        # pcd_robot = robot_mesh.sample_points_uniformly(number_of_points=int(len(robot_points)/20))
        # pcd_robot_points = np.asarray(pcd_robot.points)
        # mesh_color = [0.0, 0.0, score*100.0]
        # mesh = pyrender.Mesh.from_points(pcd_robot_points, colors=np.tile(mesh_color, (pcd_robot_points.shape[0], 1)), poses=pose)
        # from mesh
        mesh = pyrender.Mesh.from_trimesh(tm, smooth=False, poses=pose)
        mesh_robot_total.append(mesh)

    return mesh_robot_total

def show_grasp_and_object_given_pcd(pcd, hand_pose, joint_conf):
    robot_urdf = URDF.load('/home/jianxiang.feng/Projects/FFHFlow/meshes/hithand.urdf')

    fk = get_robot_fk(robot_urdf, joint_conf)

    # compute robot link
    fk_link = robot_urdf.link_fk()
    assert robot_urdf.links[2].name == 'palm_link_hithand'  # link 2 must be palm
    hand_base_T_palm = fk_link[robot_urdf.links[2]]  # get the transform from base to palm

    # Compute the transform from base to object centroid frame
    palm_T_centr = np.linalg.inv(hand_pose)
    hand_base_T_centr = np.matmul(hand_base_T_palm, palm_T_centr)
    # pcd.translate(-1*pcd.get_center())
    pts = np.asarray(pcd.points)
    obj_geometry = pyrender.Mesh.from_points(pts, colors=np.tile([55, 55, 4], (pts.shape[0], 1)))
    # Construct a scene
    scene = pyrender.Scene()
    centr_T_hand_base = np.linalg.inv(hand_base_T_centr)
    # Add the robot to the scene
    for tm in fk:
        pose = fk[tm]
        pose = np.matmul(centr_T_hand_base, pose)
        mesh = pyrender.Mesh.from_trimesh(tm, smooth=False)
        scene.add(mesh, pose=pose)

    # Add cloud to scene
    # centr_T_base
    scene.add(obj_geometry, pose=np.eye(4))

    # Add more light to scene
    pose_light = np.eye(4)
    pose_light[:3, 3] = [-0.5, 0, 0]
    scene.add(pyrender.PointLight(intensity=10), pose=pose_light)
    pose_light = np.eye(4)
    pose_light[:3, 3] = [0.5, 0, 0]
    scene.add(pyrender.PointLight(intensity=10), pose=pose_light)
    pose_light = np.eye(4)
    pose_light[:3, 3] = [0, 0.9, 0]
    scene.add(pyrender.PointLight(intensity=10), pose=pose_light)
    pose_light = np.eye(4)
    pose_light[:3, 3] = [0, -0.9, 0]
    scene.add(pyrender.PointLight(intensity=10), pose=pose_light)

    T_view_1 = np.array([[0.38758592, 0.19613444, -0.90072662, -0.54629509],
                         [0.34160963, -0.93809507, -0.05727561, -0.12045398],
                         [-0.85620091, -0.28549766, -0.43059386, -0.25333053], [0., 0., 0., 1.]])
    T_view_2 = np.array([[0.38043475, 0.20440112, -0.90193658, -0.48869244],
                         [0.36146523, -0.93055351, -0.05842123, -0.11668246],
                         [-0.85124161, -0.30379325, -0.4278988, -0.22640526], [0., 0., 0., 1.]])

    # View the scene
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    # print(scene.scale)
    nc = pyrender.Node(camera=cam, matrix=T_view_2)
    scene.add_node(nc)

    pyrender.Viewer(scene, viewer_flags={"fullscreen": False}, use_raymond_lighting=True)


def show_grasp_and_object(path, palm_T_centr, joint_conf):
    """Visualize the grasp object and the hand relative to it

    Args:
        path (str): Path to bps or pointcloud or mesh of object
        palm_T_centr (4*4 array): Homogeneous transform that describes the grasp (palm pose) w.r.t to object centroid.
        joint_conf (15 or 20*1 array): 15 or 20 dimensional joint configuration
    """
    robot = URDF.load(os.path.join(
        BASE_PATH, 'meshes/hithand_palm/hithand.urdf'))

    # get the full joint config
    if joint_conf.shape[0] == 15:
        joint_conf_full = utils.full_joint_conf_from_partial_joint_conf(
            joint_conf)
    elif joint_conf.shape[0] == 20:
        joint_conf_full = joint_conf
    else:
        raise Exception('Joint_conf has the wrong size in dimension one: %d. Should be 15 or 20' %
                        joint_conf.shape[0])
    cfg_map = utils.get_hand_cfg_map(joint_conf_full)

    # compute fk for meshes and links
    fk = robot.visual_trimesh_fk(cfg=cfg_map)
    fk_link = robot.link_fk()
    assert robot.links[2].name == 'palm_link_hithand'  # link 2 must be palm
    # get the transform from base to palm
    palm_T_base = fk_link[robot.links[2]]

    # Compute the transform from base to object centroid frame
    centr_T_palm = np.linalg.inv(palm_T_centr)
    centr_T_base = np.matmul(palm_T_base, centr_T_palm)

    # Turn open3d pcd into pyrender mesh or load trimesh from path
    if 'bps' in path or 'pcd' in path:
        obj_pcd = utils.load_rendered_pcd(path)
        pts = np.asarray(obj_pcd.points)
        obj_geometry = pyrender.Mesh.from_points(pts,
                                                 colors=np.tile([55, 55, 4], (pts.shape[0], 1)))
    else:
        mesh = trimesh.load_mesh(path)
        obj_geometry = pyrender.Mesh.from_trimesh(mesh,
                                                  material=pyrender.MetallicRoughnessMaterial(
                                                      emissiveFactor=[
                                                          255, 0, 0],
                                                      doubleSided=True,
                                                      baseColorFactor=[255, 0, 0, 1]))

    # Construct a scene
    scene = pyrender.Scene()

    base_T_centr = np.linalg.inv(centr_T_base)
    # Add the robot to the scene
    for tm in fk:
        pose = fk[tm]
        pose = np.matmul(base_T_centr, pose)
        mesh = pyrender.Mesh.from_trimesh(tm, smooth=False)
        scene.add(mesh, pose=pose)

    # Add cloud to scene
    # centr_T_base
    scene.add(obj_geometry, pose=np.eye(4))

    # Add more light to scene
    pose_light = np.eye(4)
    pose_light[:3, 3] = [-0.5, 0, 0]
    scene.add(pyrender.PointLight(intensity=10), pose=pose_light)
    pose_light = np.eye(4)
    pose_light[:3, 3] = [0.5, 0, 0]
    scene.add(pyrender.PointLight(intensity=10), pose=pose_light)
    pose_light = np.eye(4)
    pose_light[:3, 3] = [0, 0.9, 0]
    scene.add(pyrender.PointLight(intensity=10), pose=pose_light)
    pose_light = np.eye(4)
    pose_light[:3, 3] = [0, -0.9, 0]
    scene.add(pyrender.PointLight(intensity=10), pose=pose_light)

    T_view_1 = np.array([[0.38758592, 0.19613444, -0.90072662, -0.54629509],
                         [0.34160963, -0.93809507, -0.05727561, -0.12045398],
                         [-0.85620091, -0.28549766, -0.43059386, -0.25333053], [0., 0., 0., 1.]])
    T_view_2 = np.array([[0.38043475, 0.20440112, -0.90193658, -0.48869244],
                         [0.36146523, -0.93055351, -0.05842123, -0.11668246],
                         [-0.85124161, -0.30379325, -0.4278988, -0.22640526], [0., 0., 0., 1.]])

    # View the scene
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    # print(scene.scale)
    nc = pyrender.Node(camera=cam, matrix=T_view_2)
    scene.add_node(nc)

    pyrender.Viewer(scene, viewer_flags={
                    "fullscreen": False}, use_raymond_lighting=True)


def render_hand_in_configuration(cfg=np.zeros(20)):
    path = os.path.dirname(os.path.abspath(__file__))
    robot = URDF.load(os.path.join(BASE_PATH, 'meshes/hithand_palm/hithand.urdf'))

    cfg_map = utils.get_hand_cfg_map(cfg)

    # compute fk for meshes and links
    fk = robot.visual_trimesh_fk(cfg=cfg_map)

    # Construct a scene
    scene = pyrender.Scene()

    # Add the robot to the scene
    for tm in fk:
        pose = fk[tm]
        mesh = pyrender.Mesh.from_trimesh(tm, smooth=False)
        scene.add(mesh, pose=pose)

    # Add more light to scene
    pose_light = np.eye(4)
    pose_light[:3, 3] = [-0.5, 0, 0]
    scene.add(pyrender.PointLight(intensity=6), pose=pose_light)
    pose_light = np.eye(4)
    pose_light[:3, 3] = [0.5, 0, 0]
    scene.add(pyrender.PointLight(intensity=6), pose=pose_light)
    pose_light = np.eye(4)
    pose_light[:3, 3] = [0, 0.9, 0]
    scene.add(pyrender.PointLight(intensity=6), pose=pose_light)
    pose_light = np.eye(4)
    pose_light[:3, 3] = [0, -0.9, 0]
    scene.add(pyrender.PointLight(intensity=6), pose=pose_light)

    # View the scene
    pyrender.Viewer(scene, use_raymond_lighting=True)


def show_individual_ground_truth_grasps(obj_name, grasp_data_path, outcome='positive'):
    # Get mesh for object
    mesh_path = get_mesh_path(obj_name)

    # Get the ground truth grasps
    data_handler = GraspDataHandlerVae(file_path=grasp_data_path)
    palm_poses, joint_confs, num_succ = data_handler.get_grasps_for_object(obj_name,
                                                                           outcome=outcome)

    # Display the grasps in a loop
    for i, (palm_pose, joint_conf) in enumerate(zip(palm_poses, joint_confs)):
        palm_hom = utils.hom_matrix_from_pos_quat_list(palm_pose)
        th = joint_conf[16]
        joint_conf = np.zeros(20)
        joint_conf[16] = th
        show_grasp_and_object(mesh_path, palm_hom, joint_conf)
        print(joint_conf)


if __name__ == "__main__":
    joint_confs = np.zeros(30, 15)
    grasps = np.eys(30, 4, 4)
    pcd = np.ones(2048, 3) 
    robot_meshes = []
    grasp_scores = np.ones(30)

    for i in range(joint_confs.shape[0]):
        joint_conf = joint_confs[i]
        robot_mesh, base_T_palm = create_robot_mesh_from_joints(joint_conf)
        robot_meshes.append((robot_mesh, base_T_palm, grasp_scores[i]))
        
    o3d_robot_meshes = all_grasps_pcd_viewer(grasps, robot_meshes)
    # hand_pcd_viewer(pcd, grasp, mesh_robot_total, base_T_palm)

