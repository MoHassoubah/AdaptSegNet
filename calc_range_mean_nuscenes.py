import os
import numpy as np

from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version='v1.0-mini', dataroot='/home/admin1/mohammed_hassoubah/lidar_datasets/nuscenes', verbose=True)

my_scene = nusc.scene[0]

first_sample_token = my_scene['first_sample_token']

my_sample = nusc.get('sample', first_sample_token)

sensor = 'LIDAR_TOP'
lidar_data = nusc.get('sample_data', my_sample['data'][sensor])

print(lidar_data)

# EXTENSIONS_SCAN = ['.bin']
# EXTENSIONS_LABEL = ['.label']


# def is_scan(filename):
  # return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)

# scan_path = "/home/admin1/lidar_datasets/nuscenes/samples/LIDAR_TOP"
# scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          # os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
          
# # laser parameters
# # fov_up = 10.0 / 180.0 * np.pi      # field of view up in rad
# # fov_down = -30.0 / 180.0 * np.pi  # field of view down in rad
# # fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

# # proj_W = 2048
# # proj_H = 32

# mean_depth = 0.0          
# mean_x = 0.0
# mean_y = 0.0
# mean_z = 0.0
# mean_emmission = 0.0
# num_points = 0.0
          
# for filename in scan_files:
    # scan = np.fromfile(filename, dtype=np.float32)
    # scan = scan.reshape((-1, 4))
    # # put in attribute
    # points = scan[:, 0:3]    # get xyz
    # remissions = scan[:, 3]  # get remission

    # # get depth of all points
    # depth = np.linalg.norm(points, 2, axis=1)

    # # get scan components
    # scan_x = points[:, 0]
    # scan_y = points[:, 1]
    # scan_z = points[:, 2]
    
    # print("/n")
    # print("max_depth = " + str(np.max(depth)))
    # print("max_x = " + str(np.max(scan_x)))
    # print("max_y = " + str(np.max(scan_y)))
    # print("max_z = " + str(np.max(scan_z)))
    # print("max_remissions = " + str(np.max(remissions)))
    
    # mean_depth = mean_depth + np.sum(depth)
    # mean_x = mean_x + np.sum(scan_x)
    # mean_y = mean_y + np.sum(scan_y)
    # mean_z = mean_z + np.sum(scan_z)
    # mean_emmission = mean_emmission + np.sum(remissions)
    
    # num_points = num_points + len(scan_x)

    # # # get angles of all points
    # # yaw = -np.arctan2(scan_y, scan_x)
    # # pitch = np.arcsin(scan_z / depth)

    # # # get projections in image coords
    # # proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
    # # proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

    # # # scale to image size using angular resolution
    # # proj_x *= proj_W                              # in [0.0, W]
    # # proj_y *= proj_H                              # in [0.0, H]

    # # # round and clamp for use as index
    # # proj_x = np.floor(proj_x)
    # # proj_x = np.minimum(proj_W - 1, proj_x)
    # # proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
    # # proj_x = np.copy(proj_x)  # store a copy in orig order

    # # proj_y = np.floor(proj_y)
    # # proj_y = np.minimum(proj_H - 1, proj_y)
    # # proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
    # # proj_y = np.copy(proj_y)  # stope a copy in original order

    # # # copy of depth in original order
    # # unproj_range = np.copy(depth)

    # # # order in decreasing depth
    # # indices = np.arange(depth.shape[0])
    # # order = np.argsort(depth)[::-1]
    # # depth = depth[order]
    # # indices = indices[order]
    # # points = points[order]
    # # remission = remissions[order]
    # # proj_y = proj_y[order]
    # # proj_x = proj_x[order]

    # # # assing to images
    # # proj_range[proj_y, proj_x] = depth
    # # proj_xyz[proj_y, proj_x] = points
    # # proj_remission[proj_y, proj_x] = remission
    # # proj_idx[proj_y, proj_x] = indices
    # # proj_mask = (proj_idx > 0).astype(np.int32)
    
# mean_depth = mean_depth / num_points
# mean_x = mean_x / num_points
# mean_y = mean_y / num_points
# mean_z = mean_z / num_points
# mean_emmission = mean_emmission / num_points

# print("means->depth, x, y,z, emission")    
# print("mean_depth= " + str(mean_depth))
# print("mean_x = " + str(mean_x))
# print("mean_y = " + str(mean_y))
# print("mean_z = " + str(mean_z))
# print("mean_emmission = " + str(mean_emmission))

# std_depth = 0.0
# std_x = 0.0
# std_y = 0.0
# std_z = 0.0
# std_emission = 0.0

# for filename in scan_files:
    # scan = np.fromfile(filename, dtype=np.float32)
    # scan = scan.reshape((-1, 4))
    # # put in attribute
    # points = scan[:, 0:3]    # get xyz
    # remissions = scan[:, 3]  # get remission

    # # get depth of all points
    # depth = np.linalg.norm(points, 2, axis=1)

    # # get scan components
    # scan_x = points[:, 0]
    # scan_y = points[:, 1]
    # scan_z = points[:, 2]
    
    # std_depth = std_depth + np.sum((depth - mean_depth)**2)
    # std_x = std_x + np.sum((scan_x - mean_x)**2)
    # std_y = std_y + np.sum((scan_y - mean_y)**2)
    # std_z = std_z + np.sum((scan_z - mean_z)**2)
    # std_emission = std_emission + np.sum((mean_emmission - mean_emmission)**2)
    
# std_depth = std_depth / num_points
# std_x = std_x / num_points
# std_y = std_y / num_points
# std_z = std_z / num_points
# std_emission = std_emission / num_points

# print("/nStds->depth, x, y,z, emission")
# print("std_depth = " + str(np.sqrt(std_depth)))  
# print("std_x = " + str(np.sqrt(std_x)))
# print("std_y = " + str(np.sqrt(std_y)))  
# print("std_z = " + str(np.sqrt(std_z))) 
# print("std_emission = " + str(np.sqrt(std_emission)))    