import os
import os.path as osp
import numpy as np

from nuscenes.nuscenes import NuScenes

from nuscenes.utils.data_classes import LidarPointCloud

sensor = 'LIDAR_TOP'

nusc = NuScenes(version='v1.0-trainval', dataroot='C:/lidar_datasets/nuscenes', verbose=True)

###########################################################
##########################################################
# my_scene = nusc.scene[0]
# print(len(nusc.scene))

# first_sample_token = my_scene['first_sample_token']

# my_sample = nusc.get('sample', first_sample_token)

# lidar_data = nusc.get('sample_data', my_sample['data'][sensor])
# lidar_seg = nusc.get('lidarseg', my_sample['data'][sensor]) #returns data as # # print(nusc.lidarseg[index])

# print(lidar_data)
# print("")
# print(lidar_seg)
# print(nusc.lidarseg[0])

# lidarseg_labels_filename = osp.join('C:/lidar_datasets/nuscenes', lidar_seg['filename'])

# points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)
# print(points_label[:10])

# print(nusc.list_lidarseg_categories(sort_by='count'))
# print("")
# print(nusc.lidarseg_idx2name_mapping)


# lidar_pc = LidarPointCloud.from_file(osp.join('C:/lidar_datasets/nuscenes', lidar_data["filename"]))

# print(lidar_pc.points)
# print("lidar_pc.points.shape = " + str(lidar_pc.points.shape))
# print("points_label.shape = " + str(points_label.shape))
# print("max_x = " + str(np.max(lidar_pc.points[0,:])))
# print("max_y = " + str(np.max(lidar_pc.points[1,:])))
# print("max_z = " + str(np.max(lidar_pc.points[2,:])))
# print("max_remissions = " + str(np.max(lidar_pc.points[3,:])))

# ################################################################
# # # scan_files = [os.path.join(scan_path, f) for f in scan_path if is_scan(f)]
# # # laser parameters
# # # fov_up = 10.0 / 180.0 * np.pi      # field of view up in rad
# # # fov_down = -30.0 / 180.0 * np.pi  # field of view down in rad
# # # fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

# # # proj_W = 2048
# # # proj_H = 32
# #############################################################

mean_depth = 0         
mean_x = 0
mean_y = 0
mean_z = 0
mean_emmission = 0
num_points = 0
max_points_scan = 0

max_intensity_val = 0

# print(nusc.lidarseg[0])

for idx in range(700):
    my_scene = nusc.scene[idx]
    # print(len(nusc.scene))

    first_sample_token = my_scene['first_sample_token']
    
    my_sample = nusc.get('sample', first_sample_token)
    # print(my_sample)

    while(True):
        lidar_data = nusc.get('sample_data', my_sample['data'][sensor])

        # print(lidar_data)
        
        scan = np.fromfile(osp.join('C:/lidar_datasets/nuscenes', lidar_data["filename"]), dtype=np.float32)
        scan_p = scan.reshape((-1, 5))[:, :4] #when reshape(-1, 4) you fit the whole scan #points times (x,y,z, intensity, ring index) in one matrix of 4 columns ->
        #that is why the results of the output matrix don't make sense-> so this is the right way to reshape
        
        points = scan_p[:, 0:3]    # get xyz
        remissions = scan_p[:, 3]  # get remission
        
        depth = np.linalg.norm(points, 2, axis=1)
        
        scan_x = points[:, 0]
        scan_y = points[:, 1]
        scan_z = points[:, 2]
                
        num_points = num_points + points.shape[0]

        
        # if(max_points_scan < points.shape[0]):
            # max_points_scan = points.shape[0]
        #####################################################
        ####################################################
        # lidar_pc = LidarPointCloud.from_file(osp.join('C:/lidar_datasets/nuscenes', lidar_data["filename"]))

        # # put in attribute
        # remissions = lidar_pc.points[3,:]  # get remission

        # # get depth of all points
        # depth = np.linalg.norm(lidar_pc.points[0:3,:], 2, axis=0)

        # # get scan components
        # scan_x = lidar_pc.points[0,:]
        # scan_y = lidar_pc.points[1,:]
        # scan_z = lidar_pc.points[2,:]
        
        # num_points = num_points + llidar_pc.points.shape[1]
            
        if(max_intensity_val < np.max(remissions)):
            max_intensity_val = np.max(remissions)
        
        # print("")
        # print("max_depth = " + str(np.max(depth)))
        # print("max_x = " + str(np.max(scan_x)))
        # print("max_y = " + str(np.max(scan_y)))
        # print("max_z = " + str(np.max(scan_z)))
        # print("max_remissions = " + str(np.max(remissions)))
        
        # print("/n")
        # print("depth[0] = " + str((depth[0])))
        # print("x[0] = " + str((scan_x[0])))
        # print("y[0] = " + str((scan_y[0])))
        # print("z[0] = " + str((scan_z[0])))
        # print("remissions[0] = " + str(np.max(remissions[0])))
        
        mean_depth = mean_depth + np.sum(depth)
        mean_x = mean_x + np.sum(scan_x)
        mean_y = mean_y + np.sum(scan_y)
        mean_z = mean_z + np.sum(scan_z)
        mean_emmission = mean_emmission + np.sum(remissions/255.0)
        
        if(my_sample['next'] == ''):
            break
        
        my_sample = nusc.get('sample', my_sample['next'])
###################################################################
###################################################################
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
###############################################################
###############################################################
    
mean_depth = mean_depth / num_points
mean_x = mean_x / num_points
mean_y = mean_y / num_points
mean_z = mean_z / num_points
mean_emmission = mean_emmission / num_points

# # mean_depth= 9.348510395805711
# # mean_x = -0.08803494651813032
# # mean_y = -0.19697970886639926
# # mean_z = -0.510999429163294
# # mean_emmission = 17.73328291036439

print("#points = " + str(num_points))
print("max_intensity_val = " + str(max_intensity_val))
print("means->depth, x, y,z, emission")    
print("mean_depth= " + str(mean_depth))
print("mean_x = " + str(mean_x))
print("mean_y = " + str(mean_y))
print("mean_z = " + str(mean_z))
print("mean_emmission = " + str(mean_emmission))

std_depth = 0
std_x = 0
std_y = 0
std_z = 0
std_emission = 0

for idx in range(700):
    my_scene = nusc.scene[idx]
    # print(len(nusc.scene))

    first_sample_token = my_scene['first_sample_token']
    
    my_sample = nusc.get('sample', first_sample_token)

    while(True):
        lidar_data = nusc.get('sample_data', my_sample['data'][sensor])

        # print(lidar_data)
        
        scan = np.fromfile(osp.join('C:/lidar_datasets/nuscenes', lidar_data["filename"]), dtype=np.float32)
        scan_p = scan.reshape((-1, 5))[:, :4]
        
        points = scan_p[:, 0:3]    # get xyz
        remissions = scan_p[:, 3]  # get remission
        
        depth = np.linalg.norm(points, 2, axis=1)
        
        scan_x = points[:, 0]
        scan_y = points[:, 1]
        scan_z = points[:, 2]
        
        ####################################################
        ####################################################
        # lidar_pc = LidarPointCloud.from_file(osp.join('C:/lidar_datasets/nuscenes', lidar_data["filename"]))

        # # put in attribute
        # remissions = lidar_pc.points[3,:]  # get remission

        # # get depth of all points
        # depth = np.linalg.norm(lidar_pc.points[0:3,:], 2, axis=0)

        # # get scan components
        # scan_x = lidar_pc.points[0,:]
        # scan_y = lidar_pc.points[1,:]
        # scan_z = lidar_pc.points[2,:]
    
        std_depth = std_depth + (np.sum((depth - mean_depth)**2)/num_points)
        std_x = std_x + (np.sum((scan_x - mean_x)**2)/num_points)
        std_y = std_y + (np.sum((scan_y - mean_y)**2)/num_points)
        std_z = std_z + (np.sum((scan_z - mean_z)**2)/num_points)
        std_emission = std_emission + (np.sum(((remissions/255.0) - mean_emmission)**2)/num_points)
        
        if(my_sample['next'] == ''):
            break
        
        my_sample = nusc.get('sample', my_sample['next'])
        
        
   


print("/nStds->depth, x, y,z, emission")
print("std_depth = " + str(np.sqrt(std_depth)))  
print("std_x = " + str(np.sqrt(std_x)))
print("std_y = " + str(np.sqrt(std_y)))  
print("std_z = " + str(np.sqrt(std_z))) 
print("std_emission = " + str(np.sqrt(std_emission)))    

