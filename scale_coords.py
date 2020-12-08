import numpy as np

import os

from random import randint

import matplotlib.pyplot as plt

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']


def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)
  

def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)
  
seqs = ["00","01","02","03","04","05","06","07","08","09","10"]
mega_scan_files = []
mega_label_files = []
for seq in seqs:
    scan_path = os.path.join("C:\lidar_datasets\kitti_data", "volodyne_points", "data_odometry_velodyne", "dataset", "sequences", seq, "velodyne")
    label_path = os.path.join("C:\lidar_datasets\kitti_data", "data_odometry_labels", "dataset", "sequences", seq, "labels")
    scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
          
    label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
      os.path.expanduser(label_path)) for f in fn if is_label(f)]
      
    if(len(scan_files) != len(label_files)):
        print("num files differ from scan and labels")
          
    mega_scan_files.extend(scan_files)
    mega_label_files.extend(label_files)
    
mega_scan_files.sort()
mega_label_files.sort() 
max_idx = 0
min_idx = 1
max_min_x = [0,0]
max_min_y = [0,0]
max_min_z = [0,0]

max_min_emission = [0,0]

mean_depth = 0.0          
mean_x = 0.0
mean_y = 0.0
mean_z = 0.0
mean_emmission = 0.0
num_points = 0.0
max_points_scan = 0

depth_building = []
emission_building = []

depth_vegetation = []
emission_vegetation = []

depth_road = []
emission_road = []

depth_sidewalk = []
emission_sidewalk = []

rand_list = []

for cnt in range(0, 4):

    i = randint(0, len(mega_scan_files))
    while(i in rand_list):
        i = randint(0, len(mega_scan_files))
        
    rand_list.append(i)
    print(i)
    # print(mega_scan_files[i])
    # print(mega_label_files[i])
    filename = mega_scan_files[i]
    scan = np.fromfile(filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    
    
    label = np.fromfile(mega_label_files[i], dtype=np.int32)
    label = label.reshape((-1))
    label = label & 0xFFFF
    
    remissions = scan[:, 3]
    remissions = remissions - 0.28743615448305476
    remissions = remissions/0.14407062272691376
    
    points = scan[:, 0:3]
    scan_x = points[:, 0]
    scan_y = points[:, 1]
    scan_z = points[:, 2]
    
    
    depth = np.linalg.norm(points, 2, axis=1)
    depth = depth - 11.61845631266461
    depth = depth/10.114014563568704
    
    # print(label.shape)
    # print(depth.shape)
    # print(remissions.shape)
    
    ind_label_building=label == 50
    # depth_building.extend(depth[ind_label_building].tolist())
    # emission_building.extend(remissions[ind_label_building].tolist())
    
    ind_label_vege=label == 70
    # depth_vegetation.extend(depth[ind_label_vege].tolist())
    # emission_vegetation.extend(remissions[ind_label_vege].tolist())
    
    ind_label_road=label == 40
    # depth_road.extend(depth[ind_label_road].tolist())
    # emission_road.extend(remissions[ind_label_road].tolist())
    
    ind_label_sidewalk=label == 48
    # depth_sidewalk.extend(depth[ind_label_sidewalk].tolist())
    # emission_sidewalk.extend(remissions[ind_label_sidewalk].tolist())
    

    plt.scatter(depth[ind_label_building],remissions[ind_label_building])
    
    # plt.xlim([-30, 30])
    # plt.ylim([-30, 30])
    # plt.yticks(np.arange(np.min(remissions[ind_label_building]), np.max(remissions[ind_label_building]), 0.0001))
    # plt.xticks(np.arange(np.min(depth[ind_label_building]), np.max(depth[ind_label_building]), 0.0001))
    plt.title('kitti_building_random_scene #' + str(cnt))
    plt.xlabel("depth")
    plt.ylabel("intenisty")
    plt.savefig('C:/Users/mohas/Desktop/emm_depth/building/kitti_building_' + str(cnt)+'.png',dpi=300)
    plt.clf()

    plt.scatter(depth[ind_label_vege],remissions[ind_label_vege])
    # plt.xlim([-30, 30])
    # plt.ylim([-30, 30])
    # plt.yticks(np.arange(np.min(remissions[ind_label_vege]), np.max(remissions[ind_label_vege]), 0.0001))
    # plt.xticks(np.arange(np.min(depth[ind_label_vege]), np.max(depth[ind_label_vege]), 0.0001))
    plt.title('kitti_vegeration_random_scene #' + str(cnt))
    plt.xlabel("depth")
    plt.ylabel("intenisty")
    plt.savefig('C:/Users/mohas/Desktop/emm_depth/vegeration/kitti_vegetation_' + str(cnt)+'.png',dpi=300)
    plt.clf()

    plt.scatter(depth[ind_label_road],remissions[ind_label_road])
    # plt.xlim([-30, 30])
    # plt.ylim([-30, 30])
    # plt.yticks(np.arange(np.min(remissions[ind_label_road]), np.max(remissions[ind_label_road]), 0.0001))
    # plt.xticks(np.arange(np.min(depth[ind_label_road]), np.max(depth[ind_label_road]), 0.0001))
    plt.title('kitti_road_random_scene #' + str(cnt))
    plt.xlabel("depth")
    plt.ylabel("intenisty")
    plt.savefig('C:/Users/mohas/Desktop/emm_depth/road/kitti_road_' + str(cnt)+'.png',dpi=300)
    plt.clf()

    plt.scatter(depth[ind_label_sidewalk],remissions[ind_label_sidewalk])
    # plt.xlim([-30, 30])
    # plt.ylim([-30, 30])
    # plt.yticks(np.arange(np.min(remissions[ind_label_sidewalk]), np.max(remissions[ind_label_sidewalk]), 0.0001))
    # plt.xticks(np.arange(np.min(depth[ind_label_sidewalk]), np.max(depth[ind_label_sidewalk]), 0.0001))
    # plt.show()
    plt.title('kitti_sidewalk_random_scene #' + str(cnt))
    plt.xlabel("depth")
    plt.ylabel("intenisty")
    plt.savefig('C:/Users/mohas/Desktop/emm_depth/sidewalk/kitti_sidewalk_' + str(cnt)+'.png',dpi=300)
    plt.clf()
    
    # num_points = num_points + points.shape[0]
    # # print(points.shape)
    # # print("#points = " + str(num_points))
    
    # mean_depth = mean_depth + np.sum(depth)
    # mean_x = mean_x + np.sum((scan_x))
    # # print("#mean_x = " + str(mean_x))
    # mean_y = mean_y + np.sum(scan_y)
    # mean_z = mean_z + np.sum(scan_z)
    # mean_emmission = mean_emmission + np.sum(remissions)

# mean_depth = mean_depth / num_points
# mean_x = mean_x / num_points
# mean_y = mean_y / num_points
# mean_z = mean_z / num_points
# mean_emmission = mean_emmission / num_points

# print("#points = " + str(num_points))
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


# for i in range(0, len(mega_scan_files)):
    # filename = mega_scan_files[i]
    # scan = np.fromfile(filename, dtype=np.float32)
    # scan = scan.reshape((-1, 4))
    # remissions = scan[:, 3]
    
    # points = scan[:, 0:3]
    # scan_x = points[:, 0]
    # scan_y = points[:, 1]
    # scan_z = points[:, 2]
    
    
    # depth = np.linalg.norm(points, 2, axis=1)
    
        
    
    # std_depth = std_depth + (np.sum((depth - mean_depth)**2)/num_points)
    # std_x = std_x + (np.sum((scan_x - mean_x)**2)/num_points)
    # std_y = std_y + (np.sum((scan_y - mean_y)**2)/num_points)
    # std_z = std_z + (np.sum((scan_z - mean_z)**2)/num_points)
    # std_emission = std_emission + (np.sum((remissions - mean_emmission)**2)/num_points)
    
    
# print("/nStds->depth, x, y,z, emission")
# print("std_depth = " + str(np.sqrt(std_depth)))  
# print("std_x = " + str(np.sqrt(std_x)))
# print("std_y = " + str(np.sqrt(std_y)))  
# print("std_z = " + str(np.sqrt(std_z))) 
# print("std_emission = " + str(np.sqrt(std_emission)))       