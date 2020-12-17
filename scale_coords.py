import numpy as np

import os

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']


def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)
  
seqs = ["00","01","02","03","04","05","06","07","08","09","10"]
mega_scan_files = []
for seq in seqs:
    scan_path = os.path.join("C:\lidar_datasets\kitti_data", "volodyne_points", "data_odometry_velodyne", "dataset", "sequences", seq, "velodyne")
    scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
          
    mega_scan_files.extend(scan_files)
    
mega_scan_files.sort() 
max_idx = 0
min_idx = 1
max_min_x = [0,0]
max_min_y = [0,0]
max_min_z = [0,0]

max_min_emission = [0,0]

mean_depth = 0         
mean_x = 0
mean_y = 0
mean_z = 0
mean_emmission = 0
num_points = 0
max_points_scan = 0

max_intensity_val = 0

for i in range(0, len(mega_scan_files)):
    filename = mega_scan_files[i]
    scan = np.fromfile(filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    remissions = scan[:, 3]
    
    points = scan[:, 0:3]
    scan_x = points[:, 0]
    scan_y = points[:, 1]
    scan_z = points[:, 2]
    
    
    depth = np.linalg.norm(points, 2, axis=1)
    
    
    num_points = num_points + points.shape[0]
    # print(points.shape)
    # print("#points = " + str(num_points))
            
    if(max_intensity_val < np.max(remissions)):
        max_intensity_val = np.max(remissions)
    
    mean_depth = mean_depth + np.sum(depth)
    mean_x = mean_x + np.sum((scan_x))
    # print("#mean_x = " + str(mean_x))
    mean_y = mean_y + np.sum(scan_y)
    mean_z = mean_z + np.sum(scan_z)
    mean_emmission = mean_emmission + np.sum(remissions)

mean_depth = mean_depth / num_points
mean_x = mean_x / num_points
mean_y = mean_y / num_points
mean_z = mean_z / num_points
mean_emmission = mean_emmission / num_points

print("max_intensity_val = " + str(max_intensity_val))
print("#points = " + str(num_points))
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


for i in range(0, len(mega_scan_files)):
    filename = mega_scan_files[i]
    scan = np.fromfile(filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    remissions = scan[:, 3]
    
    points = scan[:, 0:3]
    scan_x = points[:, 0]
    scan_y = points[:, 1]
    scan_z = points[:, 2]
    
    
    depth = np.linalg.norm(points, 2, axis=1)
    
        
    
    std_depth = std_depth + (np.sum((depth - mean_depth)**2)/num_points)
    std_x = std_x + (np.sum((scan_x - mean_x)**2)/num_points)
    std_y = std_y + (np.sum((scan_y - mean_y)**2)/num_points)
    std_z = std_z + (np.sum((scan_z - mean_z)**2)/num_points)
    std_emission = std_emission + (np.sum((remissions - mean_emmission)**2)/num_points)
    
    
print("/nStds->depth, x, y,z, emission")
print("std_depth = " + str(np.sqrt(std_depth)))  
print("std_x = " + str(np.sqrt(std_x)))
print("std_y = " + str(np.sqrt(std_y)))  
print("std_z = " + str(np.sqrt(std_z))) 
print("std_emission = " + str(np.sqrt(std_emission)))       