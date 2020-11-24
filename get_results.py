import evaluate_cityscapes
import compute_iou
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':#was added to overcome the thread issue
    mious = []
    x = list(range(5000,155000,5000))
    
    parser = evaluate_cityscapes.getValidData()
    for i in x:
        mious.append(evaluate_cityscapes.main('./snapshots/Kitti2Nuscenes_multi/kitti_'+str(i)+'.pth', parser))
        # mious.append(round(np.nanmean(compute_iou.compute_mIoU('./data/Cityscapes/gtFine/val', 'result/cityscapes', 'dataset/cityscapes_list')) * 100, 2))

        
    plt.plot(x, mious)
    plt.show()