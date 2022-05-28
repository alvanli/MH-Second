import pickle

if __name__ == "__main__":
    info_path = "/home/nuScenes/v1.0-trainval/nuscenes_infos_10sweeps_train.pkl"
    with open(info_path, 'rb') as f:
        infos = pickle.load(f)
    print("gt_boxes", infos[1]["gt_boxes"].shape)
    print("gt_names", infos[1]["gt_names"])
    print("num_lidar_pts", infos[1]["num_lidar_pts"])
    print(infos[1].keys())