from easydict import EasyDict as edict


MODEL_CONFIG = edict({
    "CLASS_NAMES": ['car','truck', 'construction_vehicle', 'bus', 'trailer',
              'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'],
    "MAP_TO_BEV": {
        "NUM_BEV_FEATURES": 256
    },
    "BACKBONE_2D": {
        "LAYER_NUMS": [5, 5],
        "LAYER_STRIDES": [1, 2],
        "NUM_FILTERS": [128, 256],
        "UPSAMPLE_STRIDES": [1, 2],
        "NUM_UPSAMPLE_FILTERS": [256, 256]
    },
    "DENSE_HEAD": {
        "NAME": "AnchorHeadMulti",
        "CLASS_AGNOSTIC": False,

        "DIR_OFFSET": 0.78539,
        "DIR_LIMIT_OFFSET": 0.0,
        "NUM_DIR_BINS": 2,

        "USE_MULTIHEAD": True,
        "SEPARATE_MULTIHEAD": True,
        "ANCHOR_GENERATOR_CONFIG": [
            {
                'class_name': "car",
                'anchor_sizes': [[4.63, 1.97, 1.74]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-0.95],
                'align_center': False,
                'feature_map_stride': 8,
                'matched_threshold': 0.6,
                'unmatched_threshold': 0.45
            },
            {
                'class_name': "truck",
                'anchor_sizes': [[6.93, 2.51, 2.84]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-0.6],
                'align_center': False,
                'feature_map_stride': 8,
                'matched_threshold': 0.55,
                'unmatched_threshold': 0.4
            },
            {
                'class_name': "construction_vehicle",
                'anchor_sizes': [[6.37, 2.85, 3.19]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-0.225],
                'align_center': False,
                'feature_map_stride': 8,
                'matched_threshold': 0.5,
                'unmatched_threshold': 0.35
            },
            {
                'class_name': "bus",
                'anchor_sizes': [[10.5, 2.94, 3.47]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-0.085],
                'align_center': False,
                'feature_map_stride': 8,
                'matched_threshold': 0.55,
                'unmatched_threshold': 0.4
            },
            {
                'class_name': "trailer",
                'anchor_sizes': [[12.29, 2.90, 3.87]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [0.115],
                'align_center': False,
                'feature_map_stride': 8,
                'matched_threshold': 0.5,
                'unmatched_threshold': 0.35
            },
            {
                'class_name': "barrier",
                'anchor_sizes': [[0.50, 2.53, 0.98]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-1.33],
                'align_center': False,
                'feature_map_stride': 8,
                'matched_threshold': 0.55,
                'unmatched_threshold': 0.4
            },
            {
                'class_name': "motorcycle",
                'anchor_sizes': [[2.11, 0.77, 1.47]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-1.085],
                'align_center': False,
                'feature_map_stride': 8,
                'matched_threshold': 0.5,
                'unmatched_threshold': 0.3
            },
            {
                'class_name': "bicycle",
                'anchor_sizes': [[1.70, 0.60, 1.28]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-1.18],
                'align_center': False,
                'feature_map_stride': 8,
                'matched_threshold': 0.5,
                'unmatched_threshold': 0.35
            },
            {
                'class_name': "pedestrian",
                'anchor_sizes': [[0.73, 0.67, 1.77]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-0.935],
                'align_center': False,
                'feature_map_stride': 8,
                'matched_threshold': 0.6,
                'unmatched_threshold': 0.4
            },
            {
                'class_name': "traffic_cone",
                'anchor_sizes': [[0.41, 0.41, 1.07]],
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-1.285],
                'align_center': False,
                'feature_map_stride': 8,
                'matched_threshold': 0.6,
                'unmatched_threshold': 0.4
            },
        ],

        "SHARED_CONV_NUM_FILTER": 64,
        "RPN_HEAD_CFGS": [
            {
                'HEAD_CLS_NAME': ['car'],
            },
            {
                'HEAD_CLS_NAME': ['truck', 'construction_vehicle'],
            },
            {
                'HEAD_CLS_NAME': ['bus', 'trailer'],
            },
            {
                'HEAD_CLS_NAME': ['barrier'],
            },
            {
                'HEAD_CLS_NAME': ['motorcycle', 'bicycle'],
            },
            {
                'HEAD_CLS_NAME': ['pedestrian', 'traffic_cone'],
            },
        ],

        "SEPARATE_REG_CONFIG": {
            "NUM_MIDDLE_CONV": 1,
            "NUM_MIDDLE_FILTER": 64,
            "REG_LIST": ['reg:2', 'height:1', 'size:3', 'angle:2']
        },
        "TARGET_ASSIGNER_CONFIG":{
            "NAME": "AxisAlignedTargetAssigner",
            "POS_FRACTION": -1.0,
            "SAMPLE_SIZE": 512,
            "NORM_BY_NUM_EXAMPLES": False,
            "MATCH_HEIGHT": False,
            "BOX_CODER": "ResidualCoder",
            "BOX_CODER_CONFIG": {
                'code_size': 7,
                'encode_angle_by_sincos': True
            }
        },
        "LOSS_CONFIG": {
            "REG_LOSS_TYPE": "WeightedL1Loss",
            "LOSS_WEIGHTS": {
                'pos_cls_weight': 1.0,
                'neg_cls_weight': 2.0,
                'cls_weight': 1.0,
                'loc_weight': 0.25,
                'dir_weight': 0.2,
                'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            }
        }
    },
    "POST_PROCESSING": {
        "RECALL_THRESH_LIST": [0.3, 0.5, 0.7],
        "SCORE_THRESH": 0.1,
        "OUTPUT_RAW_SCORE": False,

        "EVAL_METRIC": "kitti",

        "NMS_CONFIG": {
            "MULTI_CLASSES_NMS": True,
            "NMS_TYPE": "nms_gpu",
            "NMS_THRESH": 0.2,
            "NMS_PRE_MAXSIZE": 1000,
            "NMS_POST_MAXSIZE": 83
        }
    }
    # "ROI_HEAD": {
    #     "SHARED_FC": [256, 256],
    #     "ROI_GRID_POOL": {
    #         "GRID_SIZE": 7
    #     },
    #     "DP_RATIO": 0.3,
    #     "IOU_FC": [256, 256],
    #     "ROI_GRID_POOL": {
    #         "DOWNSAMPLE_RATIO": 8,
    #         "GRID_SIZE": 7,
    #         "IN_CHANNEL": 512
    #     },
    #     "NMS_CONFIG": {
    #         "TRAIN":{
    #             "NMS_TYPE": "nms_gpu",
    #             "MULTI_CLASSES_NMS": False,
    #             "NMS_PRE_MAXSIZE": 9000,
    #             "NMS_POST_MAXSIZE": 512,
    #             "NMS_THRESH": 0.8
    #         },
    #         "TEST":{
    #             "NMS_TYPE": "nms_gpu",
    #             "MULTI_CLASSES_NMS": False,
    #             "NMS_PRE_MAXSIZE": 1024,
    #             "NMS_POST_MAXSIZE": 100,
    #             "NMS_THRESH": 0.7
    #         },
    #         "NMS_TYPE": "nms_gpu",
    #         "MULTI_CLASSES_NMS": False,
    #         "NMS_PRE_MAXSIZE": 1024,
    #         "NMS_POST_MAXSIZE": 100,
    #         "NMS_THRESH": 0.7
    #     },
    #     "TARGET_CONFIG":{
    #         "BOX_CODER": "ResidualCoder",
    #         "ROI_PER_IMAGE": 128,
    #         "FG_RATIO": 0.5,

    #         "SAMPLE_ROI_BY_EACH_CLASS": True,
    #         "CLS_SCORE_TYPE": "raw_roi_iou",

    #         "CLS_FG_THRESH": 0.75,
    #         "CLS_BG_THRESH": 0.25,
    #         "CLS_BG_THRESH_LO": 0.1,
    #         "HARD_BG_RATIO": 0.8,

    #         "REG_FG_THRESH": 0.55
    #     },
    #     "LOSS_CONFIG": {
    #         "LOSS_WEIGHTS": {
    #             'rcnn_iou_weight': 1.0,
    #             'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    #         }
    #     }
    # } 
})
