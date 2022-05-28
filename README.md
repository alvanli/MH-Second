This is just a repo to test some object detection algos with minor modifications to the original <br/>
- Original `ST3D` repo from https://github.com/CVMI-Lab/ST3D 
- Original `OpenPCDet` repo from https://github.com/open-mmlab/OpenPCDet
- Original `LISA` from https://github.com/velatkilic/LISA

JustSECOND is under `OpenPCDet/tools/JustSECOND`
- `./spinup.sh` to start docker container
- data augmentation is sometimes done in `OpenPCDet/pcdet/datasets/nuscenes/nuscenes_dataset.py`
- config is in `OpenPCDet/tools/cfgs/da-nuscenes-wato_models/mhsecond.yaml`
- train with `python3 train.py --cfg_file cfgs/da-nuscenes-wato_models/mhsecond.yaml --pretrained_model /home/OpenPCDet/outputs/cbgs_second_multihead_nds6229_updated.pth `
