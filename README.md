# SloMo
Video Frame Interpolation
## Installation
 ```
 conda install pytorch=0.4.1 cuda92 torchvision==0.2.0
  ```
## Training
### Adobe240fps
 ```
 python data\create_dataset.py --ffmpeg_dir path\to\folder\containing\ffmpeg --videos_folder path\to\adobe240fps\videoFolder --dataset_folder path\to\dataset --dataset adobe240fps
  ```
### Custom
 ```
python data\create_dataset.py --ffmpeg_dir path\to\folder\containing\ffmpeg --videos_folder path\to\adobe240fps\videoFolder --dataset_folder path\to\dataset
 ```
### Training
 ```
 python train.py --dataset_root path\to\dataset --checkpoint_dir path\to\save\checkpoints
  ```
## Tensorboard
```
tensorboard --logdir log --port 6007
 ```
## Evaluation
### Video 
```
python video_to_slomo.py --ffmpeg path\to\folder\containing\ffmpeg --video path\to\video.mp4 --sf N --checkpoint path\to\checkpoint.ckpt --fps M --output path\to\output.mkv
```
### Figure
```
python eval.py data/input.mp4 --checkpoint=data/SuperSloMo.ckpt --output=data/output.mp4 --scale=4
```
