### DAgger Diffusion Navigation: DAgger Boosted Diffusion Policy for Vision-Language Navigation

## TODOs

* [x] Release the evaluation code of the DifNav Model.
* [x] Release the checkpoints of the DifNav Model for each scene.
* [ ] Release the online data augmentation code.
* [ ] Release the training data and code.

## Requirements

1. Install `Habitat simulator`: follow instructions from [ETPNav](https://github.com/MarSaKi/ETPNav) and [VLN-CE](https://github.com/jacobkrantz/VLN-CE).
2. Download the `Matterport3D Scene Dataset (MP3D)` from [Matterport](https://github.com/niessner/Matterport)
3. Install the dependencies of the [ETPNav](https://github.com/MarSaKi/ETPNav) and [Nomad](https://github.com/robodhruv/visualnav-transformer).
4. Download annotations and trained models from [Google_Drive](https://drive.google.com/drive/u/1/folders/1BcEmhBIjMo7aDo1ORbB8sjfOpjpmUrso).
5. The data should be stored under the `data` folder with the following structure:
```
data-
├── scene_datasets
│ └── mp3d
├── datasets
│ └── R2R_VLNCE_v1-2_preprocessed_BERTidx
├── checkpoints
│ └── open_area.pth
└── ddppo-models
```

## Evaluation

Evaluate each scene in our experiment(Open Area,Narrow Space,Stairs):
```
bash run_r2r/main.bash open_area eval 2333
```
