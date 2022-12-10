# DET_IMG_SEG
This package contains the scripts to train and infer the image segmentation model using Detectron2.

## Dependencies
- Python 3.8.10
- PyTorch
- Detectron2
- CUDA

## Usage

### Training
To train the model, run the following command:
```
python3 [path_to_train] --model [model_name] --weights [weights] --dataset_dir [dataset_directory] --output_dir [output_directory]
```

### Inference
To infer the model, run the following command:
```
python3 [path_to_infer]
```

## License
The source code is released under a [MIT license](LICENSE).