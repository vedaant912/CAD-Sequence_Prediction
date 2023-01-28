# CAD Operation Sequence

predict sequence of operations from CAD file

## 3D data available

'data/three_ops/expressions.txt'.

The 3D data is available in the following format :

- sp(x,y,z,r), cu(x,y,z,r) and cy(x,y,z,r,h). These are the three primitives present in the dataset.
- Here, (x,y,z) are the center (location) of the shapes in the 3D space.
- r denotes the radius for sphere and cylinder whereas r specifies size for the cube.
- h denotes the height of the cylinder.

## Dependency

- Python > 3.5

- Create conda env using environment.yml file
```bash
conda env create -f environment.yml -n CAD
conda activate CAD
```

## Data

Before training on the data, download the dataset and generate voxel files for training.

- Download the CAD expressions [dataset](https://seafile.rlp.net/d/6f37b5ebaf4e492f8500/) for different program length and unzip in the root. The dataset is in the form of program expressions.
- How to create **Voxels** file from program expressions?

For generating voxel files for one_op
```python
python voxel_represenation_from_expressions.py --voxel_expression_path ./data/one_op/expressions_with_labels.csv --voxel_data_path ./data/one_op/voxel_representation
```

For generating voxel files for two_ops
```python
python voxel_represenation_from_expressions.py --voxel_expression_path ./data/two_ops/expressions_with_labels.csv --voxel_data_path ./data/two_ops/voxel_representation
```



## Supervised Learning

- To train, following flags can be set while running the training file:
    - model : Choose the model that needs to be trained. (basic_model, c3d)
    - --forward_approach : Choose the forward method for DecoderRNN. simple/teacher_forcer. (Mandatory argument)
    - --model_path : Path for saving the trained models.
    - --voxels_data_path : Path for .vox files used for training.
    - --voxels_label_path : Path for labels for each voxel file.
    - --embed_size : Dimension for word embeddings for decoder.
    - --hidden_size : Dimension for lstm hidden states.
    - --epochs : Number of epochs for the training.
    - --batch_size : Chossing the batch size for training.
    - --learning_rate : Learning rate for training.
    - --save_step : Choosing the number of steps after which trained model will be saved as a checkpoint.
    - --proportion : Choosing the proportion of dataset required for the model to be trained on.

By default the forward approach is set to 'teacher forcer'

```python
python main.py basic_model
```
