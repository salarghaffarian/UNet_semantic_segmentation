# UNet_semantic_segmentation
UNet Semantic Segmentation: Training, Testing, and Deployment

### Dataset
(1) Dataset is downloaded from the kaggle datasets in this [Link](https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset?select=class_dict_seg.csv)
- There are 20 classes are labels for the semantic segmentation task.
- The classes are as the list below:
> tree, gras, other vegetation, dirt, gravel, rocks, water, paved area, pool, person, dog, car, bicycle, roof, wall, fence, fence-pole, window, door, obstacle


(2) 2nd dataset is download from the Pascal VOC2012 dataset using the code below.
> torchvision.datasets.VOCSegmentation(root='data/PascalVOC_Dataset', year='2012', image_set='train', download=True)