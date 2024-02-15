# UNet_semantic_segmentation
UNet Semantic Segmentation: Training, Testing, and Deployment

### Dataset
(1) Dataset is downloaded from the kaggle datasets in this [Link](https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset?select=class_dict_seg.csv)
- There are 20 classes are labels for the semantic segmentation task.
- The classes are as the list below:
> tree, gras, other vegetation, dirt, gravel, rocks, water, paved area, pool, person, dog, car, bicycle, roof, wall, fence, fence-pole, window, door, obstacle


(2) 2nd dataset is download from the Pascal VOC2012 dataset using the code below.
```python
import torchvision
torchvision.datasets.VOCSegmentation(root='data/PascalVOC_Dataset', year='2012', image_set='train', download=True)
```

### Installation

- To use this project, you'll need to set up a Python environment and install the required dependencies. Below are the steps to get started:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/UNet_semantic_segmentation.git
   cd UNet_semantic_segmentation
   ```

2. **Create a virtual Environment: (Optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: .\venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Download Datasets:**
    Download the Datasets from the websites explained in the Dataset section above.