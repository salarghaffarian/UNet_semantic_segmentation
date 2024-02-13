from torchvision import datasets, transforms

def download_VOC2012(root='data/PascalVOC_Dataset'):
    """Download the Pascal VOC2012 dataset and store the data in the data/"""
    datasets.VOCSegmentation(root, year='2012', image_set='train', download=True)
    datasets.VOCSegmentation(root, year='2012', image_set='val', download=True)                  
    datasets.VOCSegmentation(root, year='2012', image_set='trainval', download=True)


if __name__ == '__main__':
    # TODO: Reseach to see if this dataset can be used for the semantic segmenatation task.
    # TODO: How many classes are there in the VOC2012 dataset?

    download_VOC2012()