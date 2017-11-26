# Brain tumor segmentation using U-Net based Fully Convolutional Networks and ExtraTrees classifier

### About the method
We developed a disriminative model for brain tumor segmentation from multimodal MRI protocols. Our model uses U-Net based fully convolutional networks to extract features from multimodal MRI training dataset and then applies to extremely randomized trees (ExtraTrees) classifier for segmenting the brain tumor. The method was evaluated on the Brain Tumor Segmentation Challenge 2013 (BRATS 2013) dataset, which have 20 HGG and 10 LGG volumes.

We train HGG and LGG together using 4 scanning image: FLAIR, T1, T1c, T2 of BRATS 2013 dataset. 

### Segmentation result
![segmentation](https://github.com/haile493/Brain-tumor-segmentation-ExtraTrees/blob/master/images/segmentation.png)

#### Note: If you want to run these codes, you need to train U-Net yourself
