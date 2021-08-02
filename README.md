# PyTorch-Image-Classification
PyTorch Image classification on ResNet,VGG,Mobilenet versions. This Repo also covers the Jetson Implementation


## Requirements

* Ubuntu (It's only tested on Ubuntu, so it may not work on Windows.)
* Python >= 3.7
* PyTorch >= 1.9.0
* torchvision
* Numpy
* Scipy

```bash
conda env create -f classifier/backup_env.yml
```

# Classifier
  Classifier contains the training script for the Image classification.
  
  # Dataset 
    Dataset should be structured like below,
    
    ```bash
       data
            train
               car
                 1.jpeg
               bike
                 1.jpeg
             test
                car
                 1.jpeg
                bike
                 1.jpeg
             eval
                car
                 1.jpeg
                bike
                 1.jpeg
     ```
    

Thanks To 


  1.https://github.com/dusty-nv
  
  
  2.https://github.com/anilsathyan7/pytorch-image-classification
  
  
  3.https://github.com/Ajithbalakrishnan :)
