KAI STRAKA 09.2026

Small galaxy morphology image classification personal project.
Built in Python using PyTorch and mrJordi0/galaxy-zoo-dataset available through Hugging Face.
The model is a convolutional neural network (CNN) with 3 convolutional layers and 2 fully connected (FC) layers.

A custom function standardizes each image into a black and white 64x64 image, then converts all pixels into a tensor which is then used by the model.
Model architecture follows a simple flow:
    Input image -> Normalize image -> {Conv layer -> Batch normalization -> ReLU activation -> Max pooling 2x2} *3 -> Flattens tensor -> Fully connected layer 1 -> ReLU -> Fully connected layer 2 -> Prediction.

The convolutional layers and batches increase in sizes equally:
    Conv 1 : 32 channels / filters,
    Conv 2 : 64 channels / filters,
    Conv 3 : 128 channels / filters

Similarly, FC1 has 8192 input features and outputs 128 into FC2, which outputs 8 morphology classes, each corresponding to 1 of 8 Galaxy Zoo dataset classes.
The Galaxy Zoo dataset has 8 galaxy classes (0 -> 7) each indicating a different galaxy type. The model predicts by returning a list of 8 logits, and identifying the largest value. mrJordi0/galaxy-zoo-dataset architecture made this project much easier and was an amazing resource.

Other notable details:
    Activation function : ReLU,
    Optimizer : Adam,
    Learning rate : 0.0005,
    Weight decay : 0.0005,
    Loss function : Cross Entropy

Libraries used:
    Numpy,
    Pandas,
    PyTorch,
    datasets,
    datetime

The files work in a simple fashion and must be run in the order as follows: 
        GalModel.py : the model itself, it gets called by other files. DO NOT RUN IT DOES NOTHING.
        1 gal_train.py : the training loop. Select the amount of epochs to train (Line 22) and let it run. The trained model will be saved as a .pth file.
        2 gal_validation.py : validates the trained model on a separate validation subset and gives an accuracy and loss reading.
        3 gal_test.py : similar to validation, it runs the trained model on another separate subset and gives an accuracy and loss reading.
        4 gal_classifier.py : classifies images stored locally. Input the desired image path (Line 16) and let it run. A prediction and confidence will be printed.

Through many iterations, a final [testing] Accuracy of ~ 75.2 % and Loss of ~ 0.67 was achieved.
These values are not intended to be extraordinary, the goal of this project was to become familiar with CNNs and image classification work, and that goal was achieved.
Attached are some images and their respective predictions, note these images are from a completely separate dataset.


<img width="927" height="364" alt="galML examples" src="https://github.com/user-attachments/assets/27540927-d560-4eed-bc17-6b687479f9aa" />


Below is the cited Galaxy Zoo dataset.

@misc{Lin2021,
    author = {Lin, Joshua Yao-Yu and Liao, Song-Mao and Huang, Hung-Jin and Kuo, Wei-Ting and Ou, Olivia Hsuan-Min},
    title = {Galaxy Morphological Classification with Efficient Vision Transformer},
    year = {2021},
    eprint = {2110.01024},
    archivePrefix = {arXiv},
    primaryClass = {astro-ph.GA},
    url = {https://arxiv.org/abs/2110.01024}
}

@article{Walmsley_2022,
    author = {Walmsley, Mike and Lintott, Chris and Géron, Tobias and Kruk, Sandor and Krawczyk, Coleman and Willett, Kyle W. and Bamford, Steven and Dickinson, Hugh and Fortson, Lucy and Gal, Yarin and Keel, William and Masters, Karen L. and Mehta, Vihang and Simmons, Brooke D. and Smethurst, Rebecca and Baeten, Elisabeth M L and Macmillan, Christine},
    title = {Galaxy Zoo DECaLS: Detailed visual morphology measurements from volunteers and deep learning for 314,000 galaxies},
    journal = {Monthly Notices of the Royal Astronomical Society},
    volume = {509},
    number = {3},
    pages = {3966-3988},
    year = {2022},
    doi = {10.1093/mnras/stab2093},
    url = {https://doi.org/10.1093/mnras/stab2093}
}
