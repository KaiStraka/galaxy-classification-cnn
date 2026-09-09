Kai Straka 2026
Galaxy morphology image classification personal project.
Built in Python using PyTorch and MrJordi0 Galaxy Zoo dataset available through Hugging Face.
The model is a convolutional neural network (CNN) with 3 convolutional layers and 2 fully connected (FC) layers.

A custom function standardizes each image into a black and white 64x64 image, then converts all pixels into a list with corresponding pixel strengths.
Model architecture follows a simple flow:
    Input Image -> {Conv layer -> Batch normalization -> ReLU activation -> Max pooling 2x2} *3 -> Fully connected layer 1 + ReLU -> Fully connected layer 2 -> Prediction.

The convolutional layers and batches increase in sizes equally:
    Conv 1 : 32 batch size
    Conv 2 : 64 batch size
    Conv 3 : 128 batch size

Similarly, FC1 has 8192 input features and outputs 128 into FC2, which outputs 8 possible answers, each corresponding to 1 of 8 Galaxy Zoo dataset types.

Other notable details:
    Optimizer : Adam
    Learning rate : 0.0005
    Weight decay : 0.0005
    Loss function : Cross Entropy

Through many iterations, a final Accuracy of ~75.2 % and Loss of ~ 0.67 was achieved.
These values are sub optimal, but the goal of this project was to become familiar with CNNs and image classification work, and that goal has been achieved.

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
