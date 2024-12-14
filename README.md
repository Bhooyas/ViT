# A Simple Vision Transformer

![ViT](https://socialify.git.ci/Bhooyas/ViT/image?font=KoHo&language=1&name=1&owner=1&pattern=Circuit%20Board&stargazers=1&theme=Auto)

A simple ViT for image classification

## Training the ViT

The first step would be to clone the project using the following command: -
```
git clone https://github.com/Bhooyas/ViT
```

The first step would be to clone the project using the following command: -
```
cd ViT
pip install -r requirements.txt
```

The next step is dataset specific: -

### Cifar 10

We can directly go ahead and run the following script to start training: -
```
python train_cifar10.py
```

### Tiny Imagenet

For Tiny ImageNet we first need to download the dataset. We can download it from this [link](https://cs231n.stanford.edu/tiny-imagenet-200.zip).

The next step is to convert this downloaded dataset to h5py. We achieve this using the following command: -
```
python create_h5.py
```

**Note**: - We might need to change image_folder in the `create_h5.py` file to the location where Tiny ImageNet dataset is stored.

There after we directly go ahead and train the model using the following command: -

```
python train_tinyimagenet.py
```
