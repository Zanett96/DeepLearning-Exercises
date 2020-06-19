###Import libraries 
import os
import copy
import torch
import matplotlib.pyplot as plt
import random
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import numpy as np
import scipy.io as io
from torchvision.utils import make_grid
from sklearn import manifold
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cv2

### Define paths
data_root_dir = '.\\'

#%% Create dataset + normalize data
train_transform = transforms.Compose([
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = MNIST(data_root_dir, train=True,  download=True, transform=train_transform)
train_data, validation_data = torch.utils.data.random_split(train_dataset, (50000, 10000))
test_dataset  = MNIST(data_root_dir, train=True, download=False, transform=test_transform)

### Define dataloader
train_dataloader = DataLoader(train_data, batch_size=512, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size=512, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False)


### Plot some sample
plt.close('all')
fig, axs = plt.subplots(5, 5, figsize=(8,8))
for ax in axs.flatten():
    img, label = random.choice(train_dataset)
    ax.imshow(img.squeeze().numpy(), cmap='gist_gray')
    ax.set_title('Label: %d' % label)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()


# Scatter with images instead of points
def imscatter(x, y, ax, imageData, zoom):
    images = []
    for i in range(len(x)):
        x0, y0 = x[i], y[i]
        # Convert to image
        img = imageData[i]*255.
        img = img.astype(np.uint8).reshape([28,28])
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        # Note: OpenCV uses BGR and plt uses RGB
        image = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(image, (x0, y0), xycoords='data', frameon=False)
        images.append(ax.add_artist(ab))
    
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()

# Show dataset images with T-sne projection of latent space encoding
def computeTSNEProjectionOfLatentSpace(X, model, display=True):
    # Compute latent space representation
    print("Computing latent space projection...")
    with torch.no_grad():
        X_encoded = model.encode(X)
    # Compute t-SNE embedding of latent space
    print("Computing t-SNE embedding...")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X_encoded)

    # Plot images according to t-sne embedding
    if display:
        print("Plotting t-SNE visualization...")
        fig, ax = plt.subplots(figsize=(20,10))
        imscatter(X_tsne[:, 0], X_tsne[:, 1], imageData=np.asarray(X), ax=ax, zoom=0.5)
        plt.show()
    else:
        return X_tsne
    
# Plot the losses over the number of epochs
def training_loss_plot(train_loss, test_loss):

    plt.close('all')
    plt.figure(figsize=(8,6))
    plt.semilogy(train_loss, label='Train loss')
    plt.semilogy(test_loss, label='Validation loss')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

# get the activation maps
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


## Random search over learning rate and the numbers of epochs
## Long time to process, excute only when researching hyperparameters!!!
def rand_search(model, test_dataloader, num_iters, weight_dec = True):
    
    # initialize some params
    loss_store, lr_store, epoch_store, encoded_dim = [], [], [], []
    best_lr, best_epochs, best = 0, 0, 0

    for i in range(num_iters):  
    
        ### Define a loss function
        loss_fn = torch.nn.MSELoss()
        
        #Set random parameters
        rand_lr = random.randint(1,5)*(10**(-random.randint(2,6)))
        rand_epoch = (random.randint(10,60))
        rand_encoded_space_dim = 10
        # Set lr
        if weight_dec: optim = torch.optim.Adam(model.parameters(), lr=rand_lr, weight_decay=1e-5) #set LR
        else: optim = torch.optim.Adam(model.parameters(), lr=rand_lr)
        
        #Initialize the network
        model = AutoencoderNet(encoded_space_dim=rand_encoded_space_dim)
        
        print("iteration: ", i, " lr: ", rand_lr, "and #epoch: ", rand_epoch) 
        #Compute the training
        for epoch in range(rand_epoch):
            train_epoch(model,  dataloader=test_dataloader, loss_fn=loss_fn, optimizer=optim)
            loss = test_epoch(model, dataloader=test_dataloader, loss_fn=loss_fn, optimizer=optim) 
        ##Store the parameters
        loss_store.append(loss)
        lr_store.append(rand_lr)
        epoch_store.append(rand_epoch)
    
        ## Update the best parameters
        if(i == 0): 
            best = loss
            print("The best loss so far is: ", best, " with lr: ", rand_lr, "and #epoch: ", rand_epoch)
        elif (loss < best):
            best_lr = rand_lr
            best_epochs = rand_epoch
            best = loss
            print("The best loss so far is: ", best, " with lr: ", rand_lr, "and #epoch: ", rand_epoch)
            
    print("The best loss is: ", best, " with lr: ", rand_lr, "and #epoch: ", rand_epoch)
    return loss_store, lr_store, epoch_store

## numbers of conbinations of hyperparameters to try
num_iters = 10
search = False
if search: loss_store, lr_store, epoch_store = rand_search(model, train_loader, num_iters)


## occlude an image with a dimxdim square placed randomly over the image
def random_occlusion(image, batch, dim=6):
    
    # random edges of the occlusion square
    idx = np.random.randint(0, 28-dim)
    idy = np.random.randint(0, 28-dim)
    
    # set the pixels to 0.0 (black)
    if batch: # set for batch of images
        for img in image:
            for i in range(idx, idx+dim):
                for j in range(idy, idy+dim):
                    img[i][j] = 0.0
    else: # set for only one img
        for i in range(idx, idx+dim):
                for j in range(idy, idy+dim):
                    image[i][j] = 0.0
            
    return image


### Training function
def train_epoch(net, dataloader, loss_fn, optimizer):
    
    # Tell it we are in training mode
    net.train()
    train_loss = []
    for images in dataloader:
        
        # Extract data and move tensors to the selected device
        image_batch = images[0].to(device)
        
        # Backward pass
        optim.zero_grad()
        
        # Forward pass
        output = net(image_batch)
        # reshape images in a suitable way for the loss calculation
        image_batch = image_batch.reshape([-1,1, 28, 28])

        ## loss calculation
        loss = loss_fn(output, image_batch)
        
        ## backward pass
        loss.backward()
        
        ## weight optimization
        optimizer.step()
        
        train_loss.append(loss.data)
    return np.mean(train_loss)


### Testing function
def test_epoch(net, dataloader, loss_fn, optimizer, corrupt, occlude):
    
    # Validation
    net.eval() # Evaluation mode (e.g. disable dropout)
    
    with torch.no_grad(): # No need to track the gradients
        
        conc_out = torch.Tensor().float()
        conc_label = torch.Tensor().float()
        
        for images in dataloader:
            
            # Extract data and move tensors to the selected device
            image_batch = images[0].to(device)
            
            # apply noise/occlude if needed
            if corrupt: image_batch = gaussian_noise(image_batch, 0.3)
            if occlude: image_batch = random_occlusion(image_batch.squeeze(), batch = True).unsqueeze(1)
                
            # Forward pass
            out = net(image_batch)
            # reshape
            image_batch = image_batch.reshape([-1,1, 28, 28])
            
            # Concatenate with previous outputs
            conc_out = torch.cat([conc_out, out.cpu()])
            conc_label = torch.cat([conc_label, image_batch.cpu()]) 
            
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data


## Apply Gaussian noise based on a noise intensity factor
def gaussian_noise(image, noise_factor):
    
    # torch.randn returns random values from a normal with mean 0 and variance 1 -> Gaussian noise!
    noisy_img = image + noise_factor * torch.randn(*image.shape)
    #clip the image
    noisy_img = np.clip(noisy_img, 0., 1.)
    
    return noisy_img

### Initialize the network
encoded_space_dim = 20
model = AutoencoderNet(encoded_space_dim=encoded_space_dim)

### Define loss function
loss_fn = torch.nn.MSELoss()

### Define optimizer
lr = 0.005 # Learning rate
optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

### If cuda is available set the device to GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
# Move all the network parameters to the selected device (if they are already on that device nothing happens)
model.to(device)


### Training cycle
num_epochs = 60 
plot = False
val_losses, train_losses = [], []

# training and validation 
for epoch in range(num_epochs):
    print('EPOCH %d/%d' % (epoch + 1, num_epochs))
    ### Training
    train_loss = train_epoch(model, dataloader=train_loader, loss_fn=loss_fn, optimizer=optim)
    ### Validation
    val_loss = test_epoch(model, dataloader=validation_loader, loss_fn=loss_fn, optimizer=optim, corrupt=False, occlude=False) 
    val_losses.append(val_loss)
    train_losses.append(train_loss) 
    
    # Print Validation loss
    print('\n\n\t VALIDATION - EPOCH %d/%d - loss: %f\n\n' % (epoch + 1, num_epochs, val_loss))
        
    if epoch % 1 == 0 and plot: #once every x epochs
        ### Plot progress
        model.eval()
        with torch.no_grad():
             rec_img  = model(test_dataset[25][0].unsqueeze(0))
        fig, axs = plt.subplots(1, 2, figsize=(6,3))
        axs[0].imshow(x_validation[2].squeeze(0).cpu().numpy(), cmap='gist_gray')
        axs[0].set_title('Original image')
        axs[1].imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
        axs[1].set_title('Reconstructed image (EPOCH %d)' % (epoch + 1))
        plt.tight_layout()
        # Save figures
        os.makedirs('autoencoder_progress_%d_features' % encoded_space_dim, exist_ok=True)
        plt.savefig('autoencoder_progress_%d_features/epoch_%d.png' % (encoded_space_dim, epoch + 1))
        plt.pause(0.1)
        #plt.show()
        #plt.close()
        
        # early stopping if validation loss doesn't improves over the last 8 epochs
        if epoch > 1 and np.mean(val_losses[epoch-15:epoch-1]) < val_loss: break


## Save the model of the network in the folder of the data
torch.save(model.state_dict(), '.\\'+ 'netModel.txt')

# plot the losses over the epochs
training_loss_plot(train_losses, val_losses)

# test the network over the unseen test_set
test_loss = []
test_loss = test_epoch(model, dataloader=test_dataloader, loss_fn=loss_fn, optimizer=optim, corrupt=False, occlude=Fal
print("The reconstruction error over the test set is ", test_loss)

# extract one image to use for actvations and noise/occlusion trials
img = test_dataset[25][0].unsqueeze(0)


## visualize kernels of the first CNN
kernels = model.encoder_cnn[0].weight.detach().clone() 
kernels = kernels - kernels.min()
kernels = kernels / kernels.max()
img = make_grid(kernels)
plt.imshow(img.permute(1, 2, 0))


## Visualize activation function of the first convolutional encoder layer 

model.encoder_cnn[0].register_forward_hook(get_activation('ext_conv1'))

output = model(img)
act = activation['ext_conv1'].squeeze()

num_plot = 10
fig, axarr = plt.subplots(1, 10, figsize=(15,5))
for idx in range(min(act.size(0), num_plot)):
    axarr[idx].imshow(act[idx])


## Visualize activation function of the second convolutional encoder layer 

model.encoder_cnn[2].register_forward_hook(get_activation('ext_conv2'))

output = model(img)
act = activation['ext_conv2'].squeeze()

num_plot = 10
fig, axarr = plt.subplots(1, 10, figsize=(15,5))
for idx in range(min(act.size(0), num_plot)):
    axarr[idx].imshow(act[idx])


## Visualize activation function of the third convolutional encoder layer 

model.encoder_cnn[4].register_forward_hook(get_activation('ext_conv4'))

output = model(img)

activ = activation['ext_conv4'].squeeze()

num_plot = 10
fig, axarr = plt.subplots(1, 10, figsize=(15,5))
for idx in range(min(activ.size(0), num_plot)):
    axarr[idx].imshow(activ[idx])

## Plot gaussian noise with various intensity
original_imgs,corr_imgs, rec_imgs  = [], [], []

# gradually apply noise and reconstruct images
for i in range (0,10):
                       
    imgc = test_dataset[25][0].unsqueeze(0)
    corr_imgs.append(gaussian_noise(img, 0.1*i)) 
    model.eval()
    
    with torch.no_grad():
        rec_imgs.append(model(corr_imgs[i]))

# plot the corrupted input image and then reconstructed images
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))
# input images on top row, reconstructions on bottom
for corr_imgs, row in zip([corr_imgs, rec_imgs], axes):
    for img, ax in zip(corr_imgs, row):
        
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

img_occl, rec_imgs = [], []

# occlude 10 images and reconstruct images
for i in range (1,10):
    
    imgc = test_dataset[25][0].unsqueeze(0)
    img_occl.append(random_occlusion(img0.squeeze(), batch=False)) 
    
    model.eval()
    with torch.no_grad():
        rec_imgs.append(model(img_occl[i-1].unsqueeze(0).unsqueeze(1)))


# plot the occluded input images and then reconstructed images
fig, axes = plt.subplots(nrows=2, ncols=9, sharex=True, sharey=True, figsize=(25,4))
# input images on top row, reconstructions on bottom
for img_occl, row in zip([img_occl, rec_imgs], axes):
    for img, ax in zip(img_occl, row):
        
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

                       
tsne_set = []
for i, images in enumerate(test_dataset):   
    # Extract data and reshape to numpy
    tsne_set.append(images[0].numpy())
    if i == 3000: break
tsne_set = torch.FloatTensor(tsne_set) #reshape to tensor

#plot the tsne of the embeddings
computeTSNEProjectionOfLatentSpace(tsne_set, model) 
