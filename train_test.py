import os

import time, os
os.environ["TZ"] = "US/Eastern"
time.tzset()

from network_visualization import *
from utils import *

import torch
import torchvision
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

path = os.getcwd()
PATH = os.path.join(path)

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

if torch.cuda.is_available:
  print('Good to go!')
else:
  print('Please set GPU via Edit -> Notebook Settings.')

print('Download and load the pretrained SqueezeNet model.')
model = torchvision.models.squeezenet1_1(pretrained=True).to(device='cuda')

# We don't want the pretrained model to be retrained, so tell PyTorch not to compute gradients
# with respect to model parameters.
for param in model.parameters():
  param.requires_grad = False
    
# Make sure the model is in "eval" mode
model.eval()

# you may see warning regarding initialization deprecated, that's fine, please continue to next steps

# download imagenet_val
if os.path.isfile('imagenet_val_25.npz'):
  print('ImageNet val images exist')
else:
  print('download ImageNet val images by using next line in the code')
#   !wget http://web.eecs.umich.edu/~justincj/teaching/eecs498/imagenet_val_25.npz -P ./datasets/

X, y, class_names = load_imagenet_val(num=5, path='./datasets/imagenet_val_25.npz')

plt.figure(figsize=(12, 6))
for i in range(5):
  plt.subplot(1, 5, i + 1)
  plt.imshow(X[i])
  plt.title(class_names[y[i]])
  plt.axis('off')
plt.gcf().tight_layout()

def show_saliency_maps(X, y):
  # Convert X and y from numpy arrays to Torch Tensors
  X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0).to(device='cuda')
  y_tensor = torch.tensor(y, device='cuda')

  # YOUR_TURN: Impelement the compute_saliency_maps function
  saliency = compute_saliency_maps(X_tensor, y_tensor, model)

  # Convert the saliency map from Torch Tensor to numpy array and show images
  # and saliency maps together.
  saliency = saliency.to('cpu').numpy()
  N = X.shape[0]
  for i in range(N):
    plt.subplot(2, N, i + 1)
    plt.imshow(X[i])
    plt.axis('off')
    plt.title(class_names[y[i]])
    plt.subplot(2, N, N + i + 1)
    plt.imshow(saliency[i], cmap=plt.cm.hot)
    plt.axis('off')
    plt.gcf().set_size_inches(12, 5)
  plt.savefig(os.path.join(PATH,'saliency_maps_results.jpg'))
  plt.show()

show_saliency_maps(X, y)

idx = 0
target_y = 6

X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0).to(device='cuda')
print('Print your progress using the following format: the model is fooled if the target score and max score are the same.')
print('Iteration %d: target score %.3f, max score %.3f')
# YOUR_TURN: Impelement the make_adversarial_attack function
X_adv = make_adversarial_attack(X_tensor[idx:idx+1], target_y, model, max_iter=100)

scores = model(X_adv)
assert target_y == scores.data.max(1)[1][0].item(), 'The model is not fooled!'

from utils import deprocess

X_adv = X_adv.to('cpu')
X_adv_np = deprocess(X_adv.clone())
X_adv_np = np.asarray(X_adv_np).astype(np.uint8)

plt.subplot(1, 4, 1)
plt.imshow(X[idx])
plt.title(class_names[y[idx]])
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(X_adv_np)
plt.title(class_names[target_y])
plt.axis('off')

plt.subplot(1, 4, 3)
X_pre = preprocess(Image.fromarray(X[idx]))
diff = np.asarray(deprocess(X_adv - X_pre, should_rescale=False))
plt.imshow(diff)
plt.title('Difference')
plt.axis('off')

plt.subplot(1, 4, 4)
diff = np.asarray(deprocess(10 * (X_adv - X_pre), should_rescale=False))
plt.imshow(diff)
plt.title('Magnified difference (10x)')
plt.axis('off')

plt.gcf().set_size_inches(12, 5)
plt.savefig(os.path.join(PATH,'adversarial_attacks_results.jpg'))
plt.show()

def create_class_visualization(target_y, model, class_names, device='cpu', save_fig=False, **kwargs):
  """
  Generate an image to maximize the score of target_y under a pretrained model.
  
  Inputs:
  - target_y: Integer in the range [0, 1000) giving the index of the class
  - model: A pretrained CNN that will be used to generate the image
  - class_names: Dictionary for class names
  - save_fig: saves the final figure for submission
  - device: 'cpu' or 'gpu'
  
  Keyword arguments:
  - num_iterations: How many iterations to use
  - blur_every: How often to blur the image as an implicit regularizer
  - max_jitter: How much to jitter the image as an implicit regularizer
  - show_every: How often to show the intermediate result
  """
  num_iterations = kwargs.pop('num_iterations', 100)
  blur_every = kwargs.pop('blur_every', 10)
  max_jitter = kwargs.pop('max_jitter', 16)
  show_every = kwargs.pop('show_every', 25)

  # Randomly initialize the image as a PyTorch Tensor, and make it requires gradient.
  img = torch.randn((1, 3, 224, 224), device=device).mul_(1.0).requires_grad_()

  for t in range(num_iterations):
    # Randomly jitter the image a bit; this gives slightly nicer results
    ox, oy = random.randint(0, max_jitter), random.randint(0, max_jitter)
    img.data.copy_(jitter(img.data, ox, oy))

    # YOUR_TURN: Impelement the create_class_visualization function to perform 
    # gradient step
    img = class_visualization_step(img, target_y, model) 
    
    # Undo the random jitter
    img.data.copy_(jitter(img.data, -ox, -oy))
    # As regularizer, clamp and periodically blur the image
    for c in range(3):
      lo = float(-SQUEEZENET_MEAN[c] / SQUEEZENET_STD[c])
      hi = float((1.0 - SQUEEZENET_MEAN[c]) / SQUEEZENET_STD[c])
      img.data[:, c].clamp_(min=lo, max=hi)
    if t % blur_every == 0:
      blur_image(img.data, sigma=0.5)

    # Periodically show the image
    if t == 0 or (t + 1) % show_every == 0 or t == num_iterations - 1:
      plt.imshow(deprocess(img.data.clone().cpu()))
      class_name = class_names[target_y]
      plt.title('%s\nIteration %d / %d' % (class_name, t + 1, num_iterations))
      plt.gcf().set_size_inches(4, 4)
      plt.axis('off')
      if save_fig:
        plt.savefig(os.path.join(PATH,'class_viz_result.jpg'))
      plt.show()

  return deprocess(img.data.cpu())

target_y = 76 # Tarantula
# target_y = 78 # Tick
# target_y = 187 # Yorkshire Terrier
# target_y = 683 # Oboe
# target_y = 366 # Gorilla
# target_y = 604 # Hourglass
# YOUR_TURN: make sure you have implemented the class_visualization_step function
out = create_class_visualization(target_y, model, class_names, save_fig=True, device='cuda')

# target_y = 78 # Tick
# target_y = 187 # Yorkshire Terrier
# target_y = 683 # Oboe
# target_y = 366 # Gorilla
# target_y = 604 # Hourglass
target_y = random.randint(0,999) # [0,999]
print(class_names[target_y])
out = create_class_visualization(target_y, model, class_names, save_fig=False, device='cuda')