# %% IMPORTS
from sklearn.ensemble import RandomForestClassifier
from skimage.io import imread, imshow
import numpy as np
import napari
from skimage import data, filters
import matplotlib.pyplot as plt
# %% LOAD AND ANNOTATE THE IMAGE
# Load the cells3D image:
image = data.cells3d()

# extract the nuclei channel and z-project:
image_2D = image[:, 1, ...].max(axis=0)

# interactively label the nuclei:
# start napari and add image:
viewer = napari.Viewer()
viewer.add_image(image_2D)
# add an empty labels layer:
labels = viewer.add_labels(np.zeros(image_2D.shape).astype(int))

"""
In Napari, we can use the labels layer to interactively label the nuclei. Label 
some nuclei (label 2) and background pixels (label 1) in the label layer.

When you're done, execute the next cell.
"""
# %% VIEW ANNONTATIONS
# take a screenshot of the annotation:
napari.utils.nbscreenshot(viewer)

# retrieve the annotations from the napari layer:
annotations = labels.data
# plot the original image and the annotations side-by-side in a subplot:
fig, axes = plt.subplots(ncols=2, figsize=(8, 4))
axes[0].imshow(image_2D)
axes[0].set_title('Original image')
axes[1].imshow(annotations)
axes[1].set_title('Annotations')
plt.show()
# %% GENERATE IMAGE FEATURE STACK
"""
We now increase the number of features, that we can take out of our image. Our image
will become a 3D stack of 2D images, where each 2D image is a feature, consisting of

* the original pixel value
* the pixel value after a Gaussian blur (=denoising)
* the pixel value of the Gaussian blurred image processed through a Sobel operator (=edge detection)
"""
def generate_feature_stack(image):
    # determine features
    blurred = filters.gaussian(image, sigma=2)
    edges = filters.sobel(blurred)

    """
    Collect features in a stack. The ravel() function turns a nD image into 
    a 1-D image. We need to use it because scikit-learn expects values in a 
    1-D format here. 
    """
    feature_stack = [image.ravel(),
                     blurred.ravel(),
                     edges.ravel()]
    
    return np.asarray(feature_stack)

feature_stack = generate_feature_stack(image_2D)

# show feature images:
fig, ax = plt.subplots(1, 3, figsize=(10,10))
# reshape(image.shape) is the opposite of ravel() here. We just need it for visualization.
ax[0].imshow(feature_stack[0].reshape(image_2D.shape), cmap=plt.cm.gray)
ax[0].set_title('Original image')
ax[1].imshow(feature_stack[1].reshape(image_2D.shape), cmap=plt.cm.gray)
ax[1].set_title('Blurred image')
ax[2].imshow(feature_stack[2].reshape(image_2D.shape), cmap=plt.cm.gray)
ax[2].set_title('Edges')
plt.show()
# %% FORMATTING DATA
"""We now need to format the input data so that it fits to what scikit learn expects. 
Scikit-learn asks for an array of shape (n, m) as input data and (n) annotations. 
n corresponds to number of pixels and m to number of features. In our case m = 3.
"""

def format_data(feature_stack, annotation):
    # reformat the data to match what scikit-learn expects
    # transpose the feature stack
    X = feature_stack.T
    # make the annotation 1-dimensional
    y = annotation.ravel()
    
    # remove all pixels from the feature and annotations which have not been annotated
    mask = y > 0
    X = X[mask]
    y = y[mask]

    return X, y

X, y = format_data(feature_stack, annotations)

print("input shape", X.shape)
print("annotation shape", y.shape)
# %% TRAIN AND PREDICT WITH RANDOM FOREST CLASSIFIER
classifier = RandomForestClassifier(max_depth=10, random_state=0)
classifier.fit(X, y)
RandomForestClassifier(max_depth=2, random_state=0)

result = classifier.predict(feature_stack.T) - 1 # we subtract 1 to make background = 0
result_2d = result.reshape(image_2D.shape)
imshow(result.reshape(image_2D.shape))
viewer.add_labels(result_2d)
napari.utils.nbscreenshot(viewer)
# %% SECOND EXAMPLE (SKIN): LOAD AND ANNOTATE
image_2d_2 = data.skin()[:,:,0]

viewer = napari.Viewer()
viewer.add_image(image_2d_2)
# add an empty labels layer:
labels = viewer.add_labels(np.zeros(image_2d_2.shape).astype(int))
# %% SECOND EXAMPLE (SKIN): VIEW ANNONTATIONS
# take a screenshot of the annotation:
napari.utils.nbscreenshot(viewer)

# retrieve the annotations from the napari layer:
annotations = labels.data
# plot the original image and the annotations side-by-side in a subplot:
fig, axes = plt.subplots(ncols=2, figsize=(8, 4))
axes[0].imshow(image_2d_2)
axes[0].set_title('Original image')
axes[1].imshow(annotations)
axes[1].set_title('Annotations')
plt.show()
# %% GENERATE IMAGE FEATURE STACK
feature_stack = generate_feature_stack(image_2d_2)

# show feature images:
fig, ax = plt.subplots(1, 3, figsize=(10,10))
# reshape(image.shape) is the opposite of ravel() here. We just need it for visualization.
ax[0].imshow(feature_stack[0].reshape(image_2d_2.shape), cmap=plt.cm.gray)
ax[0].set_title('Original image')
ax[1].imshow(feature_stack[1].reshape(image_2d_2.shape), cmap=plt.cm.gray)
ax[1].set_title('Blurred image')
ax[2].imshow(feature_stack[2].reshape(image_2d_2.shape), cmap=plt.cm.gray)
ax[2].set_title('Edges')
plt.show()
# %% FORMATTING DATA

X, y = format_data(feature_stack, annotations)

print("input shape", X.shape)
print("annotation shape", y.shape)
# %% TRAIN AND PREDICT WITH RANDOM FOREST CLASSIFIER
classifier = RandomForestClassifier(max_depth=10, random_state=0, max_samples=0.05, n_estimators=50)
classifier.fit(X, y)
RandomForestClassifier(max_depth=2, random_state=0)

result = classifier.predict(feature_stack.T) - 1 # we subtract 1 to make background = 0
result_2d = result.reshape(image_2d_2.shape)
imshow(result.reshape(image_2d_2.shape))
viewer.add_labels(result_2d)
napari.utils.nbscreenshot(viewer)
# %% END