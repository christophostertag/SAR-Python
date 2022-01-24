# Detecting persons using temporal difference

<span style="font-size:larger;">
Computer Vision Lab, W2021  
</span>
<br>
<span style="font-size:larger;">
Group D1  
</span>

* Aliaksandr Kahadouski
* Christoph Ostertag
* Simeon David Quant
* Markus Steindl

## Configuration

In order to run the project, configure your paths in `common.py`:
```
class Paths:
    data = Path(__file__).parent / 'data'

    data_train = data / 'train'
    data_test = data / 'test'
    validation_labels = data / 'validation/labels.json'

    output = Path(__file__).parent / 'output'
```


## Dataloader

The function `get_image_sets` iterates over sets of 70 images in the specified directory tree.
```
common.get_image_sets(
        dir=Paths.data,
        filter='train',
        mask=True,
        warp=True,
        timesteps=range(-3, 4),
        views=range(10),
        cropx=0,
        size_filter: Optional[Tuple[int]] = (1024, 1024, 3),
        sets=1,
)
```
We can choose whether to apply the mask, warp, or crop.
The `size_filter` makes sure only images of the specified size will be loaded.
The `sets` parameter limits the number of sets returned. E.g. if `sets == 2`, then the function yields two sets of 70 images each.

## Detection using temporal difference

```
temporal_diff.temporal_diff.main(
        draw_boxes=True,
        draw_ourboxes=True,
        show=True,
        dataset='val',  # 'val' or 'test'
        skip=0,
)
```
This function loads images, computes bounding boxes, and writes the following to the file system:
* `val.json` bounding boxes
* `test.json` bounding boxes
* intermediate heatmaps
* images with drawn boxes and segmented pixels

The `main()` function computes results for each set of 70 images (10 views and 7 timesteps).  
These computations are exectued:

* `get_temporal_diff_heatmaps()`  
For each timestep, we subtract the mean of all the other timesteps. 
This produces 70 color difference maps.
Each of these is then blurred to get rid of moving leaves and warping inaccuracies.
We originally used a simple Gaussian filter for blurring. However, this was replaced by the more sophisticated blurring algorithm `blor()`, which is explained below.
The blurred color difference maps are then reduced to one channel by computing the L2-norm of the color difference triples. The shape of the resulting heatmap tensor is `(7, 10, 1024, 1024)`.
* `blor()`  
  TODO Simeon
* `get_detection_map()`  
First, a Gaussian filter is applied to the heatmaps from `get_temporal_diff_heatmaps()`.
Then, for each of the resulting 70 heapmaps, an individual threshold is computed:  
`thresh = 2 * np.quantile(heatmap, 95)`  
We then mark all pixels that exceed the threshold.
The sum of the resulting 70 binary maps is then stored in `detection_map`, which is of the shape `(1024, 1024)`.
* `find_clusters()`  
  TODO Simeon
* `find_boxes()`
  TODO Simeon
* `keep_top_n()` TODO Simeon and rename

## Discussion

This approach compares each timestep with the mean of the other timesteps.
Therefore, it can only detect moving objects or persons, but no person standing still or lying on the ground, for example. Since the temporal difference also includes noise such as moving leaves and warping inaccuracies, we apply a cascade of filters and computations to separate the persons from the noise. Thus the temporal difference algorithm strongly relies on a significant change in color or brightness when persons are moving. This essentially means that persons that are dressed in forest colors may not be detected.

# Other approaches

## Pixel clustering

We presented the pixel clustering algorithm in the secend Lab, slide set S2.
This approach extracts features from pixels by aggregating the colors of the pixel's neighborhood. 
Then we create a feature table, where the rows are the pixels and the columns the extracted features. 
We optionally apply PCA and then we cluster using minibatch k-means. 
The pixels far from their cluster center are considered outliers.

This algorithm was able to segment most parts of the beacon, but no humans.
We did not develop this algorithm any further and switched to the temporal difference method explained above.
