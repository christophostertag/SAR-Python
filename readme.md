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
The blor function is called for each image seperately. The input images normally have high values wherever there is a person or a tree. The blor function computes the geometric mean of an image and its offsets ((x offset, y offset):(0,0)(1,0)(0,1)(1,1)). The result is blurred using a Gaussian. Then the saturation is calculated. The saturation has very low values wherever there is a tree due to the arithmetic mean. But some pixels of the ground have high saturation values. Therefore to get a map, which (almost) only contains people the saturation is blurred using a Gaussian again and then multiplied with the original image and blurred using a Gaussian a third time.
* `get_detection_map()`  
First, a Gaussian filter is applied to the heatmaps from `get_temporal_diff_heatmaps()`.
Then, for each of the resulting 70 heapmaps, an individual threshold is computed:  
`thresh = 2 * np.nanquantile(heatmap, 0.98)`  
We then mark all pixels that exceed the threshold.
The sum of the resulting 70 binary maps is then stored in `detection_map`, which is of the shape `(1024, 1024)`.
* `find_clusters()`  
  Find clusters is a function to get all local maxima on a detection map as well as the corresponding intensity. It works using a simple flooding algorithm.
* `create_label_clusters()`  
This is function is called after removing all elements from the arrays returned by `find_clusters()` that are below a certain intensity threshold. The function uses the detection map together with the data from `find_cluster()` to aggregate all pixels which make up a cluster. A cluster is discarded if it has less than `100` pixels. The resulting cluster size is evaluated by keeping the `100` nearest pixels to the weighted average position of the cluster and those pixels which are above the threshold `cluster_intensity / 4` using the calculation `weights / (1 + np.sqrt(distance) * 0.2)`. The remaining pixels are discarded. Furthermore a cluster which overlaps with an existing cluster is assumed to be part of it and therefore discarded.   
* `find_boxes()`  
The label map containing the clusters is iterated and if a cluster is found, the bounding box of the cluster is evaluated by usin flooding again and taking the minimum and maximum positions.

## Discussion

This approach compares each timestep with the mean of the other timesteps.
Therefore, it can only detect moving objects or persons, but no person standing still or lying on the ground, for example. Since the temporal difference also includes noise such as moving leaves and warping inaccuracies, we apply a cascade of filters and computations to separate the persons from the noise. Thus the temporal difference algorithm strongly relies on a significant change in color or brightness when persons are moving. This essentially means that persons that are dressed in forest colors may not be detected.

# Other approaches

## Pixel clustering

We presented the pixel clustering algorithm in the second Lab, slide set S2.
This approach extracts features from pixels by aggregating the colors of the pixel's neighborhood. 
Then we create a feature table, where the rows are the pixels and the columns the extracted features. 
We optionally apply PCA and then we cluster using minibatch k-means. 
The pixels far from their cluster center are considered outliers.

This algorithm was able to segment most parts of the beacon, but no humans.
We did not develop this algorithm any further and switched to the temporal difference method explained above.

## Auto-Encoders

We presented the method in the second Lab, slide set S2.
This approach uses auto-encoders for image reconstruction to find anomalies using reconstruction error. This was done using different compression sizes to get different reconstruction error maps. In the end these reconstruction error maps were combined to get anomaly maps.

This method was discarded since while it may be usable for this kind of task it is difficult to balance, need more post processing on the result and conbsidering the similarity of our data set (mor or less always the same scene) may not generalise well.