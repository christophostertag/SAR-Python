from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image
from PIL import ImageColor
from matplotlib import pyplot
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from common import get_centered_rectangle_slice, load_image_batches, load_mask, crop190, select_bounding_boxes, \
    draw_bounding_boxes, load_bounding_boxes
from conf import Paths


class FeatureExtractor:
    def __init__(
            self,
            color_center=True,
            color_mean=True,
            color_std=False,
            color_median=True,
            color_p25=True,
            color_p75=True,
    ):
        self.color_center = color_center
        self.color_mean = color_mean
        self.color_std = color_std
        self.color_median = color_median
        self.color_p25 = color_p25
        self.color_p75 = color_p75

    def get_features(
            self,
            snippets: np.ndarray,
    ) -> np.ndarray:
        snippet_axes = (-3, -2)

        features = []
        dtype = np.float32

        if self.color_center:
            snippet_size = np.array(snippets.shape)[np.array(snippet_axes)]
            center_x, center_y = (snippet_size - 1) // 2
            slices = [slice(None) for _ in snippets.shape]
            slices[snippet_axes[0]] = center_x
            slices[snippet_axes[1]] = center_y

            feature = snippets[tuple(slices)].astype(dtype)
            features.append(feature)

        if self.color_mean:
            feature = np.mean(snippets, dtype=dtype, axis=snippet_axes)
            features.append(feature)

        if self.color_std:
            feature = np.std(snippets, dtype=dtype, axis=snippet_axes)
            features.append(feature)

        if self.color_median:
            feature = np.median(snippets, axis=snippet_axes).astype(dtype)
            features.append(feature)

        if self.color_p25:
            feature = np.percentile(snippets, 0.25, axis=snippet_axes).astype(dtype)
            features.append(feature)

        if self.color_p75:
            feature = np.percentile(snippets, 0.75, axis=snippet_axes).astype(dtype)
            features.append(feature)

        features = np.concatenate(features, axis=-1)

        return features


def get_snippet_features(
        images: np.ndarray,
        feature_extractor: Callable,
        snip_size=5,
) -> np.ndarray:
    n, lx, ly, _ = images.shape
    image_idx = np.arange(n)

    h = (snip_size - 1) // 2
    features = []
    index = []
    for i in range(h, lx - h):
        features_i = []
        for j in range(h, ly - h):
            features_j = images[:, i - h:i + h + 1, j - h:j + h + 1]
            # features_j = feature_extractor(features_j)
            features_i.append(features_j)

        features_i = feature_extractor(np.array(features_i))
        features.append(features_i)

    features = np.array(features)
    features = np.moveaxis(features, 2, 0)
    features = np.nan_to_num(features, nan=0)
    return features


def get_rgb_colors(n: int = 10) -> np.array:
    colors = pyplot.rcParams['axes.prop_cycle'].by_key()['color']
    rgb_colors = [ImageColor.getrgb(c) for c in colors]

    missing = n - len(rgb_colors)
    if missing > 0:
        rgb_colors += [rgb_colors[-1]] * missing

    rgb_colors = np.array(rgb_colors)

    return rgb_colors[:n]


def argsort_labels_by_occ(labels: np.ndarray) -> np.ndarray:
    mapping = np.argsort(-np.bincount(labels)).astype(labels.dtype)
    # mapping = np.argsort(mapping).astype(labels.dtype)
    return mapping


def apply_centered_overlay(
        images: np.ndarray,
        overlays: np.ndarray,
        alpha: int = 0.4,
        # black_margin=True,
):
    overlay_images = images.copy()

    overlay_images_selection = get_centered_rectangle_slice(overlay_images, overlays.shape[:3])
    overlay_images_selection[:] = ((1 - alpha) * overlay_images_selection + alpha * overlays).astype(np.uint8)

    return overlay_images


def apply_matrix_operation_to_tensor(
        tensor: np.ndarray,
        operation: Callable,
        first_feature_axis=-1,
):
    sample_shape = tensor.shape[:first_feature_axis]
    tensor = tensor.reshape(np.product(sample_shape), -1)
    tensor = operation(tensor)
    if tensor is not None:
        tensor = tensor.reshape(*sample_shape, -1)
    return tensor


class Checkpoints(Enum):
    none = 0
    features = 1
    pca = 2
    clustering = 3


def main(
        image_path=Paths.data,
        output_path_template=Paths.output / 'clusters={clusters}_pca={pca}',
        data='validation',

        n_images=140,
        n_clusters=15,
        dbscan_eps=.5,
        pca_components: Optional[int] = 6,

        extract_color_center=True,
        extract_color_mean=True,
        extract_color_std=True,
        extract_color_median=False,
        extract_color_q1=False,
        extract_color_q3=False,
):
    load_checkpoint = Checkpoints.none

    # feature_method_name = f'{data}_median_q1_q3'
    feature_method_name = f'{data}{n_images}_center_mean_std'
    cluster_method_name = f'{n_clusters}kmeans'
    # cluster_method_name =  f'DBSCAN'
    cluster_method_name = (f'pca{pca_components}_' if pca_components is not None else '') + cluster_method_name

    feature_output = Paths.output / feature_method_name
    cluster_output = feature_output / cluster_method_name

    cluster_output.mkdir(parents=True, exist_ok=True)

    f_path = feature_output / 'features.npy'
    pca_path = feature_output / f'features_pca{pca_components}.npy'
    pixel_label_path = cluster_output / 'pixel_labels.npy'
    cluster_center_path = cluster_output / 'cluster_centers.npy'
    cd_path = cluster_output / 'center_distances.npy'

    cluster_model = MiniBatchKMeans(n_clusters)
    # cluster_model = DBSCAN(min_samples=50, eps=0.1)

    ###

    image_batches = load_image_batches(path=image_path, batch_size=n_images, filter=data)
    bounding_boxes = load_bounding_boxes()
    images, rel_image_paths = next(image_batches)
    rel_image_paths = [p.relative_to(image_path) for p in rel_image_paths]

    # mask and crop
    image_mask = crop190(load_mask())
    images = crop190(images) * image_mask

    if load_checkpoint.value == Checkpoints.features.value:
        features = np.load(f_path)
    elif load_checkpoint.value < Checkpoints.features.value:
        print('Processing features... ', end='')
        feature_extractor = FeatureExtractor(
            color_center=extract_color_center,
            color_mean=extract_color_mean,
            color_std=extract_color_std,
            color_median=extract_color_median,
            color_p25=extract_color_q1,
            color_p75=extract_color_q3,
        )
        features = get_snippet_features(images, feature_extractor.get_features)
        print('done')
        np.save(f_path, features)
        print(f'Stored features: {f_path}')

    if load_checkpoint.value == Checkpoints.pca.value:
        features = np.load(pca_path)
    elif load_checkpoint.value < Checkpoints.pca.value and pca_components is not None:
        scaler = MinMaxScaler()
        features = apply_matrix_operation_to_tensor(features, scaler.fit_transform)

        print('Applying PCA... ', end='')
        pca = PCA(n_components=pca_components)
        features = apply_matrix_operation_to_tensor(features, pca.fit_transform).astype(np.float32)
        np.save(pca_path, features)
        np.cumsum(pca.explained_variance_ratio_)
        print('done')

    if load_checkpoint.value == Checkpoints.clustering.value:
        pixel_labels = np.load(pixel_label_path)
        cluster_centers = np.load(cluster_center_path)
        all_center_distances = np.load(cd_path)
        center_distances = all_center_distances.min(-1)
    elif load_checkpoint.value < Checkpoints.clustering.value:
        print('Clustering... ', end='')
        scaler2 = MinMaxScaler()
        features = apply_matrix_operation_to_tensor(features, scaler2.fit_transform)

        # cluster_model = DBSCAN(min_samples=50, eps=0.1)
        # cluster_model = OPTICS(min_samples=5, cluster_method='dbscan')
        # cluster_model = OPTICS(min_samples=100e-5, xi=0.05, n_jobs=-1)
        # x = features.reshape(-1, features.shape[-1])
        # selected_idx = np.arange(len(x))
        # selected_idx = np.random.choice(selected_idx, size=100000, replace=False)
        # x = x[selected_idx]
        # cluster_model.fit(x)
        # # pd.Series(Counter(y))
        #
        # labels_050 = cluster_optics_dbscan(
        #     reachability=cluster_model.reachability_,
        #     core_distances=cluster_model.core_distances_,
        #     ordering=cluster_model.ordering_,
        #     eps=0.5,
        # )
        # labels_010 = cluster_optics_dbscan(
        #     reachability=cluster_model.reachability_,
        #     core_distances=cluster_model.core_distances_,
        #     ordering=cluster_model.ordering_,
        #     eps=0.05,
        # )
        # z = pd.Series(Counter(labels_010))

        all_center_distances = apply_matrix_operation_to_tensor(features, cluster_model.fit_transform)
        center_distances = all_center_distances.min(-1)
        pixel_labels = cluster_model.labels_.astype(np.uint8)

        # order labels by highest occurrence
        if True:
            sort_idx = argsort_labels_by_occ(pixel_labels)
            all_center_distances = all_center_distances[..., sort_idx]
            pixel_labels = np.argsort(sort_idx).astype(pixel_labels.dtype)[pixel_labels]

        # reshape labels to image shape
        pixel_labels = pixel_labels.reshape(features.shape[:-1])
        # np.sum(all_center_distances.argmin(-1) != pixel_labels)

        try:
            cluster_centers = np.array(cluster_model.cluster_centers_)
            cluster_centers = cluster_centers[sort_idx]
        except:
            cluster_centers = None

        np.save(pixel_label_path, pixel_labels)
        if cluster_centers is not None:
            np.save(cluster_center_path, cluster_centers)
        np.save(cd_path, all_center_distances)
        print('done')

    center_distance_thresh = np.quantile(center_distances, 0.98)
    outlier_mask = (center_distances > center_distance_thresh)

    outlier_marked_images = images.copy()
    selection = get_centered_rectangle_slice(outlier_marked_images, outlier_mask.shape[:3])
    selection[np.where(outlier_mask)] = [255, 0, 0]

    rgb_colors = get_rgb_colors(n_clusters)
    color_overlays = rgb_colors[pixel_labels]

    label_overlay_images = apply_centered_overlay(images, color_overlays, 0.4)
    label_images = apply_centered_overlay(images, color_overlays, 1.)

    for label_image, label_overlay_image, outlier_image, rel_image_path in zip(
            label_images,
            label_overlay_images,
            outlier_marked_images,
            rel_image_paths,
    ):
        store_path_str = str(cluster_output / rel_image_path)
        Path(store_path_str).parent.mkdir(parents=True, exist_ok=True)
        boxes = select_bounding_boxes(rel_image_path, bounding_boxes)

        p = store_path_str.replace('.png', '_labels.png')
        draw_bounding_boxes(label_image, boxes, x_offset=-190)
        Image.fromarray(label_image).save(p, 'png')
        print(f'PNG created: {p}')

        p = store_path_str.replace('.png', '_labels_ov.png')
        draw_bounding_boxes(label_overlay_image, boxes, x_offset=-190)
        Image.fromarray(label_overlay_image).save(p, 'png')
        print(f'PNG created: {p}')

        p = store_path_str.replace('.png', '_outliers.png')
        draw_bounding_boxes(outlier_image, boxes, x_offset=-190)
        Image.fromarray(outlier_image).save(p, 'png')
        print(f'PNG created: {p}')


if __name__ == '__main__':
    main()
