from tqdm import tqdm
import numpy as np

def equalize_color_distribution(image_sets):

    print("preprocessing images started")

    for images, paths, boxes in tqdm(image_sets):
        # 1. select each image
        for _ in images:
            for image in _:
                # 2. find 25% and 75% percentile over this image on the three color channels
                p25 = np.percentile(image, 25, axis=(0,1))
                p75 = np.percentile(image, 75, axis=(0,1))
                # 3. change color distribution
                # 3.1 set 25% percentile to 0 and set 75% percentile to 1
                image[:,:,0] = (image[:,:,0] - p25[0]) / (p75[0]-p25[0])
                image[:,:,1] = (image[:,:,1] - p25[1]) / (p75[1]-p25[1])
                image[:,:,2] = (image[:,:,2] - p25[2]) / (p75[2]-p25[2])
                # 4. Check if it worked
                # print()
                # print(np.percentile(image, 25, axis=(0,1)))
                # print(np.percentile(image, 75, axis=(0,1)))

        pass

    print("preprocessing images finished")

    return images