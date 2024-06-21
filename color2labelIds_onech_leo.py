import os
import cv2
import numpy as np

dir_path = '/home/jackal-desktop/pict/input'
save_dir_path = '/home/jackal-desktop/pict/output'

def get_images_loader(dir_path, cv_bgr=True):
    images = []
    for filename in os.listdir(dir_path):
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            img_path = os.path.join(dir_path, filename)
            img = cv2.imread(img_path)
            if not cv_bgr:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    return images

imageset = get_images_loader(dir_path=dir_path, cv_bgr=True)

for i, img in enumerate(imageset):
    print(f'\r({i}/{len(imageset)})', end="")
    h, w, c = img.shape

    lower_bgr_tree = np.array([0, 0, 40])
    upper_bgr_tree = np.array([0, 0, 355])

    lower_bgr_road = np.array([140, 80, 130])
    upper_bgr_road = np.array([235, 150, 255])

    lower_bgr_sky = np.array([0, 0, 0])
    upper_bgr_sky = np.array([5, 5, 5])

    lower_bgr_terrain = np.array([0, 30, 30])
    upper_bgr_terrain = np.array([75, 170, 165])

    lower_bgr_building = np.array([240, 240, 240])
    upper_bgr_building = np.array([255, 255, 255])

    img_mask_building = cv2.inRange(img, lower_bgr_building, upper_bgr_building)
    img_mask_tree = cv2.inRange(img, lower_bgr_tree, upper_bgr_tree)
    img_mask_road = cv2.inRange(img, lower_bgr_road, upper_bgr_road)
    img_mask_sky = cv2.inRange(img, lower_bgr_sky, upper_bgr_sky)
    img_mask_terrain = cv2.inRange(img, lower_bgr_terrain, upper_bgr_terrain)


    result = np.zeros((h, w), np.uint8)
    result += 255

    tree_ids = np.zeros((h, w), np.uint8)
    tree_ids += 8

    road_ids = np.zeros((h, w), np.uint8)
    road_ids -= 255

    sky_ids = np.zeros((h, w), np.uint8)
    sky_ids += 11

    terrain_ids = np.zeros((h, w), np.uint8)
    terrain_ids += 10

    building_ids = np.zeros((h, w), np.uint8)
    building_ids += 1

    result = cv2.bitwise_not(result, result, mask=img_mask_tree)

    tmp_tree = cv2.bitwise_and(tree_ids, tree_ids, mask=img_mask_tree)
    tmp_road = cv2.bitwise_and(road_ids, road_ids, mask=img_mask_road)
    tmp_sky = cv2.bitwise_and(sky_ids, sky_ids, mask=img_mask_sky)
    tmp_terrain = cv2.bitwise_and(terrain_ids, terrain_ids, mask=img_mask_terrain)
    tmp_building = cv2.bitwise_and(building_ids, building_ids, mask=img_mask_building)


    result += tmp_tree
    result += tmp_road
    result += tmp_sky
    result += tmp_terrain
    result += tmp_building

    cv2.imwrite(os.path.join(save_dir_path, f'{i}_gtFine_labelTrainIds.png'), result)

print('>>> OK!!!')
