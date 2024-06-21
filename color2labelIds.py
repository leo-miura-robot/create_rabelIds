import cv2
import numpy as np

# file = 'images/color/Screenshot from 2022-12-14 10-50-56.png'
# file = 'images/color/Screenshot from 2022-12-14 12-03-00.png'
# file = '/home/envy/Desktop/unity_ano/0.png' 
file = '20240615191043022.jpeg' 
file = "test.png"
img = cv2.imread(file)

img = cv2.resize(img, (800, 500))
h, w, c = img.shape

print(f'h: {h}, w: {w}, c: {c}')

cv2.imshow('color image', img)
# cv2.waitKey(0)

lower_bgr_tree = np.array([0,0,40])
upper_bgr_tree = np.array([0,0,355])

lower_bgr_road = np.array([140,80,130])
upper_bgr_road = np.array([235,150,255])

lower_bgr_sky = np.array([0,0,0])
upper_bgr_sky = np.array([5,5,5])

lower_bgr_terrain = np.array([0,30,30])
upper_bgr_terrain = np.array([75,170,165])


img_mask_tree = cv2.inRange(img, lower_bgr_tree, upper_bgr_tree)
img_mask_road = cv2.inRange(img, lower_bgr_road, upper_bgr_road)
img_mask_sky = cv2.inRange(img, lower_bgr_sky, upper_bgr_sky)
img_mask_terrain = cv2.inRange(img, lower_bgr_terrain, upper_bgr_terrain)



result = np.zeros((h, w, c), np.uint8)
result += 255 # background is 255 in cityscapes.



# label TrainIdsの画像を作成
tree_ids = np.zeros((h, w, c), np.uint8)
#tree_ids += 21
tree_ids += 8

road_ids = np.zeros((h, w, c), np.uint8)
#road_ids += 7
road_ids -= 255 # TrainIds: 0

sky_ids = np.zeros((h, w, c), np.uint8)
#sky_ids += 23
sky_ids += 11 # TrainIds: 10

terrain_ids = np.zeros((h, w, c), np.uint8)
#terrain_ids += 22
terrain_ids += 10 # TrainIds: 9


result = cv2.bitwise_not(result, result, mask=img_mask_tree)


#result = cv2.bitwise_and(img, img, mask=img_mask_tree)
tmp_tree= cv2.bitwise_and(tree_ids, tree_ids, mask=img_mask_tree)
#tmp_road = cv2.bitwise_and(img, img, mask=img_mask_road)
tmp_road = cv2.bitwise_and(road_ids, road_ids, mask=img_mask_road)
#tmp_sky = cv2.bitwise_and(img, img, mask=img_mask_sky)
tmp_sky = cv2.bitwise_and(sky_ids, sky_ids, mask=img_mask_sky)
#tmp_terrain = cv2.bitwise_and(img, img, mask=img_mask_terrain)
tmp_terrain = cv2.bitwise_and(terrain_ids, terrain_ids, mask=img_mask_terrain)

result += tmp_tree
result += tmp_road
result += tmp_sky
result += tmp_terrain
#result[y_offset:y_offset + h, x_offset:x_offset + w] = tmp_tree

cv2.imshow('result', result)
#cv2.imshow('tree', result)
#cv2.imshow('tmp_road', tmp_road)
#cv2.imshow('tmp_sky', tmp_sky)
#cv2.imshow('tmp_terrain', tmp_terrain)
cv2.waitKey(0)

cv2.imwrite('./ano.png', result)