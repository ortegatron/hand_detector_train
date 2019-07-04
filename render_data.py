HandKeypointsPairs = [[0,1],[1,2], [2,3], [3,4],[0,5],[5,6], [6,7],[7,8], [0,9],
    [9,10],[10,11], [11,12], [0,13],[13,14], [14,15], [15,16], [0,17], [17,18], [18,19], [19,20]]
HandKeypointsColors = [[100,  100,  100], \
[100,    0,    0], \
[150,    0,    0], \
[200,    0,    0], \
[255,    0,    0], \
[100,  100,    0], \
[150,  150,    0], \
[200,  200,    0], \
[255,  255,    0], \
[0,  100,   50], \
[0,  150,   75], \
[0,  200,  100], \
[0,  255,  125], \
[0,   50,  100], \
[0,   75,  150], \
[0,  100,  200], \
[0,  125,  255], \
[100,    0,  100], \
[150,    0,  150], \
[200,    0,  200], \
[255,    0,  255]]
HandKeypoints = 21

def draw_humans(img, hand_list):
    img_copied = np.copy(img)
    image_h, image_w = img_copied.shape[:2]
    centers = {}
    for hand in hand_list:
        part_idxs = hand.keys()

        # draw point
        for i in range(HandKeypoints):
            if i not in part_idxs:
                continue
            part_coord = hand[i][0:2]
            center = (int(part_coord[1] * image_w + 0.5), int(part_coord[0] * image_h + 0.5))
            centers[i] = center
            cv2.circle(img_copied, center, 3, HandKeypointsColors[i], thickness=3, lineType=8, shift=0)

        # draw line
        for pair_order, pair in enumerate(HandKeypointsPairs):
            if pair[0] not in part_idxs or pair[1] not in part_idxs:
                continue

            img_copied = cv2.line(img_copied, centers[pair[0]], centers[pair[1]], HandKeypointsColors[pair_order], 3)

    return img_copied

def render_image(path, hand_list):
    img = cv2.imread(path)
    image = draw_humans(image, hand_list)
    image = cv2.resize(image, (368, 368), interpolation=cv2.INTER_AREA)
    cv2.imshow('result', image)
    cv2.waitKey(0)
