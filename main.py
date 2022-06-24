# import the packages
import numpy as np
import cv2
# from keras.applications import xception  # this is the model we'll be using for object detection
# from keras.applications.xception import preprocess_input  # for preprocessing the input
# from keras.applications import imagenet_utils
# from keras.preprocessing.image import img_to_array
# from imutils.object_detection import non_max_suppression

# read the input image
img = cv2.imread('Assets/img2.jpg')

# instanciate the selective search segmentation algorithm of opencv
search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
search.setBaseImage(img)  # set the base image as the input image

search.switchToSelectiveSearchFast()  # you can also use this for more accuracy -> search.switchToSelectiveSearchQuality()

rects = search.process()  # process the image

rois = []
boxes = []
(H, W) = img.shape[:2]
rois = []
boxes = []
(H, W) = img.shape[:2]
for (x, y, w, h) in rects:
    # check if the ROI has atleast
    # 20% the size of our image
    if w / float(W) < 0.2 or h / float(H) < 0.2:
        continue

    # Extract the Roi from image
    roi = img[y:y + h, x:x + w]
    # Convert it to RGB format
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    # Resize it to fit the input requirements of the model
    roi = cv2.resize(roi, (299, 299))

    # Further preprocessing
    roi = img_to_array(roi)
    roi = preprocess_input(roi)

    # Append it to our rois list
    rois.append(roi)

    # now let's store the box co-ordinates
    x1, y1, x2, y2 = x, y, x + w, y + h
    boxes.append((x1, y1, x2, y2))


# # convert the proposals list into NumPy array and show its dimensions
# proposals = np.array(proposals)
# print("[INFO] proposal shape: {}".format(proposals.shape))
# # classify each of the proposal ROIs using ResNet and then decode the
# # predictions
# print("[INFO] classifying proposals...")
# preds = model.predict(proposals)
# preds = imagenet_utils.decode_predictions(preds, top=1)
# # initialize a dictionary which maps class labels (keys) to any
# # bounding box associated with that label (values)
# labels = {}