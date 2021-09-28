
import numpy as np 
import cv2
from keras.layers import Input
from keras import Model
from PIL import Image
import pickle
import os
import io
import tensorflow as tf
import keras

from utils.nms import apply_regr, non_max_suppression_fast, rpn_to_roi
from utils.image_processing import format_img, get_real_coordinates
from config import config

cfg = config.Config()

# def init_config(net, weights):
#     cfg_path = 'config/res_config.pickle' if net == 'res' else 'config/vgg_config.pickle'
#     with open(cfg_path, 'rb') as f_in:
#         cfg = pickle.load(f_in)
#     cfg.use_horizontal_flips = False
#     cfg.use_vertical_flips = False
#     cfg.rot_90 = False
#     cfg.model_path = weights

#     return cfg

def load_model():

    cfg = config.Config()
    cfg.model_path = 'weights/vgg_frcnn-best.hdf5'

    class_mapping = cfg.class_mapping
    class_mapping = {v: k for k, v in class_mapping.items()}
    class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}

    num_features = 512
    from layers.vgg16 import nn_base, rpn_layer, classifier_layer

    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, num_features)

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(cfg.num_rois, 4))
    feature_map_input = Input(shape=input_shape_features)

    # define the base network (VGG here, can be Resnet50, Inception, etc)
    shared_layers = nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = len(cfg.anchor_box_scales) * len(cfg.anchor_box_ratios)
    rpn_layers = rpn_layer(shared_layers, num_anchors)

    classifier = classifier_layer(feature_map_input, roi_input, cfg.num_rois, nb_classes=len(cfg.class_mapping))

    model_rpn = Model(img_input, rpn_layers)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    model_classifier = Model([feature_map_input, roi_input], classifier)

    print('Loading weights from {}'.format(cfg.model_path))
    # model_rpn.load_weights(cfg.model_path, by_name=True)
    # model_classifier.load_weights(cfg.model_path, by_name=True)

    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')

    return model_rpn, model_classifier, model_classifier_only, class_mapping, class_to_color

def predict_vgg(img_np, model_rpn, model_classifier, model_classifier_only, class_mapping, class_to_color):

    keras.backend.clear_session()

    img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    X, ratio = format_img(img, cfg)
    X = np.transpose(X, (0, 2, 3, 1))

    [Y1, Y2, F] = model_rpn.predict(X)
    R = rpn_to_roi(Y1, Y2, cfg, 'tensorflow', overlap_thresh=0.7)

    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    # apply the spatial pyramid pooling to the proposed regions
    bboxes = {}
    probs = {}

    for jk in range(R.shape[0] // cfg.num_rois + 1):
        ROIs = np.expand_dims(R[cfg.num_rois * jk:cfg.num_rois * (jk + 1), :], axis=0)
        if ROIs.shape[1] == 0:
            break

        if jk == R.shape[0] // cfg.num_rois:
            # pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0], cfg.num_rois, curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded

        [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

        # Calculate bboxes coordinates on resized image
        for ii in range(P_cls.shape[1]):
            # Ignore 'bg' class
            if np.max(P_cls[0, ii, :]) < 0.7 or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                continue

            cls_name = np.argmax(P_cls[0, ii, :])
            # class_mapping[]

            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []

            (x, y, w, h) = ROIs[0, ii, :]

            cls_num = np.argmax(P_cls[0, ii, :])
            try:
                (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                tx /= cfg.classifier_regr_std[0]
                ty /= cfg.classifier_regr_std[1]
                tw /= cfg.classifier_regr_std[2]
                th /= cfg.classifier_regr_std[3]
                x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
            except:
                pass
            bboxes[cls_name].append(
                [cfg.rpn_stride * x, cfg.rpn_stride * y, cfg.rpn_stride * (x + w), cfg.rpn_stride * (y + h)])
            probs[cls_name].append(np.max(P_cls[0, ii, :]))

    all_dets = []

    preds = []
    for key in bboxes:
        bbox = np.array(bboxes[key])

        new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.2)

        for jk in range(new_boxes.shape[0]):
            (x1, y1, x2, y2) = new_boxes[jk, :]
            (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
            coord =  {
                "x":real_x1,
                "y":real_y1,
                "x2":real_x2,
                "y2":real_y2,
                "score":round(new_probs[jk], 2),
                "label":key
            }

            preds.append(coord)
            # print(coord)
            # label = '{} {:.2f}'.format(key, new_probs[jk])

            # cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[class_mapping[key]][0]), int(class_to_color[class_mapping[key]][1]), int(class_to_color[class_mapping[key]][2])), 7)

            # textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
            all_dets.append((key,100*new_probs[jk]))

            #(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,3)
            #textOrg = (real_x1, real_y1-0)

            #cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), -1)
            #cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
            #cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 3)

    # img.show()
    # plt.imshow(img)
    # plt.show()
    # img = cv2.resize(img,(int(img.shape[0]/5), int(img.shape[0]/5)))

    return preds

model_rpn, model_classifier, model_classifier_only, class_mapping, class_to_color = load_model()

cv_img = cv2.imread("img.png")
bbxs = predict_vgg(cv_img, model_rpn, model_classifier, model_classifier_only, class_mapping, class_to_color)
