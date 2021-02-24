import math

class Config:

    def __init__(self):
        # Print the process or not
        self.verbose = True

        # Name of base network
        self.network = 'vgg'

        # Setting for data augmentation
        self.use_horizontal_flips = True
        self.use_vertical_flips = True
        self.rot_90 = True

        # Anchor box scales
        # Note that if im_size is smaller, anchor_box_scales should be scaled
        # Original anchor_box_scales in the paper is [128, 256, 512]
        self.anchor_box_scales = [8, 16, 32, 64]

        # Anchor box ratios
        self.anchor_box_ratios = [[1, 1], [1. / math.sqrt(2), 2. / math.sqrt(2)],
                                  [2. / math.sqrt(2), 1. / math.sqrt(2)]]

        # Size to resize the smallest side of the image
        # Original setting in paper is 600. Set to 300 in here to save training time
        self.im_size = 300

        # image channel-wise mean to subtract
        self.img_channel_mean = [103.939, 116.779, 123.68]
        self.img_scaling_factor = 1.0

        # number of ROIs at once
        self.num_rois = 4

        # stride at the RPN (this depends on the network configuration)
        self.rpn_stride = 4

        # normally set to false but change for testing purpose
        # self.balanced_classes = False
        self.balanced_classes = True

        # scaling the stdev
        self.std_scaling = 4.0
        self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]

        # overlaps for RPN
        self.rpn_min_overlap = 0.3
        # self.rpn_max_overlap = 0.7
        self.rpn_max_overlap = 0.5

        # overlaps for classifier ROIs
        self.classifier_min_overlap = 0.1
        self.classifier_max_overlap = 0.5

        # TODO
        # should be replace with your own dataset's mapping
        # with bg as the number of classes
        # for example, if your dataset has three classes
        # your class_mapping should go like this {'class1': 0, 'class2': 1, 'class3': 2, 'bg':3}
        # normally should be this, but we change temporary for testing
        # need change id augmentation to do this properly

        self.class_mapping = {'Aortic enlargement': 0,
                              'Atelectasis': 1,
                              'Calcification': 2,
                              'Cardiomegaly': 3,
                              'Consolidation': 4,
                              'ILD': 5,
                              'Infiltration': 6,
                              'Lung Opacity' : 7,
                              'Nodule/Mass' : 8,
                              'Other lesion' : 9,
                              'Pleural effusion' : 10,
                              'Pleural thickening' : 11,
                              'Pneumothorax' : 12,
                              'Pulmonary fibrosis' : 13,
                              'bg' : 14,
                             }


        self.model_path = "../input/frcnn/vgg_frcnn-test-10.hdf5"
        self.training_annotation = "/content/drive/My Drive/Work/Manager_one/data/FRCNN/data-big/augmented-colab/annot/annot-augmented.csv'"
        self.cfg_save_path = None
        self.classes_count = None