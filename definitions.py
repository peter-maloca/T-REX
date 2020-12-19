import os
from collections import namedtuple

PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_IN_DIR = os.path.join(PROJECT_ROOT_DIR, 'csv')
PLOTS_OUT_DIR = os.path.join(PROJECT_ROOT_DIR, 'out')

DL_PATCH_HEIGHT = 512
DL_PATCH_WIDTH = 512
DL_LABEL_NR = 4

OUT_IMAGE_FORMAT = 'png'
OUT_IMAGE_DPI = 300

ALL_COMPARTMENTS_TEST_SET_DETAILS_CSV = os.path.join(CSV_IN_DIR, 'overall_hamming_distances.csv')
VITREOUS_TEST_SET_DETAILS_CSV = os.path.join(CSV_IN_DIR, 'vitreous_hamming_distances.csv')
RETINA_TEST_SET_DETAILS_CSV = os.path.join(CSV_IN_DIR, 'retina_hamming_distances.csv')
CHOROID_TEST_SET_DETAILS_CSV = os.path.join(CSV_IN_DIR, 'choroid_hamming_distances.csv')
SCLERA_TEST_SET_DETAILS_CSV = os.path.join(CSV_IN_DIR, 'sclera_hamming_distances.csv')

ALL_COMPARTMENTS_TEST_SET_SUMMARY_CSV = os.path.join(CSV_IN_DIR, 'overall_average_hamming_distance.csv')
VITREOUS_TEST_SET_SUMMARY_CSV = os.path.join(CSV_IN_DIR, 'vitreous_average_hamming_distance.csv')
RETINA_TEST_SET_SUMMARY_CSV = os.path.join(CSV_IN_DIR, 'retina_average_hamming_distance.csv')
CHOROID_TEST_SET_SUMMARY_CSV = os.path.join(CSV_IN_DIR, 'choroid_average_hamming_distance.csv')
SCLERA_TEST_SET_SUMMARY_CSV = os.path.join(CSV_IN_DIR, 'sclera_average_hamming_distance.csv')

ColName = namedtuple('ColName', ["csv", "plot"])
COL_G1_G2 = ColName('hd_g1_g2', 'g1,g2')
COL_G1_G3 = ColName('hd_g1_g3', 'g1,g3')
COL_G2_G3 = ColName('hd_g2_g3', 'g2,g3')
COL_G1_CNN = ColName('hd_g1_cnn', 'g1,cnn')
COL_G2_CNN = ColName('hd_g2_cnn', 'g2,cnn')
COL_G3_CNN = ColName('hd_g3_cnn', 'g3,cnn')

SUBPLOT_NAME_ALL_COMPARTMENTS = '(a) all compartments'
SUBPLOT_NAME_VITREOUS = '(b) vitreous'
SUBPLOT_NAME_RETINA = '(c) retina'
SUBPLOT_NAME_CHOROID = '(d) choroid'
SUBPLOT_NAME_SCLERA = '(e) sclera'
