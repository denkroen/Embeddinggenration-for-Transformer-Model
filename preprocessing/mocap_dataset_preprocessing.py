'''
Created on May 18, 2019

@author: fernando moya rueda
@email: fernando.moya@tu-dortmund.de
'''

import os
import sys
import numpy as np

import csv_reader
from sliding_window import sliding_window
import pickle


import matplotlib.pyplot as plt

from scipy.stats import norm, mode


#folder path of the dataset
FOLDER_PATH = "/vol/actrec/DFG_Project/2019/LARa_dataset/MoCap/recordings_2019/16_Annotated_Dataset/"

# Hardcoded number of sensor channels employed in the MoCap dataset
NB_SENSOR_CHANNELS = 134



NUM_CLASSES = 8
NUM_ATTRIBUTES = 19

NORM_MAX_THRESHOLDS = [3631.08295833,  4497.89608551,  3167.75032512,  7679.5730537,
                       7306.54182726,  5053.64124207,  3631.08295833,  4497.89608551,
                       3167.75032512,  7520.1195731,   5866.25362466,  4561.74563579,
                       8995.09142766, 10964.53262598,  9098.53329506,  5449.80983967,
                       5085.72851705,  3473.14411695, 21302.63337367, 18020.90101605,
                       16812.03779666, 15117.39792079, 18130.13729404, 20435.92835743,
                       11904.27678521, 10821.65838575, 12814.3751877,   6207.89358209,
                       6207.66099742,  5744.44264059, 18755.48551252, 13983.17089764,
                       15044.68926247, 16560.57636066, 16249.15177318, 50523.97223427,
                       13977.42954035, 11928.06755543, 14040.39506186, 12506.74970729,
                       11317.24558625, 18513.54602435, 17672.82964044, 12698.55673051,
                       10791.77877743, 21666.83178164, 24842.59483402, 17764.63662927,
                       18755.48551252, 13983.17089764, 15044.68926247, 31417.17595547,
                       25543.65023889, 41581.37446104, 17854.61535493, 17888.99265748,
                       18270.05893641, 21940.59554685, 28842.57380555, 26487.70921625,
                       17854.61535493, 17888.99265748, 18270.05893641, 26450.29857186,
                       29322.63199557, 33366.25283016, 14493.32082359,  7227.39740818,
                       13485.20175734,  5449.80983967,  5085.72851705,  3473.14411695,
                       18651.75669658, 23815.12584062, 21579.39921193, 32556.06051458,
                       19632.56510825, 38223.1285115,  23864.06687714,  8143.79426126,
                       10346.98996386, 10062.37072833, 14465.46088042,  6363.84691989,
                       15157.19066909, 14552.35778047, 17579.07446021, 28342.70805834,
                       30250.26765128, 38822.32871634, 11675.2974133,  21347.97047233,
                       9688.26485256, 15386.32967605,  6973.26742725,  8413.29172314,
                       17034.91758251, 13001.19959282, 11449.20721127, 28191.66779469,
                       23171.19896564, 18113.64230323, 15157.19066909, 14552.35778047,
                       17579.07446021, 33429.96810456, 24661.64883529, 42107.2934863,
                       17288.1377315,  27332.9219541,  18660.71577123, 38596.98925921,
                       34977.05772231, 60992.88899744, 17288.1377315, 27332.9219541,
                       18660.71577123, 48899.45655,    45458.05180034, 70658.90456363,
                       13302.12166712,  6774.86630402,  5359.33623984,  4589.29865259,
                       5994.63947534,  3327.83594399]

NORM_MIN_THRESHOLDS = [-2948.30406222,  -5104.92518724,  -3883.21490707,  -8443.7668244,
                       -8476.79530575,  -6094.99248245,  -2948.30406222,  -5104.92518724,
                       -3883.21490707,  -7638.48719528,  -7202.24599521,  -6357.05970106,
                       -7990.36654959, -12559.35302651, -11038.18468099,  -5803.12768035,
                       -6046.67001691,  -4188.84645697, -19537.6152259,  -19692.04664185,
                       -18548.37708291, -18465.90530892, -22119.96935856, -16818.45181433,
                       -10315.80425454, -11129.02825056, -10540.85493494,  -5590.78061688,
                       -6706.54838582,  -7255.845227,   -20614.03216916, -15124.01287142,
                       -14418.97715774, -15279.30518515, -15755.49700464, -52876.67430633,
                       -17969.48805632, -11548.41807713, -12319.34970371, -15331.29246293,
                       -13955.81324322, -15217.97507736, -17828.76429939, -12235.18670802,
                       -10508.19455787, -21039.94502811, -23382.9517919,  -16289.47810937,
                       -20614.03216916, -15124.01287142, -14418.97715774, -29307.81700117,
                       -27385.77923632, -39251.64863972, -18928.76957804, -20399.72095829,
                       -17417.32884474, -18517.86724449, -23566.88454454, -22782.77912723,
                       -18928.76957804, -20399.72095829, -17417.32884474, -23795.85006226,
                       -29914.3625062,  -27826.42086488, -11998.09109479,  -7978.46422461,
                       -12388.18397068,  -5803.12768035,  -6046.67001691,  -4188.84645697,
                       -20341.80941731, -23459.72733752, -20260.81953868, -33146.50450199,
                       -18298.11527347, -41007.64090081, -20861.12016565, -10084.98928355,
                       -12620.01970423,  -8183.86583411, -11868.40952478,  -6055.26285391,
                       -12839.53720997, -14943.34999686, -19473.17909211, -26832.66919396,
                       -28700.83598723, -31873.10404748, -14363.94407474, -19923.81826152,
                       -10022.64019372, -12509.07807217,  -8077.30383941,  -7964.13296659,
                       -14361.88766202, -13910.99623182, -13936.53426527, -34833.81498682,
                       -28282.58885647, -19432.17640984, -12839.53720997, -14943.34999686,
                       -19473.17909211, -30943.62377719, -26322.96645724, -44185.31075859,
                       -18305.69197916, -29297.10158134, -18929.46219474, -35741.42796924,
                       -37958.82196459, -65424.30589802, -18305.69197916, -29297.10158134,
                       -18929.46219474, -45307.70596895, -48035.24434893, -75795.15002644,
                       -11224.17786938,  -6928.28917534,  -4316.037138,    -4770.62854206,
                       -7629.61899295,  -4021.62984035]


headers = ["sample", "label", "head_RX", "head_RY", "head_RZ", "head_TX", "head_TY", "head_TZ", "head_end_RX",
           "head_end_RY", "head_end_RZ", "head_end_TX", "head_end_TY", "head_end_TZ", "L_collar_RX", "L_collar_RY",
           "L_collar_RZ", "L_collar_TX", "L_collar_TY", "L_collar_TZ", "L_elbow_RX", "L_elbow_RY", "L_elbow_RZ",
           "L_elbow_TX", "L_elbow_TY", "L_elbow_TZ", "L_femur_RX", "L_femur_RY", "L_femur_RZ", "L_femur_TX",
           "L_femur_TY", "L_femur_TZ", "L_foot_RX", "L_foot_RY", "L_foot_RZ", "L_foot_TX", "L_foot_TY", "L_foot_TZ",
           "L_humerus_RX", "L_humerus_RY", "L_humerus_RZ", "L_humerus_TX", "L_humerus_TY", "L_humerus_TZ", "L_tibia_RX",
           "L_tibia_RY", "L_tibia_RZ", "L_tibia_TX", "L_tibia_TY", "L_tibia_TZ", "L_toe_RX", "L_toe_RY", "L_toe_RZ",
           "L_toe_TX", "L_toe_TY", "L_toe_TZ", "L_wrist_RX", "L_wrist_RY", "L_wrist_RZ", "L_wrist_TX", "L_wrist_TY",
           "L_wrist_TZ", "L_wrist_end_RX", "L_wrist_end_RY", "L_wrist_end_RZ", "L_wrist_end_TX", "L_wrist_end_TY",
           "L_wrist_end_TZ", "R_collar_RX", "R_collar_RY", "R_collar_RZ", "R_collar_TX", "R_collar_TY", "R_collar_TZ",
           "R_elbow_RX", "R_elbow_RY", "R_elbow_RZ", "R_elbow_TX", "R_elbow_TY", "R_elbow_TZ", "R_femur_RX",
           "R_femur_RY", "R_femur_RZ", "R_femur_TX", "R_femur_TY", "R_femur_TZ", "R_foot_RX", "R_foot_RY", "R_foot_RZ",
           "R_foot_TX", "R_foot_TY", "R_foot_TZ", "R_humerus_RX", "R_humerus_RY", "R_humerus_RZ", "R_humerus_TX",
           "R_humerus_TY", "R_humerus_TZ", "R_tibia_RX", "R_tibia_RY", "R_tibia_RZ", "R_tibia_TX", "R_tibia_TY",
           "R_tibia_TZ", "R_toe_RX", "R_toe_RY", "R_toe_RZ", "R_toe_TX", "R_toe_TY", "R_toe_TZ", "R_wrist_RX",
           "R_wrist_RY", "R_wrist_RZ", "R_wrist_TX", "R_wrist_TY", "R_wrist_TZ", "R_wrist_end_RX", "R_wrist_end_RY",
           "R_wrist_end_RZ", "R_wrist_end_TX", "R_wrist_end_TY", "R_wrist_end_TZ", "root_RX", "root_RY", "root_RZ",
           "root_TX", "root_TY", "root_TZ"]

annotator = {"S01": "A17", "S02": "A03", "S03": "A08", "S04": "A06", "S05": "A12", "S06": "A13",
             "S07": "A05", "S08": "A17", "S09": "A03", "S10": "A18", "S11": "A08", "S12": "A11",
             "S13": "A08", "S14": "A06", "S15": "A05", "S16": "A05"}

SCENARIO = {'R01': 'L01', 'R02': 'L01', 'R03': 'L02', 'R04': 'L02', 'R05': 'L02', 'R06': 'L02', 'R07': 'L02',
            'R08': 'L02', 'R09': 'L02', 'R10': 'L02', 'R11': 'L02', 'R12': 'L02', 'R13': 'L02', 'R14': 'L02',
            'R15': 'L02', 'R16': 'L02', 'R17': 'L03', 'R18': 'L03', 'R19': 'L03', 'R20': 'L03', 'R21': 'L03',
            'R22': 'L03', 'R23': 'L03', 'R24': 'L03', 'R25': 'L03', 'R26': 'L03', 'R27': 'L03', 'R28': 'L03',
            'R29': 'L03', 'R30': 'L03'}

#scenario = ['S01']
persons = ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09",
           "S10", "S11", "S12", "S13", "S14", "S15", "S16"]
repetition = ["N01", "N02"]
annotator_S01 = ["A17", "A12"]

labels_persons = {"S01": 0, "S02": 1, "S03": 2, "S04": 3, "S05": 4, "S06": 5, "S07": 6, "S08": 7, "S09": 8,
                  "S10": 9, "S11": 10, "S12": 11, "S13": 12, "S14": 13, "S15": 14, "S16": 15}



def norm_mean_std(data):
    """
    Zero Mean and Unit variance Normalization of all sensor channels

    @param data: numpy integer matrix
        Sensor data
    @return:
        Normalized sensor data
    """
    '''
    mean_values = np.array([1.7000e-03,    3.0000e-03,    2.0000e-04,    7.9530e-01,    -1.6210e-01,    -7.6498e+00,
                            1.7000e-03,    3.0000e-03,    2.0000e-04,    7.9680e-01,    -1.5780e-01,    -7.6666e+00,
                            7.0000e-03,    -2.7000e-03,    -2.9000e-03,    3.2810e-01,    9.0850e-01,    -1.3978e+00,
                            1.0100e-02,    -5.5000e-03,    2.2000e-03,    -3.1980e-01,    6.1110e-01,    6.5190e-01,
                            -3.8000e-03,    -4.5000e-03,    6.0000e-03,    8.2730e-01,    5.9650e-01,    -5.8962e+00,
                            1.6000e-03,    -4.0000e-04,    -3.1000e-03,    -4.0300e-01,    6.8460e-01,    -3.0375e+00,
                            -1.3000e-03,    2.2000e-03,    -1.0000e-03,    -3.2370e-01,    1.5106e+00,    1.2480e+00,
                            -3.7000e-03,    -8.0000e-04,    7.2000e-03,    9.9260e-01,    -1.3420e-01,    -5.8047e+00,
                            1.6000e-03,    -4.0000e-04,    -3.1000e-03,    -3.8690e-01,    7.0290e-01,    -3.0385e+00,
                            8.6000e-03,    -8.5000e-03,    -3.9000e-03,    -8.0130e-01,    5.2700e-02,    1.6870e-01,
                            8.6000e-03,    -8.5000e-03,    -3.9000e-03,    -8.3240e-01,    1.3810e-01,    1.8800e-02,
                            6.7000e-03,    4.0000e-03,    2.9000e-03,    -3.6760e-01,    7.6400e-02,    -7.0350e-01,
                            5.7000e-03,    -9.0000e-04,    5.1000e-03,    3.6550e-01,    6.0460e-01,    1.1312e+00,
                            -5.8000e-03,    -1.0000e-04,    -8.3000e-03,    -1.7512e+00,    3.5390e-01,    -5.3584e+00,
                            0.0000e+00,    1.7000e-03,    1.3000e-03,    4.0760e-01,    -4.0570e-01,    -4.2498e+00,
                            3.7000e-03,    6.4000e-03,    2.7000e-03,    -6.8890e-01,    -1.4930e+00,    5.8060e-01,
                            -6.1000e-03,    -4.3000e-03,    -9.8000e-03,    -1.7537e+00,    -8.7710e-01,    -5.3975e+00,
                            0.0000e+00,    1.7000e-03,    1.3000e-03,    3.7250e-01,    -4.1670e-01,    -4.2086e+00,
                            7.4000e-03,    1.6000e-03,    2.8000e-03,    7.5380e-01,    1.3619e+00,    4.2910e-01,
                            7.4000e-03,    1.6000e-03,    2.8000e-03,    7.4610e-01,    1.4089e+00,    3.6480e-01,
                            1.0000e-03,    -5.0000e-04,    1.0000e-04,    1.2100e-02,    -1.5800e-01,    -9.1455e+00]
                            )'''
                            
    mean_values = np.array([-1.6000e-03, 1.4000e-03, -7.0000e-04, 3.4800e-01, -6.3280e-01, -7.5923e+00,
                            -1.6000e-03, 1.4000e-03, -7.0000e-04, 3.4830e-01, -6.3190e-01, -7.6037e+00,
                            2.6000e-03, -1.3000e-03, 3.0000e-04, -2.1038e+00, 2.3570e-01, -1.5009e+00,
                            -1.0000e-04, -3.5000e-03, -2.1000e-03, 1.9961e+00, -2.2746e+00, 7.7250e-01,
                            -7.1000e-03, 6.4000e-03, 8.2000e-03, 1.8307e+00, -6.3500e-02, -6.3027e+00,
                            2.8000e-03, -2.9000e-03, 5.0000e-04, -9.7480e-01, 1.3448e+00, -3.5118e+00,
                            2.0000e-03, 2.2000e-03, 4.1000e-03, -1.9786e+00, -2.5674e+00, 1.1473e+00,
                            -6.8000e-03, 4.2000e-03, 5.9000e-03, 9.5750e-01, -7.8590e-01, -6.2745e+00,
                            2.8000e-03, -2.9000e-03, 5.0000e-04, -9.6320e-01, 1.3374e+00, -3.4579e+00,
                            -7.0000e-04, -5.0000e-04, -5.0000e-04, 2.1670e-01, -1.8719e+00, 3.5690e-01,
                            -7.0000e-04, -5.0000e-04, -5.0000e-04, 1.8780e-01, -1.8345e+00, 3.3630e-01,
                            3.4000e-03, 2.8000e-03, -2.7000e-03, 8.7780e-01, -1.8100e-02, -7.5300e-01,
                            3.0000e-04, -2.0000e-03, -3.6000e-03, -8.8300e-01, -2.1414e+00, 9.1320e-01,
                            -9.3000e-03, -1.1400e-02, -1.0000e-02, -2.0146e+00, -3.5800e-01, -5.5261e+00,
                            -8.0000e-04, 0.0000e+00, -1.8000e-03, 5.4150e-01, 1.1460e-01, -4.6931e+00,
                            2.6000e-03, 1.9000e-03, 1.5000e-03, -1.0074e+00, -1.6004e+00, 6.0490e-01,
                            -1.0300e-02, -6.9000e-03, -6.4000e-03, -6.8860e-01, -1.5154e+00, -5.6251e+00,
                            -8.0000e-04, 0.0000e+00, -1.8000e-03, 5.1090e-01, 1.0670e-01, -4.6293e+00,
                            3.5000e-03, 1.8000e-03, 1.9000e-03, 7.6370e-01, -1.4697e+00, 3.9970e-01,
                            3.5000e-03, 1.8000e-03, 1.9000e-03, 7.4640e-01, -1.4457e+00, 3.8630e-01,
                            8.0000e-04, -3.0000e-04, -1.0000e-04, 7.9100e-02, -1.9900e-02, -9.1681e+00]
                           )
                            



    mean_values = np.reshape(mean_values, [1, 126])

    '''
    std_values = np.array([0.52295,    0.42407,    0.39099,    4.89407,    4.51983,    4.23851,    0.52295,
                           0.42407,    0.39099,    5.18099,    5.16119,    4.36874,    1.69787,    1.1684,
                           1.23195,    25.02545,    20.59924,    18.43413,    4.29162,    2.78653,    2.9752,
                           94.96388,    79.652,    87.81026,    1.30248,    1.07576,    1.01212,    8.73997,
                           8.36857,    7.95617,    1.6288,    1.34426,    1.21621,    56.79301,    50.6708,
                           47.20261,    1.65167,    1.11243,    1.38116,    29.45059,    26.29006,    27.47176,
                           1.35961,    1.07605,    0.99073,    25.44898,    23.19467,    20.93677,    1.6288,
                           1.34426,    1.21621,    57.14826,    49.24714,    49.09993,    3.87289,    3.2798,
                           4.10213,    104.81611,    158.07939,    108.78809,    3.87289,    3.2798,    4.10213,
                           103.98942,    148.04898,    103.32891,    2.323,    1.37348,    1.31813,    31.68585,
                           26.88045,    23.31267,    2.53725,    2.56811,    2.50781,    81.22631,    68.9508,
                           74.13007,    1.30681,    0.83385,    0.87436,    10.32767,    13.26679,    8.60524,
                           2.20039,    1.44667,    1.48765,    95.8076,    67.85622,    63.03837,    2.37677,
                           1.34351,    1.90272,    49.41886,    52.88895,    47.15615,    1.32301,    0.88566,
                           0.87466,    37.77399,    40.47674,    23.47575,    2.20039,    1.44667,    1.48765,
                           96.17357,    68.18398,    64.01532,    2.71997,    2.4048,    2.62287,    101.28354,
                           119.18785,    120.47364,    2.71997,    2.4048,    2.62287,    105.59451,    123.5979,
                           122.01776,    0.72374,    0.51977,    0.50875,    4.26357,    3.90542,    3.8055])'''

    std_values = np.array([0.42428, 0.40512, 0.50368, 6.47701, 6.51766, 4.05592, 0.42428,
                           0.40512, 0.50368, 6.93332, 7.88211, 4.10296, 1.19096, 1.21057,
                           1.91201, 16.29901, 17.06449, 14.94955, 3.36838, 3.06988, 4.19589,
                           116.49753, 81.00731, 92.27294, 1.22327, 0.98928, 1.39384, 8.01251,
                           8.17852, 8.51991, 1.14389, 1.31315, 1.91251, 56.22116, 45.44321,
                           46.21487, 1.13487, 1.33469, 1.75481, 30.6831, 39.51951, 34.90322,
                           1.1385, 1.01123, 1.39284, 23.28236, 21.60107, 23.97951, 1.14389,
                           1.31315, 1.91251, 55.18452, 44.29709, 48.16269, 3.66534, 4.02052,
                           3.52749, 127.12043, 128.72466, 120.11466, 3.66534, 4.02052, 3.52749,
                           126.34024, 120.61691, 116.67072, 1.31386, 1.37969, 2.13014, 19.26322,
                           29.27579, 26.64587, 2.22397, 2.40963, 3.39725, 91.55696, 90.21959,
                           89.21447, 1.04201, 0.91833, 1.22407, 8.77702, 10.55981, 8.04777,
                           1.4737, 1.58069, 1.84547, 79.52957, 60.98258, 62.17366, 1.64698,
                           1.33645, 2.2807, 36.24398, 43.82675, 41.93683, 1.11872, 0.87364,
                           1.23656, 23.6577, 32.01995, 20.51407, 1.4737, 1.58069, 1.84547,
                           68.82119, 62.17224, 60.72623, 2.59212, 2.44541, 2.84046, 104.00399,
                           123.48703, 112.32595, 2.59212, 2.44541, 2.84046, 106.06369, 124.3566,
                           117.30626, 0.67726, 0.55095, 0.62137, 3.86223, 3.9958, 4.44722]
                          )

    std_values = np.reshape(std_values, [1, 126])

    mean_array = np.repeat(mean_values, data.shape[0], axis=0)
    std_array = np.repeat(std_values, data.shape[0], axis=0)

    max_values = mean_array + 2 * std_array
    min_values = mean_array - 2 * std_array

    data_norm = (data - min_values) / (max_values - min_values)

    data_norm[data_norm>1] = 1
    data_norm[data_norm < 0] = 0

    #data_norm = (data - mean_array) / std_array

    return data_norm

def select_columns_opp(data):
    """
    Selection of the columns employed in the MoCAP
    excluding the measurements from lower back,
    as this became the center of the human body,
    and the rest of joints are normalized
    with respect to this one

    :param data: numpy integer matrix
        Sensor data (all features)
    :return: numpy integer matrix
        Selection of features
    """

    #included-excluded
    features_delete = np.arange(68, 74)
    
    return np.delete(data, features_delete, 1)

def divide_x_y(data):
    """Segments each sample into features and label

    :param data: numpy integer matrix
        Sensor data
    :return: numpy integer matrix, numpy integer array
        Features encapsulated into a matrix and labels as an array
    """
    data_t = data[:, 0]
    data_y = data[:, 1]
    data_x = data[:, 2:]
    

    return data_t, data_x, data_y

def normalize(data):
    """
    Normalizes all sensor channels

    :param data: numpy integer matrix
        Sensor data
    :return:
        Normalized sensor data
    """
    try:
        max_list, min_list = np.array(NORM_MAX_THRESHOLDS), np.array(NORM_MIN_THRESHOLDS)
        diffs = max_list - min_list
        for i in np.arange(data.shape[1]):
            data[:, i] = (data[:, i]-min_list[i])/diffs[i]
        #     Checking the boundaries
        data[data > 1] = 0.99
        data[data < 0] = 0.00
    except:
        raise("Error in normalization")
        
    return data


def opp_sliding_window(data_x, data_y, ws, ss, label_pos_end=True):
    '''
    Performs the sliding window approach on the data and the labels

    return three arrays.
    - data, an array where first dim is the windows
    - labels per window according to end, middle or mode
    - all labels per window

    @param data_x: ids for train
    @param data_y: ids for train
    @param ws: ids for train
    @param ss: ids for train
    @param label_pos_end: ids for train
    @return data_x: Sequence train inputs [Batch,1, C, T]
    @return data_y_labels: Activity classes [B, 1]
    @return data_y_all: Activity classes for samples [Batch,1,T]
    '''

    print("Sliding window: Creating windows {} with step {}".format(ws, ss))

    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))

    # Label from the end
    if label_pos_end:
        data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, (ws, data_y.shape[1]), (ss, 1))])
    else:
        if False:
            # Label from the middle
            # not used in experiments
            data_y_labels = np.asarray(
                [[i[i.shape[0] // 2]] for i in sliding_window(data_y, (ws, data_y.shape[1]), (ss, 1))])
        else:
            # Label according to mode
            try:
                data_y_labels = []
                for sw in sliding_window(data_y, (ws, data_y.shape[1]), (ss, 1)):
                    labels = np.zeros((20)).astype(int)
                    count_l = np.bincount(sw[:, 0], minlength=NUM_CLASSES)
                    idy = np.argmax(count_l)
                    attrs = np.sum(sw[:, 1:], axis=0)
                    attrs[attrs > 0] = 1
                    labels[0] = idy
                    labels[1:] = attrs
                    data_y_labels.append(labels)
                data_y_labels = np.asarray(data_y_labels)


            except:
                print("Sliding window: error with the counting {}".format(count_l))
                print("Sliding window: error with the counting {}".format(idy))
                return np.Inf

            # All labels per window
            data_y_all = np.asarray([i[:] for i in sliding_window(data_y, (ws, data_y.shape[1]), (ss, 1))])

    return data_x.astype(np.float32), data_y_labels.astype(np.uint8), data_y_all.astype(np.uint8)



def compute_max_min(ids):
    '''
    Compute the max and min values for normalizing the data.
    
    
    print max and min.
    These values will be computed only once and the max min values
    will be place as constants
    
    @param ids: ids for train
    '''

    recordings = ['R{:02d}'.format(r) for r in range(1, 31)]
    
    max_values_total = np.zeros((132))
    min_values_total = np.ones((132)) * 1000000
    for P in persons:
        if P in ids:
            for r, R in enumerate(recordings):
                # All of these if-cases are coming due to the naming of the recordings in the data.
                # Not all the subjects have the same
                # annotated recordings, nor annotators, nor annotations runs, nor scenarios.
                # these will include all of the recordings for the subjects
                if P in ["S01", "S02", "S03", "S04", "S05", "S06"]:
                    S = "L01"
                else:
                    S = SCENARIO[r]
                for N in repetition:
                    annotator_file = annotator[P]
                    if P == 'S07' and SCENARIO[r] == 'S01':
                        annotator_file = "A03"
                    if P == 'S14' and SCENARIO[r] == 'S03':
                        annotator_file = "A19"
                    if P == 'S11' and SCENARIO[r] == 'S01':
                        annotator_file = "A03"
                    if P == 'S11' and R in ['R04', 'R08', 'R09', 'R10', 'R11', 'R12', 'R13', 'R15']:
                        annotator_file = "A02"
                    if P == 'S13' and R in ['R28']:
                        annotator_file = "A01"
                    if P == 'S13' and R in ['R29', 'R30']:
                        annotator_file = "A11"
                    if P == 'S09' and R in ['R28', 'R29']:
                        annotator_file = "A01"
                    if P == 'S09' and R in ['R21', 'R22', 'R23', 'R24', 'R25']:
                        annotator_file = "A11"

                    file_name_norm = "{}/{}_{}_{}_{}_{}_norm_data.csv".format(P, S, P, R, annotator_file, N)

                    try:
                        print(FOLDER_PATH + file_name_norm)
                        data = csv_reader.reader_data(FOLDER_PATH + file_name_norm)
                        print("Files loaded")

                        data_t, data_x, data_y = divide_x_y(data)
                        del data_t
                        del data_y

                        max_values = np.max(data_x, axis = 0)
                        min_values = np.min(data_x, axis = 0)

                        max_values_total = np.max((max_values, max_values_total), axis = 0)
                        min_values_total = np.min((min_values, min_values_total), axis = 0)

                    except:
                        print("No file {}".format(FOLDER_PATH + file_name_norm))
    
    print("Max values \n{}".format(max_values_total))
    print("Min values \n{}".format(min_values_total))
    
    return




def compute_min_num_samples(ids, boolean_classes=True, attr=0):
    '''
    Compute the minimum duration of a sequences with the same classes or attribute
    
    This value will help selecting the best sliding window size
    
    @param ids: ids for train
    @param boolean_classes: selecting between classes or attributes
    @param attr: ids for attribute
    '''

    recordings = ['R{:02d}'.format(r) for r in range(1, 31)]

    if boolean_classes:
        NUM_CLASSES = 8
    else:
        NUM_CLASSES = 2

    #min_durations = np.ones((NUM_CLASSES)) * 10000000
    min_durations = np.empty((0,NUM_CLASSES))
    hist_classes_all = np.zeros((NUM_CLASSES))
    for P in persons:
        if P in ids:
            for r, R in enumerate(recordings):
                # All of these if-cases are coming due to the naming of the recordings in the data.
                # Not all the subjects have the same
                # annotated recordings, nor annotators, nor annotations runs, nor scenarios.
                # these will include all of the recordings for the subjects
                if P in ["S01", "S02", "S03", "S04", "S05", "S06"]:
                    S = "L01"
                else:
                    S = SCENARIO[r]
                for N in repetition:
                    annotator_file = annotator[P]
                    if P == 'S07' and SCENARIO[r] == 'L01':
                        annotator_file = "A03"
                    if P == 'S14' and SCENARIO[r] == 'L03':
                        annotator_file = "A19"
                    if P == 'S11' and SCENARIO[r] == 'L01':
                        annotator_file = "A03"
                    if P == 'S11' and R in ['R04', 'R08', 'R09', 'R10', 'R11', 'R12', 'R13', 'R15']:
                        annotator_file = "A02"
                    if P == 'S13' and R in ['R28']:
                        annotator_file = "A01"
                    if P == 'S13' and R in ['R29', 'R30']:
                        annotator_file = "A11"
                    if P == 'S09' and R in ['R28', 'R29']:
                        annotator_file = "A01"
                    if P == 'S09' and R in ['R21', 'R22', 'R23', 'R24', 'R25']:
                        annotator_file = "A11"
                    file_name_label = "{}/{}_{}_{}_{}_{}_labels.csv".format(P, S, P, R, annotator_file,N)

                    try:
                        data = csv_reader.reader_labels(FOLDER_PATH + file_name_label)
                        labels = data[:,attr]
                        print("Files loaded")

                        min_duration = np.zeros((1,NUM_CLASSES))
                        for c in range(NUM_CLASSES):

                            #indexes per class
                            idxs = np.where(labels == c)[0]
                            counter = 0
                            min_counter = np.Inf
                            #counting if continuity in labels
                            for idx in range(idxs.shape[0] - 1):
                                if idxs[idx + 1] - idxs[idx] == 1:
                                    counter += 1
                                else:
                                    if counter < min_counter:
                                        min_counter = counter
                                        counter = 0
                            if counter < min_counter:
                                min_counter = counter
                                counter = 0
                            min_duration[0,c] = min_counter

                            print("class  {} counter size {}".format(c, min_counter))

                        min_durations = np.append(min_durations, min_duration, axis = 0)
                        #Statistics

                        hist_classes = np.bincount(labels.astype(int), minlength = NUM_CLASSES)
                        hist_classes_all += hist_classes

                    except:
                        print("No file {}".format(FOLDER_PATH + file_name_label))
    
    min_durations[min_durations == 0] = np.Inf
    print("Minimal duration per class \n{}".format(min_durations))
    
    print("Number of samples per class {}".format(hist_classes_all))
    print("Number of samples per class {}".format(hist_classes_all / np.float(np.sum(hist_classes_all)) * 100))
    
    return np.min(min_durations, axis = 0)



def compute_statistics_samples(ids, boolean_classes=True, attr=0):
    '''
    Compute some statistics of the duration of the sequences data:

    print:
    Max and Min durations per class or attr
    Mean and Std durations per class or attr
    Lower whiskers durations per class or attr
    1st quartile of durations per class or attr
    Histogram of proportion per class or attr
    
    @param ids: ids for train
    @param boolean_classes: selecting between classes or attributes
    @param attr: ids for attribute
    '''

    recordings = ['R{:02d}'.format(r) for r in range(1, 31)]

    counter_list_class = {}

    if boolean_classes:
        NUM_CLASSES = 8
    else:
        NUM_CLASSES = 2
    
    for cl in range(NUM_CLASSES):
        counter_list_class[cl] = []
    
    hist_classes_all = np.zeros((NUM_CLASSES))
    for P in persons:
        if P in ids:
            for r, R in enumerate(recordings):
                # All of these if-cases are coming due to the naming of the recordings in the data.
                # Not all the subjects have the same
                # annotated recordings, nor annotators, nor annotations runs, nor scenarios.
                # these will include all of the recordings for the subjects
                if P in ["S01", "S02", "S03", "S04", "S05", "S06"]:
                    S = "L01"
                else:
                    S = SCENARIO[r]
                for N in repetition:
                    annotator_file = annotator[P]
                    if P == 'S07' and SCENARIO[r] == 'L01':
                        annotator_file = "A03"
                    if P == 'S14' and SCENARIO[r] == 'L03':
                        annotator_file = "A19"
                    if P == 'S11' and SCENARIO[r] == 'L01':
                        annotator_file = "A03"
                    if P == 'S11' and R in ['R04', 'R08', 'R09', 'R10', 'R11', 'R12', 'R13', 'R15']:
                        annotator_file = "A02"
                    if P == 'S13' and R in ['R28']:
                        annotator_file = "A01"
                    if P == 'S13' and R in ['R29', 'R30']:
                        annotator_file = "A11"
                    if P == 'S09' and R in ['R28', 'R29']:
                        annotator_file = "A01"
                    if P == 'S09' and R in ['R21', 'R22', 'R23', 'R24', 'R25']:
                        annotator_file = "A11"
                    file_name_label = "{}/{}_{}_{}_{}_{}_labels.csv".format(P, S, P, R, annotator_file, N)

                    try:
                        data = csv_reader.reader_labels(FOLDER_PATH + file_name_label)
                        labels = data[:,attr]
                        print("Files loaded")

                        for c in range(NUM_CLASSES):

                            #indexes per class
                            idxs = np.where(labels == c)[0]
                            counter = 0

                            #counting if continuity in labels
                            for idx in range(idxs.shape[0] - 1):
                                if idxs[idx + 1] - idxs[idx] == 1:
                                    counter += 1
                                else:
                                    counter_list_class[c].append(counter)
                                    counter = 0

                                if (idx+1) == (idxs.shape[0] - 1):
                                    counter_list_class[c].append(counter)
                                    counter = 0
                        #Statistics

                        hist_classes = np.bincount(labels.astype(int), minlength = NUM_CLASSES)
                        hist_classes_all += hist_classes
                    except:
                        print("No file {}".format(FOLDER_PATH + file_name_label))

    fig = plt.figure()
    axis_list = []
    axis_list.append(fig.add_subplot(421))
    axis_list.append(fig.add_subplot(422))
    axis_list.append(fig.add_subplot(423))
    axis_list.append(fig.add_subplot(424))
    axis_list.append(fig.add_subplot(425))
    axis_list.append(fig.add_subplot(426))
    axis_list.append(fig.add_subplot(427))
    axis_list.append(fig.add_subplot(428))
    
    fig2 = plt.figure()
    axis_list_2 = []
    axis_list_2.append(fig2.add_subplot(111))

    fig3 = plt.figure()
    axis_list_3 = []
    axis_list_3.append(fig3.add_subplot(421))
    axis_list_3.append(fig3.add_subplot(422))
    axis_list_3.append(fig3.add_subplot(423))
    axis_list_3.append(fig3.add_subplot(424))
    axis_list_3.append(fig3.add_subplot(425))
    axis_list_3.append(fig3.add_subplot(426))
    axis_list_3.append(fig3.add_subplot(427))
    axis_list_3.append(fig3.add_subplot(428))  

    colours = {0 : 'b', 1 : 'g', 2 : 'r', 3 : 'c', 4 : 'm', 5 : 'y', 6 : 'k', 7 : 'greenyellow'}
    
    mins = []
    mus = []
    sigmas = []
    min_1_data = []
    min_2_data = []
    min_3_data = []
    medians = []
    lower_whiskers = []
    Q1s = []
    for cl in range(NUM_CLASSES):
        mu = np.mean(np.array(counter_list_class[cl]))
        sigma = np.std(np.array(counter_list_class[cl]))
        
        mus.append(mu)
        sigmas.append(sigma)
        min_1_data.append(- 1 * sigma + mu)
        min_2_data.append(- 2 * sigma + mu)
        min_3_data.append(- 3 * sigma + mu)
        mins.append(np.min(np.array(counter_list_class[cl])))
        medians.append(np.median(np.array(counter_list_class[cl])))
        
        x = np.linspace(-3 * sigma + mu, 3 * sigma + mu, 100)
        
        axis_list[cl].plot(x, norm.pdf(x,mu,sigma) / np.float(np.max(norm.pdf(x,mu,sigma))),
                           '-b', label='mean:{}_std:{}'.format(mu, sigma))
        axis_list[cl].plot(counter_list_class[cl], np.ones(len(counter_list_class[cl])) , 'ro')
        result_box = axis_list[cl].boxplot(counter_list_class[cl], vert=False)
        lower_whiskers.append(result_box['whiskers'][0].get_data()[0][0])
        Q1s.append(result_box['whiskers'][0].get_data()[0][1])
        
        axis_list_2[0].plot(x, norm.pdf(x,mu,sigma) /  np.float(np.max(norm.pdf(x,mu,sigma))),
                            '-b', label='mean:{}_std:{}'.format(mu, sigma), color = colours[cl])
        axis_list_2[0].plot(counter_list_class[cl], np.ones(len(counter_list_class[cl])) , 'ro')
                            #color = colours[cl], marker='o')
                            
                            
        axis_list_3[cl].boxplot(counter_list_class[cl])

        axis_list_2[0].relim()
        axis_list_2[0].autoscale_view()
        axis_list_2[0].legend(loc='best')

        fig.canvas.draw()
        fig2.canvas.draw()
        plt.pause(2.0)
    
    print("Mins {} Min {} Argmin {}".format(mins, np.min(mins), np.argmin(mins)))
    print("Means {} Min {} Argmin {}".format(mus, np.min(mus), np.argmin(mus)))
    print("Stds {} Min {}".format(sigmas, sigmas[np.argmin(mus)]))
    print("Medians {} Min {} Argmin {}".format(medians, np.min(medians), np.argmin(medians)))
    print("Lower Whiskers {} Min {} Argmin {}".format(lower_whiskers, np.min(lower_whiskers), np.argmin(lower_whiskers)))
    print("Q1s {} Min {} Argmin {}".format(Q1s, np.min(Q1s), np.argmin(Q1s)))
    
    
    print("1sigma from mu {}".format(min_1_data))
    print("2sigma from mu {}".format(min_2_data))
    print("3sigma from mu {}".format(min_3_data))
    
    print("Min 1sigma from mu {}".format(np.min(min_1_data)))
    print("Min 2sigma from mu {}".format(np.min(min_2_data)))
    print("Min 3sigma from mu {}".format(np.min(min_3_data)))
    
    print("Number of samples per class {}".format(hist_classes_all))
    print("Number of samples per class {}".format(hist_classes_all / np.float(np.sum(hist_classes_all)) * 100))
    
    return


################
# Generate data
#################
def generate_data(ids, sliding_window_length, sliding_window_step, data_dir=None, half=False,
                  identity_bool=False, usage_modus='train'):
    '''
    creates files for each of the sequences, which are extracted from a file
    following a sliding window approach
    
    returns
    Sequences are stored in given path
    
    @param ids: ids for train, val or test
    @param sliding_window_length: length of window for segmentation
    @param sliding_window_step: step between windows for segmentation
    @param data_dir: path to dir where files will be stored
    @param half: using the half of the recording frequency
    @param identity_bool: selecting for identity experiment
    @param usage_modus: selecting Train, Val or testing
    '''

    if identity_bool:
        if usage_modus == 'train':
            recordings = ['R{:02d}'.format(r) for r in range(1, 21)]
        elif usage_modus == 'val':
            recordings = ['R{:02d}'.format(r) for r in range(21, 26)]
        elif usage_modus == 'test':
            recordings = ['R{:02d}'.format(r) for r in range(26, 31)]
    else:
        recordings = ['R{:02d}'.format(r) for r in range(1, 31)]

    counter_seq = 0
    hist_classes_all = np.zeros(NUM_CLASSES)

    for P in persons:
        if P not in ids:
            print("\nNo Person in expected IDS {}".format(P))
        else:
            if P == 'S11':
                # Selecting the proportions of the train, val or testing according to the quentity of
                # recordings per subject, as there are not equal number of recordings per subject
                # see dataset for checking the recording files per subject
                if identity_bool:
                    if usage_modus == 'train':
                        recordings = ['R{:02d}'.format(r) for r in range(1, 10)]
                    elif usage_modus == 'val':
                        recordings = ['R{:02d}'.format(r) for r in range(10, 12)]
                    elif usage_modus == 'test':
                        recordings = ['R{:02d}'.format(r) for r in range(12, 15)]
                else:
                    recordings = ['R{:02d}'.format(r) for r in range(1, 31)]
            elif P == 'S12':
                if identity_bool:
                    if usage_modus == 'train':
                        recordings = ['R{:02d}'.format(r) for r in range(1, 25)]
                    elif usage_modus == 'val':
                        recordings = ['R{:02d}'.format(r) for r in range(25, 28)]
                    elif usage_modus == 'test':
                        recordings = ['R{:02d}'.format(r) for r in range(28, 31)]
                else:
                    recordings = ['R{:02d}'.format(r) for r in range(1, 31)]
            else:
                if identity_bool:
                    if usage_modus == 'train':
                        recordings = ['R{:02d}'.format(r) for r in range(1, 21)]
                    elif usage_modus == 'val':
                        recordings = ['R{:02d}'.format(r) for r in range(21, 26)]
                    elif usage_modus == 'test':
                        recordings = ['R{:02d}'.format(r) for r in range(26, 31)]
                else:
                    recordings = ['R{:02d}'.format(r) for r in range(1, 31)]
            for R in recordings:
                # All of these if-cases are coming due to the naming of the recordings in the data.
                # Not all the subjects have the same
                # annotated recordings, nor annotators, nor annotations runs, nor scenarios.
                # these will include all of the recordings for the subjects
                if P in ["S01", "S02", "S03", "S04", "S05", "S06"]:
                    S = "L01"
                else:
                    S = SCENARIO[R]
                for N in repetition:
                    annotator_file = annotator[P]
                    if P == 'S07' and SCENARIO[R] == 'L01':
                        annotator_file = "A03"
                    if P == 'S14' and SCENARIO[R] == 'L03':
                        annotator_file = "A19"
                    if P == 'S11' and SCENARIO[R] == 'L01':
                        annotator_file = "A03"
                    if P == 'S11' and R in ['R04', 'R08', 'R09', 'R10', 'R11', 'R12', 'R13', 'R15']:
                        annotator_file = "A02"
                    if P == 'S13' and R in ['R28']:
                        annotator_file = "A01"
                    if P == 'S13' and R in ['R29', 'R30']:
                        annotator_file = "A11"
                    if P == 'S09' and R in ['R28', 'R29']:
                        annotator_file = "A01"
                    if P == 'S09' and R in ['R21', 'R22', 'R23', 'R24', 'R25']:
                        annotator_file = "A11"

                    file_name_norm = "{}/{}_{}_{}_{}_{}_norm_data.csv".format(P, S, P, R, annotator_file, N)
                    file_name_label = "{}/{}_{}_{}_{}_{}_labels.csv".format(P, S, P, R, annotator_file, N)

                    try:
                        #getting data
                        #print(FOLDER_PATH + file_name_norm)
                        data = csv_reader.reader_data(FOLDER_PATH + file_name_norm)
                        print("\nFiles loaded in modus {}\n{}".format(usage_modus, file_name_norm))
                        data = select_columns_opp(data)
                        print("Columns selected")
                    except:
                        print("\n In generating data, No file {}".format(FOLDER_PATH + file_name_norm))
                        continue

                    try:
                        #Getting labels and attributes
                        labels = csv_reader.reader_labels(FOLDER_PATH + file_name_label)
                        class_labels = np.where(labels[:, 0] == 7)[0]

                        # Deleting rows containing the "none" class
                        data = np.delete(data, class_labels, 0)
                        labels = np.delete(labels, class_labels, 0)

                        # halving the frequency
                        if half:
                            downsampling = range(0, data.shape[0], 2)
                            data = data[downsampling]
                            labels = labels[downsampling]
                            data_t, data_x, data_y = divide_x_y(data)
                            del data_t
                        else:
                            data_t, data_x, data_y = divide_x_y(data)
                            del data_t

                    except:
                        print("\n In generating data, Error getting the data {}".format(FOLDER_PATH + file_name_norm))
                        continue

                    try:
                        # checking if annotations are consistent
                        data_x = normalize(data_x)
                        if np.sum(data_y == labels[:, 0]) == data_y.shape[0]:

                            # Sliding window approach
                            print("Starting sliding window")
                            X, y, y_all = opp_sliding_window(data_x, labels.astype(int),
                                                             sliding_window_length,
                                                             sliding_window_step, label_pos_end = False)
                            print("Windows are extracted")

                            # Statistics
                            hist_classes = np.bincount(y[:, 0], minlength=NUM_CLASSES)
                            hist_classes_all += hist_classes
                            print("Number of seq per class {}".format(hist_classes_all))
                            #print(X.shape[0])

                            for f in range(X.shape[0]):
                                #print("beep")
                                try:

                                    sys.stdout.write('\r' + 'Creating sequence file '
                                                            'number {} with id {}'.format(f, counter_seq))
                                    sys.stdout.flush()
                                    #print("boop")

                                    # print "Creating sequence file number {} with id {}".format(f, counter_seq)
                                    seq = np.reshape(X[f], newshape = (1, X.shape[1], X.shape[2]))
                                    seq = np.require(seq, dtype=np.float)

                                    # Storing the sequences
                                    #print(os.path.join(data_dir, 'seq_{0:06}.pkl'.format(counter_seq)))
                                    obj = {"data": seq, "label": y[f], "labels": y_all[f],
                                           "identity": labels_persons[P]}
                                    f = open(os.path.join(data_dir, 'seq_{0:06}.pkl'.format(counter_seq)), 'wb')
                                    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
                                    f.close()

                                    counter_seq += 1
                                except:
                                    raise('\nError adding the seq')

                            print("\nCorrect data extraction from {}".format(FOLDER_PATH + file_name_norm))

                            del data
                            del data_x
                            del data_y
                            del X
                            del labels
                            del class_labels

                        else:
                            print("\nNot consisting annotation in  {}".format(file_name_norm))
                            continue

                    except:
                        print("\n In generating data, No file {}".format(FOLDER_PATH + file_name_norm))
            
    return



def generate_CSV(csv_dir, data_dir):
    '''
    Generate CSV file with path to all (Training) of the segmented sequences
    This is done for the DATALoader for Torch, using a CSV file with all the paths from the extracted
    sequences.

    @param csv_dir: Path to the dataset
    @param data_dir: Path of the training data
    '''
    f = []
    for dirpath, dirnames, filenames in os.walk(data_dir):
        for n in range(len(filenames)):
            f.append(data_dir + 'seq_{0:06}.pkl'.format(n))

    np.savetxt(csv_dir, f, delimiter="\n", fmt='%s')
    
    return

def generate_CSV_final(csv_dir, data_dir1, data_dir2):
    '''
    Generate CSV file with path to all (Training and Validation) of the segmented sequences
    This is done for the DATALoader for Torch, using a CSV file with all the paths from the extracted
    sequences.

    @param csv_dir: Path to the dataset
    @param data_dir1: Path of the training data
    @param data_dir2: Path of the validation data
    '''
    f = []
    for dirpath, dirnames, filenames in os.walk(data_dir1):
        for n in range(len(filenames)):
            f.append(data_dir1 + 'seq_{0:06}.pkl'.format(n))

    for dirpath, dirnames, filenames in os.walk(data_dir2):
        for n in range(len(filenames)):
            f.append(data_dir1 + 'seq_{0:06}.pkl'.format(n))

    np.savetxt(csv_dir, f, delimiter="\n", fmt='%s')

    return

def general_statistics(ids):
    '''
    Computing min duration of activity classes

    @param ids: IDS for subjects in the dataset.
    '''
    #compute_max_min(ids)
    attr_check = 19
    min_durations = compute_min_num_samples(ids, boolean_classes=False, attr=attr_check)

    compute_statistics_samples(ids, boolean_classes=False, attr=attr_check)

    print("Minimum per class {}".format(min_durations))
    print("Minimum ordered {}".format(np.sort(min_durations)))
    return


def create_dataset(half=False):
    '''
    create dataset
    - Segmentation
    - Storing sequences

    @param half: set for creating dataset with half the frequence.
    '''

    train_ids = ["S01", "S02", "S03", "S04", "S07", "S08", "S09", "S10", "S15", "S16"]
    train_final_ids = ["S01", "S02", "S03", "S04", "S05", "S07", "S08", "S09", "S10" "S11", "S12", "S15", "S16"]
    val_ids = ["S05", "S11", "S12"]
    test_ids = ["S06", "S13", "S14"]

    all_data = ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11", "S12", "S13", "S14"]

    #general_statistics(train_ids)

    if half:
        "Path to the segmented sequences"
        base_directory = '/data/dkroen/dataset/mocap/'
        sliding_window_length = 100
        sliding_window_step = 12
    else:
        "Path to the segmented sequences"
        base_directory = '/data/dkroen/dataset/mocap/'
        sliding_window_length = 200
        sliding_window_step = 25

    data_dir_train = base_directory + 'sequences_train/'
    data_dir_val = base_directory + 'sequences_val/'
    data_dir_test = base_directory + 'sequences_test/'

    generate_data(train_ids, sliding_window_length=sliding_window_length,
                  sliding_window_step=sliding_window_step, data_dir=data_dir_train, half=half, usage_modus='train')
    generate_data(val_ids, sliding_window_length=sliding_window_length,
                  sliding_window_step=sliding_window_step, data_dir=data_dir_val, half=half)
    generate_data(test_ids, sliding_window_length=sliding_window_length,
                  sliding_window_step=sliding_window_step, data_dir=data_dir_test, half=half)

    generate_CSV(base_directory + "train.csv", data_dir_train)
    generate_CSV(base_directory + "val.csv", data_dir_val)
    generate_CSV(base_directory + "test.csv", data_dir_test)
    generate_CSV_final(base_directory + "train_final.csv", data_dir_train, data_dir_val)

    return



if __name__ == '__main__':
    # Creating dataset for LARA Mocap 200Hz or LARA Mocap 100Hz
    # Set the path to where the segmented windows will be located
    # This path will be needed for the main.py

    # Dataset (extracted segmented windows) will be stored in a given folder by the user,
    # However, inside the folder, there shall be the subfolders (sequences_train, sequences_val, sequences_test)
    # These folders and subfolfders gotta be created manually by the user
    # This as a sort of organisation for the dataset
    # MoCap_dataset/sequences_train
    # MoCap_dataset/sequences_val
    # MoCap_dataset/sequences_test

    create_dataset(half=True)

    print("Done")
