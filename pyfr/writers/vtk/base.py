from collections import defaultdict
from pathlib import Path

import numpy as np

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.shapes import BaseShape
from pyfr.util import subclass_where
from pyfr.writers import BaseWriter


def interpolate_pts(op, pts):
    ipts = op.astype(pts.dtype) @ pts.reshape(op.shape[1], -1)
    ipts = ipts.reshape(op.shape[0], *pts.shape[1:])

    return ipts


class BaseVTKWriter(BaseWriter):
    # Supported file types and extensions
    name = 'vtk'
    extn = ['.vtu', '.pvtu']

    # Type of export (volume/boundary/STL)
    type = None

    # If to output curvature or partition number data
    output_curved = False
    output_partition = False

    # VTK high-order types
    _vtk_types_ho = {'tri': 69, 'quad': 70, 'tet': 71, 'pri': 73, 'hex': 72}

    # Mappings betwen the node ordering of PyFR and that of VTK
    _nodemaps = {
        ('quad', 4): [0, 1, 3, 2],
        ('quad', 9): [0, 2, 8, 6, 1, 5, 7, 3, 4],
        ('quad', 16): [0, 3, 15, 12, 1, 2, 7, 11, 13, 14, 4, 8, 5, 6, 9, 10],
        ('quad', 25): [0, 4, 24, 20, 1, 2, 3, 9, 14, 19, 21, 22, 23, 5, 10, 15,
                       6, 7, 8, 11, 12, 13, 16, 17, 18],
        ('quad', 36): [0, 5, 35, 30, 1, 2, 3, 4, 11, 17, 23, 29, 31, 32, 33,
                       34, 6, 12, 18, 24, 7, 8, 9, 10, 13, 14, 15, 16, 19, 20,
                       21, 22, 25, 26, 27, 28],
        ('quad', 49): [0, 6, 48, 42, 1, 2, 3, 4, 5, 13, 20, 27, 34, 41, 43, 44,
                       45, 46, 47, 7, 14, 21, 28, 35, 8, 9, 10, 11, 12, 15, 16,
                       17, 18, 19, 22, 23, 24, 25, 26, 29, 30, 31, 32, 33, 36,
                       37, 38, 39, 40],
        ('quad', 64): [
            0, 7, 63, 56, 1, 2, 3, 4, 5, 6, 15, 23, 31, 39, 47, 55, 57, 58, 59,
            60, 61, 62, 8, 16, 24, 32, 40, 48, 9, 10, 11, 12, 13, 14, 17, 18,
            19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 33, 34, 35, 36, 37, 38, 41,
            42, 43, 44, 45, 46, 49, 50, 51, 52, 53, 54
        ],
        ('quad', 81): [
            0, 8, 80, 72, 1, 2, 3, 4, 5, 6, 7, 17, 26, 35, 44, 53, 62, 71, 73,
            74, 75, 76, 77, 78, 79, 9, 18, 27, 36, 45, 54, 63, 10, 11, 12, 13,
            14, 15, 16, 19, 20, 21, 22, 23, 24, 25, 28, 29, 30, 31, 32, 33, 34,
            37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 55, 56, 57,
            58, 59, 60, 61, 64, 65, 66, 67, 68, 69, 70
        ],
        ('hex', 8): [0, 1, 3, 2, 4, 5, 7, 6],
        ('hex', 27): [0, 2, 8, 6, 18, 20, 26, 24, 1, 5, 7, 3, 19, 23, 25, 21,
                      9, 11, 17, 15, 12, 14, 10, 16, 4, 22, 13],
        ('hex', 64): [
            0, 3, 15, 12, 48, 51, 63, 60, 1, 2, 7, 11, 13, 14, 4, 8, 49, 50,
            55, 59, 61, 62, 52, 56, 16, 32, 19, 35, 31, 47, 28, 44, 20, 24, 36,
            40, 23, 27, 39, 43, 17, 18, 33, 34, 29, 30, 45, 46, 5, 6, 9, 10,
            53, 54, 57, 58, 21, 22, 25, 26, 37, 38, 41, 42
        ],
        ('hex', 125): [
            0, 4, 24, 20, 100, 104, 124, 120, 1, 2, 3, 9, 14, 19, 21, 22, 23,
            5, 10, 15, 101, 102, 103, 109, 114, 119, 121, 122, 123, 105, 110,
            115, 25, 50, 75, 29, 54, 79, 49, 74, 99, 45, 70, 95, 30, 35, 40,
            55, 60, 65, 80, 85, 90, 34, 39, 44, 59, 64, 69, 84, 89, 94, 26, 27,
            28, 51, 52, 53, 76, 77, 78, 46, 47, 48, 71, 72, 73, 96, 97, 98, 6,
            7, 8, 11, 12, 13, 16, 17, 18, 106, 107, 108, 111, 112, 113, 116,
            117, 118, 31, 32, 33, 36, 37, 38, 41, 42, 43, 56, 57, 58, 61, 62,
            63, 66, 67, 68, 81, 82, 83, 86, 87, 88, 91, 92, 93
        ],
        ('hex', 216): [
            0, 5, 35, 30, 180, 185, 215, 210, 1, 2, 3, 4, 11, 17, 23, 29, 31,
            32, 33, 34, 6, 12, 18, 24, 181, 182, 183, 184, 191, 197, 203, 209,
            211, 212, 213, 214, 186, 192, 198, 204, 36, 72, 108, 144, 41, 77,
            113, 149, 71, 107, 143, 179, 66, 102, 138, 174, 42, 48, 54, 60, 78,
            84, 90, 96, 114, 120, 126, 132, 150, 156, 162, 168, 47, 53, 59, 65,
            83, 89, 95, 101, 119, 125, 131, 137, 155, 161, 167, 173, 37, 38,
            39, 40, 73, 74, 75, 76, 109, 110, 111, 112, 145, 146, 147, 148, 67,
            68, 69, 70, 103, 104, 105, 106, 139, 140, 141, 142, 175, 176, 177,
            178, 7, 8, 9, 10, 13, 14, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28,
            187, 188, 189, 190, 193, 194, 195, 196, 199, 200, 201, 202, 205,
            206, 207, 208, 43, 44, 45, 46, 49, 50, 51, 52, 55, 56, 57, 58, 61,
            62, 63, 64, 79, 80, 81, 82, 85, 86, 87, 88, 91, 92, 93, 94, 97, 98,
            99, 100, 115, 116, 117, 118, 121, 122, 123, 124, 127, 128, 129,
            130, 133, 134, 135, 136, 151, 152, 153, 154, 157, 158, 159, 160,
            163, 164, 165, 166, 169, 170, 171, 172
        ],
        ('hex', 343): [
            0, 6, 48, 42, 294, 300, 342, 336, 1, 2, 3, 4, 5, 13, 20, 27, 34,
            41, 43, 44, 45, 46, 47, 7, 14, 21, 28, 35, 295, 296, 297, 298, 299,
            307, 314, 321, 328, 335, 337, 338, 339, 340, 341, 301, 308, 315,
            322, 329, 49, 98, 147, 196, 245, 55, 104, 153, 202, 251, 97, 146,
            195, 244, 293, 91, 140, 189, 238, 287, 56, 63, 70, 77, 84, 105,
            112, 119, 126, 133, 154, 161, 168, 175, 182, 203, 210, 217, 224,
            231, 252, 259, 266, 273, 280, 62, 69, 76, 83, 90, 111, 118, 125,
            132, 139, 160, 167, 174, 181, 188, 209, 216, 223, 230, 237, 258,
            265, 272, 279, 286, 50, 51, 52, 53, 54, 99, 100, 101, 102, 103,
            148, 149, 150, 151, 152, 197, 198, 199, 200, 201, 246, 247, 248,
            249, 250, 92, 93, 94, 95, 96, 141, 142, 143, 144, 145, 190, 191,
            192, 193, 194, 239, 240, 241, 242, 243, 288, 289, 290, 291, 292, 8,
            9, 10, 11, 12, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 29, 30, 31,
            32, 33, 36, 37, 38, 39, 40, 302, 303, 304, 305, 306, 309, 310, 311,
            312, 313, 316, 317, 318, 319, 320, 323, 324, 325, 326, 327, 330,
            331, 332, 333, 334, 57, 58, 59, 60, 61, 64, 65, 66, 67, 68, 71, 72,
            73, 74, 75, 78, 79, 80, 81, 82, 85, 86, 87, 88, 89, 106, 107, 108,
            109, 110, 113, 114, 115, 116, 117, 120, 121, 122, 123, 124, 127,
            128, 129, 130, 131, 134, 135, 136, 137, 138, 155, 156, 157, 158,
            159, 162, 163, 164, 165, 166, 169, 170, 171, 172, 173, 176, 177,
            178, 179, 180, 183, 184, 185, 186, 187, 204, 205, 206, 207, 208,
            211, 212, 213, 214, 215, 218, 219, 220, 221, 222, 225, 226, 227,
            228, 229, 232, 233, 234, 235, 236, 253, 254, 255, 256, 257, 260,
            261, 262, 263, 264, 267, 268, 269, 270, 271, 274, 275, 276, 277,
            278, 281, 282, 283, 284, 285
        ],
        ('hex', 512): [
            0, 7, 63, 56, 448, 455, 511, 504, 1, 2, 3, 4, 5, 6, 15, 23, 31, 39,
            47, 55, 57, 58, 59, 60, 61, 62, 8, 16, 24, 32, 40, 48, 449, 450,
            451, 452, 453, 454, 463, 471, 479, 487, 495, 503, 505, 506, 507,
            508, 509, 510, 456, 464, 472, 480, 488, 496, 64, 128, 192, 256,
            320, 384, 71, 135, 199, 263, 327, 391, 127, 191, 255, 319, 383,
            447, 120, 184, 248, 312, 376, 440, 72, 80, 88, 96, 104, 112, 136,
            144, 152, 160, 168, 176, 200, 208, 216, 224, 232, 240, 264, 272,
            280, 288, 296, 304, 328, 336, 344, 352, 360, 368, 392, 400, 408,
            416, 424, 432, 79, 87, 95, 103, 111, 119, 143, 151, 159, 167, 175,
            183, 207, 215, 223, 231, 239, 247, 271, 279, 287, 295, 303, 311,
            335, 343, 351, 359, 367, 375, 399, 407, 415, 423, 431, 439, 65, 66,
            67, 68, 69, 70, 129, 130, 131, 132, 133, 134, 193, 194, 195, 196,
            197, 198, 257, 258, 259, 260, 261, 262, 321, 322, 323, 324, 325,
            326, 385, 386, 387, 388, 389, 390, 121, 122, 123, 124, 125, 126,
            185, 186, 187, 188, 189, 190, 249, 250, 251, 252, 253, 254, 313,
            314, 315, 316, 317, 318, 377, 378, 379, 380, 381, 382, 441, 442,
            443, 444, 445, 446, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22,
            25, 26, 27, 28, 29, 30, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45,
            46, 49, 50, 51, 52, 53, 54, 457, 458, 459, 460, 461, 462, 465, 466,
            467, 468, 469, 470, 473, 474, 475, 476, 477, 478, 481, 482, 483,
            484, 485, 486, 489, 490, 491, 492, 493, 494, 497, 498, 499, 500,
            501, 502, 73, 74, 75, 76, 77, 78, 81, 82, 83, 84, 85, 86, 89, 90,
            91, 92, 93, 94, 97, 98, 99, 100, 101, 102, 105, 106, 107, 108, 109,
            110, 113, 114, 115, 116, 117, 118, 137, 138, 139, 140, 141, 142,
            145, 146, 147, 148, 149, 150, 153, 154, 155, 156, 157, 158, 161,
            162, 163, 164, 165, 166, 169, 170, 171, 172, 173, 174, 177, 178,
            179, 180, 181, 182, 201, 202, 203, 204, 205, 206, 209, 210, 211,
            212, 213, 214, 217, 218, 219, 220, 221, 222, 225, 226, 227, 228,
            229, 230, 233, 234, 235, 236, 237, 238, 241, 242, 243, 244, 245,
            246, 265, 266, 267, 268, 269, 270, 273, 274, 275, 276, 277, 278,
            281, 282, 283, 284, 285, 286, 289, 290, 291, 292, 293, 294, 297,
            298, 299, 300, 301, 302, 305, 306, 307, 308, 309, 310, 329, 330,
            331, 332, 333, 334, 337, 338, 339, 340, 341, 342, 345, 346, 347,
            348, 349, 350, 353, 354, 355, 356, 357, 358, 361, 362, 363, 364,
            365, 366, 369, 370, 371, 372, 373, 374, 393, 394, 395, 396, 397,
            398, 401, 402, 403, 404, 405, 406, 409, 410, 411, 412, 413, 414,
            417, 418, 419, 420, 421, 422, 425, 426, 427, 428, 429, 430, 433,
            434, 435, 436, 437, 438
        ],
        ('hex', 729): [
            0, 8, 80, 72, 648, 656, 728, 720, 1, 2, 3, 4, 5, 6, 7, 17, 26, 35,
            44, 53, 62, 71, 73, 74, 75, 76, 77, 78, 79, 9, 18, 27, 36, 45, 54,
            63, 649, 650, 651, 652, 653, 654, 655, 665, 674, 683, 692, 701,
            710, 719, 721, 722, 723, 724, 725, 726, 727, 657, 666, 675, 684,
            693, 702, 711, 81, 162, 243, 324, 405, 486, 567, 89, 170, 251, 332,
            413, 494, 575, 161, 242, 323, 404, 485, 566, 647, 153, 234, 315,
            396, 477, 558, 639, 90, 99, 108, 117, 126, 135, 144, 171, 180, 189,
            198, 207, 216, 225, 252, 261, 270, 279, 288, 297, 306, 333, 342,
            351, 360, 369, 378, 387, 414, 423, 432, 441, 450, 459, 468, 495,
            504, 513, 522, 531, 540, 549, 576, 585, 594, 603, 612, 621, 630,
            98, 107, 116, 125, 134, 143, 152, 179, 188, 197, 206, 215, 224,
            233, 260, 269, 278, 287, 296, 305, 314, 341, 350, 359, 368, 377,
            386, 395, 422, 431, 440, 449, 458, 467, 476, 503, 512, 521, 530,
            539, 548, 557, 584, 593, 602, 611, 620, 629, 638, 82, 83, 84, 85,
            86, 87, 88, 163, 164, 165, 166, 167, 168, 169, 244, 245, 246, 247,
            248, 249, 250, 325, 326, 327, 328, 329, 330, 331, 406, 407, 408,
            409, 410, 411, 412, 487, 488, 489, 490, 491, 492, 493, 568, 569,
            570, 571, 572, 573, 574, 154, 155, 156, 157, 158, 159, 160, 235,
            236, 237, 238, 239, 240, 241, 316, 317, 318, 319, 320, 321, 322,
            397, 398, 399, 400, 401, 402, 403, 478, 479, 480, 481, 482, 483,
            484, 559, 560, 561, 562, 563, 564, 565, 640, 641, 642, 643, 644,
            645, 646, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 24, 25,
            28, 29, 30, 31, 32, 33, 34, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48,
            49, 50, 51, 52, 55, 56, 57, 58, 59, 60, 61, 64, 65, 66, 67, 68, 69,
            70, 658, 659, 660, 661, 662, 663, 664, 667, 668, 669, 670, 671,
            672, 673, 676, 677, 678, 679, 680, 681, 682, 685, 686, 687, 688,
            689, 690, 691, 694, 695, 696, 697, 698, 699, 700, 703, 704, 705,
            706, 707, 708, 709, 712, 713, 714, 715, 716, 717, 718, 91, 92, 93,
            94, 95, 96, 97, 100, 101, 102, 103, 104, 105, 106, 109, 110, 111,
            112, 113, 114, 115, 118, 119, 120, 121, 122, 123, 124, 127, 128,
            129, 130, 131, 132, 133, 136, 137, 138, 139, 140, 141, 142, 145,
            146, 147, 148, 149, 150, 151, 172, 173, 174, 175, 176, 177, 178,
            181, 182, 183, 184, 185, 186, 187, 190, 191, 192, 193, 194, 195,
            196, 199, 200, 201, 202, 203, 204, 205, 208, 209, 210, 211, 212,
            213, 214, 217, 218, 219, 220, 221, 222, 223, 226, 227, 228, 229,
            230, 231, 232, 253, 254, 255, 256, 257, 258, 259, 262, 263, 264,
            265, 266, 267, 268, 271, 272, 273, 274, 275, 276, 277, 280, 281,
            282, 283, 284, 285, 286, 289, 290, 291, 292, 293, 294, 295, 298,
            299, 300, 301, 302, 303, 304, 307, 308, 309, 310, 311, 312, 313,
            334, 335, 336, 337, 338, 339, 340, 343, 344, 345, 346, 347, 348,
            349, 352, 353, 354, 355, 356, 357, 358, 361, 362, 363, 364, 365,
            366, 367, 370, 371, 372, 373, 374, 375, 376, 379, 380, 381, 382,
            383, 384, 385, 388, 389, 390, 391, 392, 393, 394, 415, 416, 417,
            418, 419, 420, 421, 424, 425, 426, 427, 428, 429, 430, 433, 434,
            435, 436, 437, 438, 439, 442, 443, 444, 445, 446, 447, 448, 451,
            452, 453, 454, 455, 456, 457, 460, 461, 462, 463, 464, 465, 466,
            469, 470, 471, 472, 473, 474, 475, 496, 497, 498, 499, 500, 501,
            502, 505, 506, 507, 508, 509, 510, 511, 514, 515, 516, 517, 518,
            519, 520, 523, 524, 525, 526, 527, 528, 529, 532, 533, 534, 535,
            536, 537, 538, 541, 542, 543, 544, 545, 546, 547, 550, 551, 552,
            553, 554, 555, 556, 577, 578, 579, 580, 581, 582, 583, 586, 587,
            588, 589, 590, 591, 592, 595, 596, 597, 598, 599, 600, 601, 604,
            605, 606, 607, 608, 609, 610, 613, 614, 615, 616, 617, 618, 619,
            622, 623, 624, 625, 626, 627, 628, 631, 632, 633, 634, 635, 636,
            637
        ],
        ('tri', 3): [2, 0, 1],
        ('tri', 6): [5, 0, 2, 3, 1, 4],
        ('tri', 10): [9, 0, 3, 7, 4, 1, 2, 6, 8, 5],
        ('tri', 15): [14, 0, 4, 12, 9, 5, 1, 2, 3, 8, 11, 13, 10, 6, 7],
        ('tri', 21): [20, 0, 5, 18, 15, 11, 6, 1, 2, 3, 4, 10, 14, 17, 19, 16,
                      7, 9, 12, 8, 13],
        ('tri', 28): [27, 0, 6, 25, 22, 18, 13, 7, 1, 2, 3, 4, 5, 12, 17, 21,
                      24, 26, 23, 8, 11, 19, 14, 9, 10, 16, 20, 15],
        ('tri', 36): [35, 0, 7, 33, 30, 26, 21, 15, 8, 1, 2, 3, 4, 5, 6, 14,
                      20, 25, 29, 32, 34, 31, 9, 13, 27, 22, 16, 10, 11, 12,
                      19, 24, 28, 23, 17, 18],
        ('tri', 45): [44, 0, 8, 42, 39, 35, 30, 24, 17, 9, 1, 2, 3, 4, 5, 6, 7,
                      16, 23, 29, 34, 38, 41, 43, 40, 10, 15, 36, 31, 25, 18,
                      11, 12, 13, 14, 22, 28, 33, 37, 32, 19, 21, 26, 20, 27],
        ('tet', 4): [3, 0, 1, 2],
        ('tet', 10): [9, 0, 2, 5, 6, 1, 7, 8, 3, 4],
        ('tet', 20): [19, 0, 3, 9, 16, 10, 1, 2, 12, 17, 18, 15, 4, 7, 6, 8,
                      13, 5, 14, 11],
        ('tet', 35): [34, 0, 4, 14, 31, 25, 15, 1, 2, 3, 18, 27, 32, 33, 30,
                      24, 5, 9, 12, 8, 11, 13, 28, 19, 22, 7, 10, 6, 29, 23,
                      21, 26, 17, 16, 20],
        ('tet', 56): [55, 0, 5, 20, 52, 46, 36, 21, 1, 2, 3, 4, 25, 39, 48,
                      53, 54, 51, 45, 35, 6, 11, 15, 18, 10, 14, 17, 19, 49,
                      26, 33, 40, 30, 43, 9, 16, 7, 13, 12, 8, 50, 34, 29, 44,
                      32, 42, 47, 24, 22, 38, 23, 37, 41, 27, 28, 31],
        ('tet', 84): [
            83, 0, 6, 27, 80, 74, 64, 49, 28, 1, 2, 3, 4, 5, 33, 53, 67, 76,
            81, 82, 79, 73, 63, 48, 7, 13, 18, 22, 25, 12, 17, 21, 24, 26, 77,
            34, 46, 68, 54, 39, 43, 61, 71, 58, 11, 23, 8, 16, 20, 19, 14, 9,
            10, 15, 78, 47, 38, 72, 62, 45, 42, 57, 70, 60, 75, 32, 29, 66, 52,
            31, 30, 50, 65, 51, 69, 35, 37, 44, 55, 36, 56, 59, 40, 41
        ],
        ('tet', 120): [
            119, 0, 7, 35, 116, 110, 100, 85, 64, 36, 1, 2, 3, 4, 5, 6, 42, 69,
            89, 103, 112, 117, 118, 115, 109, 99, 84, 63, 8, 15, 21, 26, 30,
            33, 14, 20, 25, 29, 32, 34, 113, 43, 61, 104, 90, 70, 49, 54, 58,
            82, 97, 107, 94, 75, 79, 13, 31, 9, 19, 24, 28, 27, 22, 16, 10, 11,
            12, 18, 23, 17, 114, 62, 48, 108, 98, 83, 60, 57, 53, 74, 93, 106,
            96, 81, 78, 111, 41, 37, 102, 88, 68, 40, 39, 38, 65, 86, 101, 87,
            67, 66, 105, 44, 47, 59, 91, 71, 45, 46, 73, 92, 95, 80, 50, 55,
            52, 56, 76, 51, 77, 72
        ],
        ('tet', 165): [
            164, 0, 8, 44, 161, 155, 145, 130, 109, 81, 45, 1, 2, 3, 4, 5, 6,
            7, 52, 87, 114, 134, 148, 157, 162, 163, 160, 154, 144, 129, 108,
            80, 9, 17, 24, 30, 35, 39, 42, 16, 23, 29, 34, 38, 41, 43, 158, 53,
            78, 149, 135, 115, 88, 60, 66, 71, 75, 106, 127, 142, 152, 139, 94,
            103, 120, 99, 124, 15, 40, 10, 22, 28, 33, 37, 36, 31, 25, 18, 11,
            12, 13, 14, 21, 32, 19, 27, 26, 20, 159, 79, 59, 153, 143, 128,
            107, 77, 74, 70, 65, 93, 119, 138, 151, 141, 105, 98, 126, 102,
            123, 156, 51, 46, 147, 133, 113, 86, 50, 49, 48, 47, 82, 110, 131,
            146, 132, 85, 83, 112, 84, 111, 150, 54, 58, 76, 136, 116, 89, 55,
            56, 57, 92, 118, 137, 140, 125, 104, 61, 67, 72, 64, 69, 73, 121,
            95, 100, 63, 68, 62, 122, 101, 97, 117, 91, 90, 96
        ],
        ('pri', 6): [0, 1, 2, 3, 4, 5],
        ('pri', 18): [0, 2, 5, 12, 14, 17, 1, 4, 3, 13, 16, 15, 6, 8, 11, 7,
                      10, 9],
        ('pri', 40): [0, 3, 9, 30, 33, 39, 1, 2, 6, 8, 7, 4, 31, 32, 36, 38,
                      37, 34, 10, 20, 13, 23, 19, 29, 5, 35, 11, 12, 21, 22,
                      16, 18, 26, 28, 17, 14, 27, 24, 15, 25],
        ('pri', 75): [
            0, 4, 14, 60, 64, 74, 1, 2, 3, 8, 11, 13, 12, 9, 5, 61, 62, 63, 68,
            71, 73, 72, 69, 65, 15, 30, 45, 19, 34, 49, 29, 44, 59, 6, 7, 10,
            66, 67, 70, 16, 17, 18, 31, 32, 33, 46, 47, 48, 23, 26, 28, 38, 41,
            43, 53, 56, 58, 27, 24, 20, 42, 39, 35, 57, 54, 50, 21, 22, 25, 36,
            37, 40, 51, 52, 55
        ],
        ('pri', 126): [
            0, 5, 20, 105, 110, 125, 1, 2, 3, 4, 10, 14, 17, 19, 18, 15, 11, 6,
            106, 107, 108, 109, 115, 119, 122, 124, 123, 120, 116, 111, 21, 42,
            63, 84, 26, 47, 68, 89, 41, 62, 83, 104, 7, 8, 9, 12, 13, 16, 112,
            113, 114, 117, 118, 121, 22, 23, 24, 25, 43, 44, 45, 46, 64, 65,
            66, 67, 85, 86, 87, 88, 31, 35, 38, 40, 52, 56, 59, 61, 73, 77, 80,
            82, 94, 98, 101, 103, 39, 36, 32, 27, 60, 57, 53, 48, 81, 78, 74,
            69, 102, 99, 95, 90, 28, 29, 30, 33, 34, 37, 49, 50, 51, 54, 55,
            58, 70, 71, 72, 75, 76, 79, 91, 92, 93, 96, 97, 100
        ],
        ('pri', 196): [
            0, 6, 27, 168, 174, 195, 1, 2, 3, 4, 5, 12, 17, 21, 24, 26, 25, 22,
            18, 13, 7, 169, 170, 171, 172, 173, 180, 185, 189, 192, 194, 193,
            190, 186, 181, 175, 28, 56, 84, 112, 140, 34, 62, 90, 118, 146, 55,
            83, 111, 139, 167, 8, 9, 10, 11, 14, 15, 16, 19, 20, 23, 176, 177,
            178, 179, 182, 183, 184, 187, 188, 191, 29, 30, 31, 32, 33, 57, 58,
            59, 60, 61, 85, 86, 87, 88, 89, 113, 114, 115, 116, 117, 141, 142,
            143, 144, 145, 40, 45, 49, 52, 54, 68, 73, 77, 80, 82, 96, 101,
            105, 108, 110, 124, 129, 133, 136, 138, 152, 157, 161, 164, 166,
            53, 50, 46, 41, 35, 81, 78, 74, 69, 63, 109, 106, 102, 97, 91, 137,
            134, 130, 125, 119, 165, 162, 158, 153, 147, 36, 37, 38, 39, 42,
            43, 44, 47, 48, 51, 64, 65, 66, 67, 70, 71, 72, 75, 76, 79, 92, 93,
            94, 95, 98, 99, 100, 103, 104, 107, 120, 121, 122, 123, 126, 127,
            128, 131, 132, 135, 148, 149, 150, 151, 154, 155, 156, 159, 160,
            163
        ],
        ('pri', 288): [
            0, 7, 35, 252, 259, 287, 1, 2, 3, 4, 5, 6, 14, 20, 25, 29, 32, 34,
            33, 30, 26, 21, 15, 8, 253, 254, 255, 256, 257, 258, 266, 272, 277,
            281, 284, 286, 285, 282, 278, 273, 267, 260, 36, 72, 108, 144, 180,
            216, 43, 79, 115, 151, 187, 223, 71, 107, 143, 179, 215, 251, 9,
            10, 11, 12, 13, 16, 17, 18, 19, 22, 23, 24, 27, 28, 31, 261, 262,
            263, 264, 265, 268, 269, 270, 271, 274, 275, 276, 279, 280, 283,
            37, 38, 39, 40, 41, 42, 73, 74, 75, 76, 77, 78, 109, 110, 111, 112,
            113, 114, 145, 146, 147, 148, 149, 150, 181, 182, 183, 184, 185,
            186, 217, 218, 219, 220, 221, 222, 50, 56, 61, 65, 68, 70, 86, 92,
            97, 101, 104, 106, 122, 128, 133, 137, 140, 142, 158, 164, 169,
            173, 176, 178, 194, 200, 205, 209, 212, 214, 230, 236, 241, 245,
            248, 250, 69, 66, 62, 57, 51, 44, 105, 102, 98, 93, 87, 80, 141,
            138, 134, 129, 123, 116, 177, 174, 170, 165, 159, 152, 213, 210,
            206, 201, 195, 188, 249, 246, 242, 237, 231, 224, 45, 46, 47, 48,
            49, 52, 53, 54, 55, 58, 59, 60, 63, 64, 67, 81, 82, 83, 84, 85, 88,
            89, 90, 91, 94, 95, 96, 99, 100, 103, 117, 118, 119, 120, 121,
            124, 125, 126, 127, 130, 131, 132, 135, 136, 139, 153, 154, 155,
            156, 157, 160, 161, 162, 163, 166, 167, 168, 171, 172, 175, 189,
            190, 191, 192, 193, 196, 197, 198, 199, 202, 203, 204, 207, 208,
            211, 225, 226, 227, 228, 229, 232, 233, 234, 235, 238, 239, 240,
            243, 244, 247
        ],
        ('pri', 405): [
            0, 8, 44, 360, 368, 404, 1, 2, 3, 4, 5, 6, 7, 16, 23, 29, 34, 38,
            41, 43, 42, 39, 35, 30, 24, 17, 9, 361, 362, 363, 364, 365, 366,
            367, 376, 383, 389, 394, 398, 401, 403, 402, 399, 395, 390, 384,
            377, 369, 45, 90, 135, 180, 225, 270, 315, 53, 98, 143, 188, 233,
            278, 323, 89, 134, 179, 224, 269, 314, 359, 10, 11, 12, 13, 14, 15,
            18, 19, 20, 21, 22, 25, 26, 27, 28, 31, 32, 33, 36, 37, 40, 370,
            371, 372, 373, 374, 375, 378, 379, 380, 381, 382, 385, 386, 387,
            388, 391, 392, 393, 396, 397, 400, 46, 47, 48, 49, 50, 51, 52, 91,
            92, 93, 94, 95, 96, 97, 136, 137, 138, 139, 140, 141, 142, 181,
            182, 183, 184, 185, 186, 187, 226, 227, 228, 229, 230, 231, 232,
            271, 272, 273, 274, 275, 276, 277, 316, 317, 318, 319, 320, 321,
            322, 61, 68, 74, 79, 83, 86, 88, 106, 113, 119, 124, 128, 131, 133,
            151, 158, 164, 169, 173, 176, 178, 196, 203, 209, 214, 218, 221,
            223, 241, 248, 254, 259, 263, 266, 268, 286, 293, 299, 304, 308,
            311, 313, 331, 338, 344, 349, 353, 356, 358, 87, 84, 80, 75, 69,
            62, 54, 132, 129, 125, 120, 114, 107, 99, 177, 174, 170, 165, 159,
            152, 144, 222, 219, 215, 210, 204, 197, 189, 267, 264, 260, 255,
            249, 242, 234, 312, 309, 305, 300, 294, 287, 279, 357, 354, 350,
            345, 339, 332, 324, 55, 56, 57, 58, 59, 60, 63, 64, 65, 66, 67, 70,
            71, 72, 73, 76, 77, 78, 81, 82, 85, 100, 101, 102, 103, 104, 105,
            108, 109, 110, 111, 112, 115, 116, 117, 118, 121, 122, 123, 126,
            127, 130, 145, 146, 147, 148, 149, 150, 153, 154, 155, 156, 157,
            160, 161, 162, 163, 166, 167, 168, 171, 172, 175, 190, 191, 192,
            193, 194, 195, 198, 199, 200, 201, 202, 205, 206, 207, 208, 211,
            212, 213, 216, 217, 220, 235, 236, 237, 238, 239, 240, 243, 244,
            245, 246, 247, 250, 251, 252, 253, 256, 257, 258, 261, 262, 265,
            280, 281, 282, 283, 284, 285, 288, 289, 290, 291, 292, 295, 296,
            297, 298, 301, 302, 303, 306, 307, 310, 325, 326, 327, 328, 329,
            330, 333, 334, 335, 336, 337, 340, 341, 342, 343, 346, 347, 348,
            351, 352, 355
        ]
    }

    def __init__(self, meshf, pname=None, *, prec='single', order=None,
                 divisor=None, fields=[]):
        super().__init__(meshf, pname)

        self.dtype = np.dtype(prec).type
        self.fields = fields

        # Divisor for each type element
        self.etypes_div = defaultdict(lambda: self.divisor)

        # Choose whether to output subdivided cells or high order VTK cells
        if order or divisor is None:
            self.ho_output = True
            self.divisor = order
            self.vtkfile_version = '2.1'
            self._get_npts_ncells_nnodes = self._get_npts_ncells_nnodes_ho
        else:
            self.ho_output = False
            self.divisor = divisor
            self.vtkfile_version = '1.0'
            self._get_npts_ncells_nnodes = self._get_npts_ncells_nnodes_lin

    def _pre_proc_fields_soln(self, soln):
        ecls = self.elementscls
        nvars = len(ecls.privars(self.ndims, self.cfg))

        # Convert the solution to primitive variables
        fields = ecls.con_to_pri(soln[:nvars], self.cfg)

        # Convert any solution gradients to primitive variables
        if self._gradients:
            diff_cons = soln[nvars:].reshape(nvars, -1, *soln.shape[1:])
            diff_pri = ecls.diff_con_to_pri(soln[:nvars], diff_cons, self.cfg)

            fields += [f for gf in diff_pri for f in gf]

        return np.array(fields)

    def _pre_proc_fields_scal(self, soln):
        return soln

    def _post_proc_fields_soln(self, vsoln):
        # Prepare the fields
        fields = []
        for vnames in self._vtk_vars.values():
            ix = [self._soln_fields.index(vn) for vn in vnames]

            fields.append(vsoln[ix])

        return fields

    def _post_proc_fields_scal(self, vsoln):
        return [vsoln[self._soln_fields.index(k)] for k in self._vtk_vars]

    def _get_npts_ncells_nnodes_lin(self, etype, neles):
        div = self.etypes_div[etype]

        # Get the number of shape points
        shapecls = subclass_where(BaseShape, name=etype)
        npts = shapecls.npts_from_order(div)*neles

        # Get the number of subdivided nodes
        subdv = subclass_where(BaseShapeSubDiv, name=etype)(div)
        ncells = len(subdv.subcells)*neles
        nnodes = len(subdv.subnodes)*neles

        return npts, ncells, nnodes

    def _get_npts_ncells_nnodes_ho(self, etype, neles):
        # Fallback to subdivision for pyramids
        if etype == 'pyr':
            return self._get_npts_ncells_nnodes_lin(etype, neles)

        # Get the shape and sub division classes
        shapecls = subclass_where(BaseShape, name=etype)

        # Total number of vis points
        npts = neles*shapecls.npts_from_order(self.etypes_div[etype])

        return npts, neles, npts

    def _get_array_attrs(self, etype=None, neles=None):
        vvars = self._vtk_vars

        # Floating point data type and size
        dtype = 'Float32' if self.dtype == np.float32 else 'Float64'
        dsize = np.dtype(self.dtype).itemsize

        # Base array attributes
        attrs = [('', dtype, '3'), ('connectivity', 'Int64', ''),
                 ('offsets', 'Int64', ''), ('types', 'UInt8', '')]

        if self.output_curved:
            attrs.append(('Curved', 'UInt8', '1'))

        if self.output_partition:
            attrs.append(('Partition', 'Int32', '1'))

        for fname, varnames in vvars.items():
            attrs.append((fname.title(), dtype, str(len(varnames))))

        if etype and neles:
            npts, ncells, nnodes = self._get_npts_ncells_nnodes(etype, neles)
            nb = npts*dsize

            sizes = [3*nb, 8*nnodes, 8*ncells, ncells]

            if self.output_curved:
                sizes.append(ncells)

            if self.output_partition:
                sizes.append(4*ncells)

            sizes.extend(len(varnames)*nb for varnames in vvars.values())

            return tuple((*a, s) for a, s in zip(attrs, sizes))
        else:
            return attrs

    def _load_soln(self, *args, **kwargs):
        super()._load_soln(*args, **kwargs)

        # Get the fields in the data set
        dfields = self.stats.get('data', 'fields').split(',')

        # Ensure a divisor has been set
        if self.divisor is None:
            self.divisor = self.cfg.getint('solver', 'order')

            if self.ho_output:
                self.etypes_div['pyr'] += 2

        # Solutions need a separate processing pipeline to other data
        if self.dataprefix == 'soln':
            self._pre_proc_fields = self._pre_proc_fields_soln
            self._post_proc_fields = self._post_proc_fields_soln
            self._soln_fields = self.elementscls.privars(self.ndims, self.cfg)
            self._vtk_vars = self.elementscls.visvars(self.ndims, self.cfg)
            self.tcurr = self.stats.getfloat('solver-time-integrator', 'tcurr')

            # See if our solution contains gradient data
            if len(dfields) == (1 + self.ndims)*len(self._soln_fields):
                self._gradients = True

                # Update list of solution fields
                self._soln_fields.extend(f'{f}-{d}'
                                         for f in list(self._soln_fields)
                                         for d in range(self.ndims))

                # Update the mapping of VTK variables to solution fields
                for var, vfields in list(self._vtk_vars.items()):
                    self._vtk_vars[f'grad {var}'] = nfields = []
                    for f in vfields:
                        nfields.extend(f'{f}-{d}' for d in range(self.ndims))
            else:
                self._gradients = False
        # Otherwise we're dealing with simple scalar data
        else:
            self._pre_proc_fields = self._pre_proc_fields_scal
            self._post_proc_fields = self._post_proc_fields_scal
            self._soln_fields = dfields
            self._vtk_vars = {k: [k] for k in self._soln_fields}
            self.tcurr = None

        # Handle field subsetting
        if self.fields:
            self._vtk_vars = {f: v for f, v in self._vtk_vars.items()
                              if f in self.fields}

            if len(self._vtk_vars) != len(self.fields):
                raise RuntimeError('Invalid field specification')

    def process(self, solnf, outfname):
        # Load the solution
        self._load_soln(solnf)

        if Path(outfname).suffix == '.vtu':
            self._write_vtu(outfname)
        else:
            self._write_pvtu(outfname)

    def _write_vtu(self, fname):
        comm, rank, root = get_comm_rank_root()

        fh = mpi.File.Open(comm, fname, mpi.MODE_CREATE | mpi.MODE_WRONLY)
        write_s = lambda s: fh.Write(s.encode())

        # Gather the element information to the root rank
        geinfo = comm.gather(self.einfo, root=root)

        # If we have any header information then write it
        if rank == root:
            write_s('<?xml version="1.0" ?>\n<VTKFile '
                    'byte_order="LittleEndian" type="UnstructuredGrid" '
                    f'version="{self.vtkfile_version}" '
                    'header_type="UInt64">\n<UnstructuredGrid>\n')

            if self.tcurr is not None:
                self._write_time_value(write_s)

            # Running byte-offset for appended data
            soffs, off = [], 0

            # Write out the array headers
            for einfo in geinfo:
                # Save the starting byte offset for this rank
                soffs.append(off)

                for etype, neles in einfo:
                    off = self._write_serial_header(write_s, etype, neles, off)

            write_s('</UnstructuredGrid>\n<AppendedData encoding="raw">\n_')

            # Get the size of the header
            hsize = fh.Get_position()

            # Use this to displace the offsets
            soffs = [s + hsize for s in soffs]

            # Compute the total size of the file sans footer
            size = hsize + off
        else:
            size, soffs = None, None

        # Distribute the total size and starting offset information
        size = comm.bcast(size, root=root)
        soff = comm.scatter(soffs, root=root)

        # Allocate space in the file
        fh.Set_size(size)

        # Have the root rank also write out the footer
        if rank == root:
            fh.Seek(0, mpi.SEEK_END)
            write_s('\n</AppendedData>\n</VTKFile>')

        # Seek to our region of the file
        fh.Seek(soff, mpi.SEEK_SET)

        # Write out our ranks data
        for etype, *_ in self.einfo:
            self._write_data(lambda b: fh.Write(b), etype)

        # Wait for all ranks to finish writing
        fh.Close()

    def _write_pvtu(self, fname):
        comm, rank, root = get_comm_rank_root()
        write_s = lambda s: fh.write(s.encode())

        # Have each rank write out its own VTU file
        with open(f'{fname[:-5]}_p{rank}.vtu', 'wb') as fh:
            write_s('<?xml version="1.0" ?>\n<VTKFile '
                    'byte_order="LittleEndian" type="UnstructuredGrid" '
                    f'version="{self.vtkfile_version}" '
                    'header_type="UInt64">\n<UnstructuredGrid>\n')

            # Running byte-offset for appended data
            off = 0

            # Write out the array headers
            for etype, neles in self.einfo:
                off = self._write_serial_header(write_s, etype, neles, off)

            write_s('</UnstructuredGrid>\n<AppendedData encoding="raw">\n_')

            # Followed by the data
            for etype, *_ in self.einfo:
                self._write_data(lambda b: fh.write(b), etype)

            write_s('\n</AppendedData>\n</VTKFile>')

        # Also have the root rank write out the PVTU file itself
        if rank == root:
            with open(fname, 'wb') as fh:
                write_s('<?xml version="1.0" ?>\n<VTKFile '
                        'byte_order="LittleEndian" type="PUnstructuredGrid" '
                        f'version="{self.vtkfile_version}">\n'
                        '<PUnstructuredGrid>\n')

                if self.tcurr is not None:
                    self._write_time_value(write_s)

                # Header
                self._write_parallel_header(write_s)

                # Constitutent pieces
                for r in range(comm.size):
                    bname = Path(f'{fname[:-5]}_p{r}.vtu').name
                    write_s(f'<Piece Source="{bname}"/>\n')

                write_s('</PUnstructuredGrid>\n</VTKFile>\n')

    def _write_darray(self, array, write, dtype):
        array = np.ascontiguousarray(array, dtype=dtype)

        write(np.uint64(array.nbytes))
        write(array)

    def _component_names(self, ncomps):
        cnames = {
            '2': ['X', 'Y'],
            '3': ['X', 'Y', 'Z'],
            '4': ['XX', 'XY', 'YX', 'YY'],
            '9': ['XX', 'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ']
        }

        if ncomps in cnames:
            return ' '.join(f'ComponentName{i}="{n}"'
                            for i, n in enumerate(cnames[ncomps]))
        else:
            return ''

    def _write_serial_header(self, write_s, etype, neles, off):
        ncelld = self.output_curved + self.output_partition
        npts, ncells = self._get_npts_ncells_nnodes(etype, neles)[:2]

        write_s(f'<Piece NumberOfPoints="{npts}" '
                f'NumberOfCells="{ncells}">\n<Points>\n')

        # Write VTK DataArray headers
        for i, (n, t, c, s) in enumerate(self._get_array_attrs(etype, neles)):
            write_s(f'<DataArray Name="{n}" type="{t}" '
                    f'NumberOfComponents="{c}" {self._component_names(c)} '
                    f'format="appended" offset="{off}"/>\n')

            off += 8 + s

            # Points => Cells => CellData => PointData transition
            if i == 0:
                write_s('</Points>\n<Cells>\n')
            if i == 3:
                write_s('</Cells>\n<CellData>\n')
            if i == 3 + ncelld:
                write_s('</CellData>\n<PointData>\n')

        # Close
        write_s('</PointData>\n</Piece>\n')

        # Return the current offset
        return off

    def _write_parallel_header(self, write_s):
        ncelld = self.output_curved + self.output_partition
        write_s('<PPoints>\n')

        # Write VTK DataArray headers
        for i, (n, t, c) in enumerate(self._get_array_attrs()):
            write_s(f'<PDataArray Name="{n}" type="{t}" '
                    f'NumberOfComponents="{c}" {self._component_names(c)}/>\n')

            # Points => Cells => CellData => PointData transition
            if i == 0:
                write_s('</PPoints>\n<PCells>\n')
            if i == 3:
                write_s('</PCells>\n<PCellData>\n')
            if i == 3 + ncelld:
                write_s('</PCellData>\n<PPointData>\n')

        # Close
        write_s('</PPointData>\n')

    def _write_time_value(self, write_s):
        write_s('<FieldData>\n'
                '<DataArray Name="TimeValue" type="Float64" '
                'NumberOfComponents="1" NumberOfTuples="1" format="ascii">\n'
                f'{self.tcurr}\n'
                '</DataArray>\n</FieldData>\n')

    def _write_data(self, write, etype):
        vpts, vsoln, curved, part = self._prepare_pts(etype)
        nsvpts, neles = vsoln.shape[0], vsoln.shape[2]

        # Write element node locations to file
        self._write_darray(vpts.swapaxes(0, 1), write, self.dtype)

        # Perform the sub division
        if etype != 'pyr' and self.ho_output:
            nodes = np.arange(nsvpts)
            subcellsoff = nsvpts
            types = self._vtk_types_ho[etype]
        else:
            subdiv = get_subdiv(etype, self.etypes_div[etype])

            nodes = subdiv.subnodes
            subcellsoff = subdiv.subcelloffs
            types = subdiv.subcelltypes

        # Prepare VTU cell arrays
        vtu_con = np.tile(nodes, (neles, 1))
        vtu_con += (np.arange(neles)*nsvpts)[:, None]

        # Generate offset into the connectivity array
        vtu_off = np.tile(subcellsoff, (neles, 1))
        vtu_off += (np.arange(neles)*len(nodes))[:, None]

        # Tile VTU cell type numbers
        vtu_typ = np.tile(types, neles)

        # Write VTU node connectivity, connectivity offsets and cell types
        self._write_darray(vtu_con, write, np.int64)
        self._write_darray(vtu_off, write, np.int64)
        self._write_darray(vtu_typ, write, np.uint8)

        # VTU cell curvature information
        if self.output_curved:
            vtu_curved = np.repeat(curved, len(vtu_typ) // neles)
            self._write_darray(vtu_curved, write, np.uint8)

        # VTU cell partition numbers
        if self.output_partition:
            vtu_part = np.repeat(part, len(vtu_typ) // neles)
            self._write_darray(vtu_part, write, np.int32)


        # Process and write out the various fields
        for arr in self._post_proc_fields(vsoln.swapaxes(0, 1)):
            self._write_darray(arr.T, write, self.dtype)


def get_subdiv(name, n):
    return subclass_where(BaseShapeSubDiv, name=name)(n)


class BaseShapeSubDiv:
    vtk_types = dict(tri=5, quad=9, tet=10, pyr=14, pri=13, hex=12)
    vtk_nodes = dict(tri=3, quad=4, tet=4, pyr=5, pri=6, hex=8)

    def __init__(self, n):
        self.n = n

    @property
    def subcelloffs(self):
        return np.cumsum([self.vtk_nodes[t] for t in self.subcells])

    @property
    def subcelltypes(self):
        return np.array([self.vtk_types[t] for t in self.subcells])


class TensorProdShapeSubDiv(BaseShapeSubDiv):
    @property
    def subcells(self):
        return [self.name]*(self.n**self.ndim)

    @property
    def subnodes(self):
        conbase = np.array([0, 1, self.n + 2, self.n + 1])

        # Extend quad mapping to hex mapping
        if self.ndim == 3:
            conbase = np.hstack((conbase, conbase + (1 + self.n)**2))

        # Calculate offset of each subdivided element's nodes
        nodeoff = np.zeros((self.n,)*self.ndim, dtype=np.int32)
        for dim, off in enumerate(np.ix_(*(range(self.n),)*self.ndim)):
            nodeoff += off*(self.n + 1)**dim

        # Tile standard element node ordering mapping, then apply offsets
        internal_con = np.tile(conbase, (self.n**self.ndim, 1))
        internal_con += nodeoff.T.ravel()[:, None]

        return np.hstack(internal_con)


class QuadShapeSubDiv(TensorProdShapeSubDiv):
    name = 'quad'
    ndim = 2


class HexShapeSubDiv(TensorProdShapeSubDiv):
    name = 'hex'
    ndim = 3


class TriShapeSubDiv(BaseShapeSubDiv):
    name = 'tri'

    @property
    def subcells(self):
        return ['tri']*(self.n**2)

    @property
    def subnodes(self):
        conlst = []

        for row in range(self.n, 0, -1):
            # Lower and upper indices
            l = (self.n - row)*(self.n + row + 3) // 2
            u = l + row + 1

            # Base offsets
            off = [l, l + 1, u, u + 1, l + 1, u]

            # Generate current row
            subin = np.ravel(np.arange(row - 1)[..., None] + off)
            subex = [ix + row - 1 for ix in off[:3]]

            # Extend list
            conlst.extend([subin, subex])

        return np.hstack(conlst)


class TetShapeSubDiv(BaseShapeSubDiv):
    name = 'tet'

    @property
    def subcells(self):
        return ['tet']*(self.n**3)

    @property
    def subnodes(self):
        conlst = []
        jump = 0

        for n in range(self.n, 0, -1):
            for row in range(n, 0, -1):
                # Lower and upper indices
                l = (n - row)*(n + row + 3) // 2 + jump
                u = l + row + 1

                # Lower and upper for one row up
                ln = (n + 1)*(n + 2) // 2 + l - n + row
                un = ln + row

                rowm1 = np.arange(row - 1)[..., None]

                # Base offsets
                offs = [(l, l + 1, u, ln), (l + 1, u, ln, ln + 1),
                        (u, u + 1, ln + 1, un), (u, ln, ln + 1, un),
                        (l + 1, u, u+1, ln + 1), (u + 1, ln + 1, un, un + 1)]

                # Current row
                conlst.extend(rowm1 + off for off in offs[:-1])
                conlst.append(rowm1[:-1] + offs[-1])
                conlst.append([ix + row - 1 for ix in offs[0]])

            jump += (n + 1)*(n + 2) // 2

        return np.hstack([np.ravel(c) for c in conlst])


class PriShapeSubDiv(BaseShapeSubDiv):
    name = 'pri'

    @property
    def subcells(self):
        return ['pri']*(self.n**3)

    @property
    def subnodes(self):
        # Triangle connectivity
        tcon = TriShapeSubDiv(self.n).subnodes.reshape(-1, 3)

        # Layer these rows of triangles to define prisms
        loff = (self.n + 1)*(self.n + 2) // 2
        lcon = [[tcon + i*loff, tcon + (i + 1)*loff] for i in range(self.n)]

        return np.hstack([np.hstack(l).flat for l in lcon])


class PyrShapeSubDiv(BaseShapeSubDiv):
    name = 'pyr'

    @property
    def subcells(self):
        cells = []

        for i in range(self.n, 0, -1):
            cells += ['pyr']*(i**2 + (i - 1)**2)
            cells += ['tet']*(2*i*(i - 1))

        return cells

    @property
    def subnodes(self):
        lcon = []

        # Quad connectivity
        qcon = [QuadShapeSubDiv(n + 1).subnodes.reshape(-1, 4)
                for n in range(self.n)]

        # Simple functions
        def _row_in_quad(n, a=0, b=0):
            return np.array([(n*i + j, n*i + j + 1)
                             for i in range(a, n + b)
                             for j in range(n - 1)])

        def _col_in_quad(n, a=0, b=0):
            return np.array([(n*i + j, n*(i + 1) + j)
                             for i in range(n - 1)
                             for j in range(a, n + b)])

        u = 0
        for n in range(self.n, 0, -1):
            l = u
            u += (n + 1)**2

            lower_quad = qcon[n - 1] + l
            upper_pts = np.arange(n**2) + u

            # First set of pyramids
            lcon.append([lower_quad, upper_pts])

            if n > 1:
                upper_quad = qcon[n - 2] + u
                lower_pts = np.hstack([range(k*(n + 1)+1, (k + 1)*n + k)
                                       for k in range(1, n)]) + l

                # Second set of pyramids
                lcon.append([upper_quad[:, ::-1], lower_pts])

                lower_row = _row_in_quad(n + 1, 1, -1) + l
                lower_col = _col_in_quad(n + 1, 1, -1) + l

                upper_row = _row_in_quad(n) + u
                upper_col = _col_in_quad(n) + u

                # Tetrahedra
                lcon.append([lower_col, upper_row])
                lcon.append([lower_row[:, ::-1], upper_col])

        return np.hstack([np.column_stack(l).flat for l in lcon])
