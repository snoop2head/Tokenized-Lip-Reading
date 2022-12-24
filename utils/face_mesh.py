# https://github.com/google/mediapipe/issues/1615
# https://github.com/google/mediapipe/blob/33d683c67100ef3db37d9752fcf65d30bea440c4/mediapipe/python/solutions/face_mesh_connections.py
# https://github.com/google/mediapipe/blob/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
# https://github.com/google/mediapipe/issues/1615
# https://github.com/tensorflow/tfjs-models/blob/838611c02f51159afdd77469ce67f0e26b7bbb23/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts
# https://github.com/google/mediapipe/blob/33d683c67100ef3db37d9752fcf65d30bea440c4/mediapipe/python/solutions/face_mesh_connections.py

FACEMESH_LEFT_EYE = list(
    set(
        [
            263,
            249,
            249,
            390,
            390,
            373,
            373,
            374,
            374,
            380,
            380,
            381,
            381,
            382,
            382,
            362,
            263,
            466,
            466,
            388,
            388,
            387,
            387,
            386,
            386,
            385,
            385,
            384,
            384,
            398,
            398,
            362,
        ]
    )
)

FACEMESH_LEFT_IRIS = list(set([474, 475, 475, 476, 476, 477, 477, 474]))

FACEMESH_LEFT_EYEBROW = list(
    set(
        [276, 283, 283, 282, 282, 295, 295, 285, 300, 293, 293, 334, 334, 296, 296, 336]
    )
)

FACEMESH_RIGHT_EYE = list(
    set(
        [
            33,
            7,
            7,
            163,
            163,
            144,
            144,
            145,
            145,
            153,
            153,
            154,
            154,
            155,
            155,
            133,
            33,
            246,
            246,
            161,
            161,
            160,
            160,
            159,
            159,
            158,
            158,
            157,
            157,
            173,
            173,
            133,
        ]
    )
)

FACEMESH_RIGHT_EYEBROW = list(
    set([46, 53, 53, 52, 52, 65, 65, 55, 70, 63, 63, 105, 105, 66, 66, 107])
)

FACEMESH_RIGHT_IRIS = list(set([469, 470, 470, 471, 471, 472, 472, 469]))

FACEMESH_NOT_USE = (
    FACEMESH_LEFT_EYE
    + FACEMESH_LEFT_IRIS
    + FACEMESH_LEFT_EYEBROW
    + FACEMESH_RIGHT_EYE
    + FACEMESH_RIGHT_EYEBROW
    + FACEMESH_RIGHT_IRIS
)
print(len(FACEMESH_NOT_USE))
FACEMESH_NOT_USE.sort()
print(FACEMESH_NOT_USE)

FACEMESH_LIPS = list(
    set(
        [
            61,
            146,
            146,
            91,
            91,
            181,
            181,
            84,
            84,
            17,
            17,
            314,
            314,
            405,
            405,
            321,
            321,
            375,
            375,
            291,
            61,
            185,
            185,
            40,
            40,
            39,
            39,
            37,
            37,
            0,
            0,
            267,
            267,
            269,
            269,
            270,
            270,
            409,
            409,
            291,
            78,
            95,
            95,
            88,
            88,
            178,
            178,
            87,
            87,
            14,
            14,
            317,
            317,
            402,
            402,
            318,
            318,
            324,
            324,
            308,
            78,
            191,
            191,
            80,
            80,
            81,
            81,
            82,
            82,
            13,
            13,
            312,
            312,
            311,
            311,
            310,
            310,
            415,
            415,
            308,
        ]
    )
)

FACEMESH_FACE_OVAL = list(
    set(
        [
            10,
            338,
            338,
            297,
            297,
            332,
            332,
            284,
            284,
            251,
            251,
            389,
            389,
            356,
            356,
            454,
            454,
            323,
            323,
            361,
            361,
            288,
            288,
            397,
            397,
            365,
            365,
            379,
            379,
            378,
            378,
            400,
            400,
            377,
            377,
            152,
            152,
            148,
            148,
            176,
            176,
            149,
            149,
            150,
            150,
            136,
            136,
            172,
            172,
            58,
            58,
            132,
            132,
            93,
            93,
            234,
            234,
            127,
            127,
            162,
            162,
            21,
            21,
            54,
            54,
            103,
            103,
            67,
            67,
            109,
            109,
            10,
        ]
    )
)


FACE_MESH_USE = list(set(range(478)) - set(FACEMESH_NOT_USE)) + [
    472,
    477,
]  # left and right iris

print(len(FACE_MESH_USE))
