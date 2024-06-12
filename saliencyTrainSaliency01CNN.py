from definitions import *
from configTrainSaliency01CNN import *
from nets import CNNmodelKeras
import glob
import os
import numpy as np
from multiprocessing import Pool

# Initialize the model
saliency_model = CNNmodelKeras(img_size, num_channels, num_classes, type)
train_data = []
train_labels = []

# Load model names
trainSet = [os.path.splitext(os.path.basename(file))[0] for file in glob.glob(rootdir + modelsDir + "*.obj")]

# def process_model(modelName):
    # ======Model information=====================================================================
    # mModelSrc = rootdir + modelsDir + modelName + '.obj'
    # print(modelName)
    # if mode == "MESH":
    #     mModel = loadObj(mModelSrc)
    #     updateGeometryAttibutes(mModel, useGuided=useGuided, numOfFacesForGuided=patchSizeGuided, computeDeltas=False,
    #                             computeAdjacency=False, computeVertexNormals=False)
    # elif mode == "PC":
    #     mModel = loadObjPC(mModelSrc, nn=pointcloudnn, simplify=presimplification)
    #     V, inds = computePointCloudNormals(mModel, pointcloudnn)
    #     exportPLYPC(mModel, modelsDir + modelName + '_pcnorm_conf.ply')

    # gtdata = np.loadtxt(rootdir + modelsDir + modelName + '.txt', delimiter=',')

    # print('Saliency ground truth data')
    # saliencyValues = gtdata.tolist() if type == 'continuous' else [int((num_classes - 1) * s) for s in gtdata.tolist()]

    # patches = []
    # if mode == "MESH":
    #     patches = [neighboursByFace(mModel, i, numOfElements)[0] for i in range(0, len(mModel.faces))]
    # elif mode == "PC":
    #     patches = [neighboursByVertex(mModel, i, numOfElements)[0] for i in range(0, len(mModel.vertices))]

    # patches = np.loadtxt("./cached_face_neighbors/" + modelName + "_neighbors.csv", delimiter=',').astype(int)

    # # sort patches by first column and then delete first column
    # patches = patches[patches[:, 0].argsort()][:, 1:]

    # patchFacesNormals = np.loadtxt("./face_normals_and_area/" + modelName + "_face_normals.csv", delimiter=',')
    # # patchFacesArea = np.loadtxt("./face_normals_and_area/" + modelName + "_face_area.csv", delimiter=',')

    # data = []
    # for i, p in enumerate(patches):
    #     # if mode == "MESH":
    #     #     patchFacesOriginal = [mModel.faces[i] for i in p]
    #     #     normalsPatchFacesOriginal = np.asarray([pF.faceNormal for pF in patchFacesOriginal])
    #     # elif mode == "PC":
    #     #     patchVerticesOriginal = [mModel.vertices[i] for i in p]
    #     #     normalsPatchVerticesOriginal = np.asarray([pF.normal for pF in patchVerticesOriginal])
    #     normalsPatch = patchFacesNormals[p]
    #     # areasPatch = patchFacesArea[p]


    #     vec = np.mean(normalsPatch, axis=0)
    #     vec = vec / np.linalg.norm(vec)
    #     axis, theta = computeRotation(vec, target)
    #     normalsPatch = rotatePatch(normalsPatch, axis, theta)
    #     normalsPatchR = normalsPatch.reshape((patchSide, patchSide, 3))
    #     if reshapeFunction == "hilbert":
    #         for hci in range(np.shape(I2HC)[0]):
    #             normalsPatchR[I2HC[hci, 0], I2HC[hci, 1], :] = normalsPatch[:, HC2I[I2HC[hci, 0], I2HC[hci, 1]]]
    #     data.append((normalsPatchR + 1.0 * np.ones(np.shape(normalsPatchR))) / 2.0)

    # np.save(f"./train_data/{modelName}.npy", data)
    # return data, saliencyValues

# Process models in parallel
# with Pool() as pool:
#     results = pool.map(process_model, trainSet)

# Collect results
for modelName in trainSet:
    train_data.extend(np.load(f"./train/{modelName}.npy", delimiter=','))
    train_labels.extend(np.loadtxt(f"./dataset/{modelName}.txt", delimiter=','))

# Dataset and labels summarization ========================================================================
train_data = np.asarray(train_data)
if type == 'continuous':
    train_labels = np.asarray([np.asarray(train_labels)]).transpose()
    seppoint = int(0.9 * len(train_data))
    X, X_test = train_data[:seppoint], train_data[seppoint:]
    Y, Y_test = train_labels[:seppoint], train_labels[seppoint:]
    data_train, data_test = X, X_test
    label_train, label_test = Y, Y_test
    saliency_model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam', metrics=[tf.keras.metrics.RootMeanSquaredError()])
# elif type == 'discrete':
#     seppoint = int(0.9 * len(train_data))
#     X, X_test = train_data[:seppoint], train_data[seppoint:]
#     Y, Y_test = train_labels[:seppoint], train_labels[seppoint:]
#     data_train, data_test = X, X_test
#     label_train, label_test = to_categorical(Y, num_classes=num_classes), to_categorical(Y_test, num_classes=num_classes)
#     saliency_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

saliency_model.summary()
saliency_model_train = saliency_model.fit(x=data_train, y=label_train, batch_size=batch_size, epochs=numEpochs, verbose=1)
saliency_model.save(rootdir + sessionsDir + keyTrain + '.h5')
