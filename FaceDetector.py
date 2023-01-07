
import pickle
import random
import shutil
import time
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm, __all__, metrics
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay


def copy_files(src, dst, count):
    data_size = len([name for name in os.listdir(src) if os.path.isfile(os.path.join(src, name))])
    randomlist = random.sample(range(0, data_size), count)
    i = 0
    for (root, dirs, file) in os.walk(dst):
        for f in file:
            os.remove(dst + "/" + f)

    for (root, dirs, file) in os.walk(src):
        for f in file:
            if i in randomlist:
                shutil.copy(src + "/" + f, dst)
                os.remove(src + "/" + f)
            i += 1
    print("Randomize of", dst, "Completed")


def randomize_data(train, validation, test):
    DIR_POSITIVE = "Data/Faces"
    DIR_NEGATIVE = "Data/Not-Faces"
    DST_TRAIN_POSITIVE = "DATA/Train/positive"
    DST_TRAIN_NEGATIVE = "DATA/Train/negative"
    DST_VALIDATION_POSITIVE = "DATA/Validation/positive"
    DST_VALIDATION_NEGATIVE = "DATA/Validation/negative"
    DST_TEST_POSITIVE = "DATA/Test/positive"
    DST_TEST_NEGATIVE = "DATA/Test/negative"

    copy_files(DIR_POSITIVE, DST_TRAIN_POSITIVE, train)
    copy_files(DIR_POSITIVE, DST_VALIDATION_POSITIVE, validation)
    copy_files(DIR_POSITIVE, DST_TEST_POSITIVE, test)
    copy_files(DIR_NEGATIVE, DST_TRAIN_NEGATIVE, train)
    copy_files(DIR_NEGATIVE, DST_VALIDATION_NEGATIVE, validation)
    copy_files(DIR_NEGATIVE, DST_TEST_NEGATIVE, test)


def hogExtraction(num, directory):
    c = 0
    Train = []
    classes = []
    class_names = []
    for it in os.scandir(directory):
        if it.is_dir():
            Class = it.path
            if directory in Class:
                Class = Class.replace(directory, '')
            fol_path = it.path
            class_names.append(Class)
            counter = 0
            for (root, dirs, file) in os.walk(fol_path):
                for f in file:

                    im = cv2.imread(fol_path + "/" + f, 1)
                    if c == 1:
                        im = im[50:-50, 50:-50]
                    descriptor = hog.compute(im)
                    if counter == 0:
                        lenght = len(descriptor)
                    try:
                        Train.append(descriptor.reshape(lenght))
                    except:
                        continue
                    classes.append(c)
                    counter += 1
            print('Calculated', c + 1, 'Class')
            c += 1
    return np.float32(Train), np.array(classes), class_names


def Training(directory):
    print('Calculating features of Dataset for Training ...')
    Train, classes, class_names = hogExtraction(1, directory)
    clf = svm.SVC()
    print('Training Model Started ...')
    clf.fit(Train, classes)
    pickle.dump(clf, open(MODEL_PATH, 'wb'))
    return clf


def FaceDetector(svm, test_directory):
    print('Calculating features of Dataset for Testing ...')
    Test, classes_test, class_names_test = hogExtraction(2, test_directory)
    print("predict TEST data ...")
    testResponse = svm.predict(Test)
    print(len(classes_test))
    mask = testResponse == classes_test
    correct = np.count_nonzero(mask)
    acc = np.round(correct * 2000.0 / classes_test.size)
    print('--', correct)
    print('ACCURACY:', acc / 20.0, '%')
    print("Drawing curves ...")
    y_score = svm.decision_function(Test)
    draw_ROC(classes_test, y_score)
    draw_RP(classes_test, y_score)


def draw_ROC(Test_classes, scores):
    RocCurveDisplay.from_predictions(Test_classes, scores, name="SVM")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC")
    plt.legend(loc="lower right")
    plt.savefig('res1.jpg')


def draw_RP(Test_classes, scores):
    PrecisionRecallDisplay.from_predictions(Test_classes, scores, name="SVM")
    plt.xlim([0.0, 1.05])
    plt.ylim([0.45, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision - Recall")
    plt.legend(loc="lower left")
    plt.savefig('res2.jpg')


def resize(img):
    return cv2.resize(img, (150, 150), interpolation=cv2.INTER_AREA)


def find_face(svm, image, start, end):
    im = image.copy()
    cv2.imwrite('h.jpg', im)
    Xcords = []
    Ycords = []
    sizes = []
    confidence = []
    threshhold = 1.1

    # box_size = 100
    for box_size in range(start, end, 15):
        print("Start search in box size of", box_size, "...")
        steps = int(box_size / 12)
        h_it = int((image.shape[0] - box_size) / steps)
        w_it = int((image.shape[1] - box_size) / steps)
        for h in range(h_it):
            for w in range(w_it):
                descriptor = hog.compute(resize(image[h * steps:h * steps + box_size, w * steps:w * steps + box_size]))
                if h == 0 and w == 0:
                    lenght = len(descriptor)
                response = svm.decision_function([descriptor.reshape(lenght)])[0]
                if response > threshhold:
                    Xcords.append(w * steps + box_size / 2)
                    Ycords.append(h * steps + box_size / 2)
                    sizes.append(int(box_size / 2))
                    confidence.append(response)
            print(h, '/', h_it)
    return Xcords, Ycords, sizes, confidence


def arrange_founded_coords(Xcords, Ycords, sizes, confidences):
    linked_points_x = [[Xcords[0]]]
    linked_points_y = [[Ycords[0]]]
    linked_sizes = [[sizes[0]]]
    linked_confidence = [[confidences[0]]]
    for i in range(len(Xcords)):
        flag = True
        for x in range(len(linked_points_x)):
            link_x = np.mean(np.array(linked_points_x[x]))
            link_y = np.mean(np.array(linked_points_y[x]))
            link_size = np.mean(np.array(linked_sizes[x])) * 1.45

            if link_x - link_size < Xcords[i] < link_x + link_size and \
                    link_y - link_size < Ycords[i] < link_y + link_size:
                flag = False
                linked_points_x[x].append(Xcords[i])
                linked_points_y[x].append(Ycords[i])
                linked_sizes[x].append(sizes[i])
                linked_confidence[x].append(confidences[i])
                break
        if flag:
            linked_points_x.append([Xcords[i]])
            linked_points_y.append([Ycords[i]])
            linked_sizes.append([sizes[i]])
            linked_confidence.append([confidences[i]])
    return linked_points_x, linked_points_y, linked_sizes, linked_confidence


def make_new_coords(linked_points_x, linked_points_y, linked_sizes, linked_confidence):
    new_cords_x = [int(np.mean(linked_points_x[0]))]
    new_cords_y = [int(np.mean(linked_points_y[0]))]
    new_sizes = [int(np.mean(linked_sizes[0]))]
    new_confidences = [int(np.max(linked_confidence[0]))]
    for x in range(1, len(linked_points_x)):
        new_cords_x.append(int(np.mean(np.array(linked_points_x[x]))))
        new_cords_y.append(int(np.mean(np.array(linked_points_y[x]))))
        new_sizes.append(int(np.mean(np.array(linked_sizes[x]))))
        new_confidences.append(np.max(np.array(linked_confidence[x])))
    return new_cords_x, new_cords_y, new_sizes, new_confidences


def draw_rectangle(new_cords_x, new_cords_y, sizes, new_confidences, image):
    top = np.max(new_confidences) * 1.05
    for rec_y, rec_x, size, c in zip(new_cords_y, new_cords_x, sizes, new_confidences):
        cv2.rectangle(image, (rec_x - size, rec_y - size), (rec_x + size, rec_y + size), (0, 0, 255), 3)
        cv2.putText(image, str(int(np.sqrt(c / top) * 10000) / 100.0) + " %", (rec_x - size, rec_y - size - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return image


def face_detector(im_path):
    image = cv2.imread(im_path, 1)
    im = np.zeros((image.shape[0] + 100, image.shape[1] + 100, 3), dtype='uint8')
    im[50:-50, 50:-50] = image
    image = im.copy()
    Xcords, Ycords, sizes, confidences = find_face(svm, image, start=100, end=210)
    linked_points_x, linked_points_y, linked_sizes, linked_confidences = arrange_founded_coords(Xcords, Ycords, sizes,
                                                                                                confidences)
    new_cords_x, new_cords_y, new_sizes, new_confidences = make_new_coords(linked_points_x, linked_points_y,
                                                                           linked_sizes, linked_confidences)
    detected = draw_rectangle(new_cords_x, new_cords_y, new_sizes, new_confidences, image)
    return detected[50:-50, 50:-50]


if __name__ == '__main__':
    DST_TRAIN = "DATA/Train"
    DST_VALIDATION = "DATA/Validation"
    DST_TEST = "DATA/TEST"
    MODEL_PATH = 'face_detection_model.dat'

    winSize = (20, 20)
    blockSize = (10, 10)
    blockStride = (5, 5)
    cellSize = (10, 10)
    nbins = 12

    t0 = time.time()

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

    randomize_data(train=10000, validation=1000, test=1000)

    svm = Training(DST_TRAIN)
    #svm = pickle.load(open(MODEL_PATH, 'rb'))

    FaceDetector(svm, DST_TEST)

    final_detected = face_detector("Esteghlal.jpg")
    cv2.imwrite('res6.jpg', final_detected)

    t1 = time.time()
    print('runtime: ' + str(int(t1 - t0)) + ' seconds')
