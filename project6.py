import cv2
import math
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from keras import backend as K
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.ndimage import measurements

np.random.seed(42)
font = cv2.FONT_HERSHEY_PLAIN


def load_mnist_get_data_info():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    n_classes = 10
    img_rows, img_cols = 28, 28
    y_train = np_utils.to_categorical(y_train, n_classes)
    y_test = np_utils.to_categorical(y_test, n_classes)

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    return x_train, y_train, x_test, y_test, input_shape, n_classes


def create_model(input_shape, n_classes):
    model = Sequential()
    model.add(Convolution2D(filters=32, kernel_size=2, padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(filters=64, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Convolution2D(filters=128, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))
    return model


def compile_fit(model, checkpointer, validation_split=0.20, batch_size=150, epochs=100):
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_split=validation_split, batch_size=batch_size, epochs=epochs, callbacks=[checkpointer],
              verbose=True, shuffle=True)
    return model, history


def compile_evaluate_trained_model(model, x_test, y_test):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    evaluate_model(model, x_test, y_test)


def compile_evaluate_trained_model_(model, x_test):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    y_pred = model.predict_classes(x_test)
    y_pred1 = model.predict(x_test)
    prob = np.max(y_pred1) * 100
    prob = float("%.2f" % prob)
    return y_pred, prob


def evaluate_model(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    print(score)
    y_pred = model.predict(x_test)
    matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    print(matrix)
    print("test score - {0}".format(score[0]))  # score 0 is test acc
    print("test accuracy - {0} %".format(score[1] * 100))  # score 0 is test loss


def load_trained_model(input_shape, n_classes, filepath='mnist.model.best.hd5'):
    model = create_model(input_shape, n_classes)
    model.load_weights(filepath)
    return model


def train_test_model_offline(input_shape, n_classes, checkpointer, x_test, y_test, validation_split=0.2, batch_size=150, epochs=100):
    model = create_model(input_shape, n_classes)
    model, history = compile_fit(model, checkpointer, validation_split=validation_split, batch_size=batch_size, epochs=epochs)
    plot_graphs(history)
    evaluate_model(model, x_test, y_test)


def plot_graphs(history):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def test_rects(frame, rect, model):
    x, y, w, h = rect
    img = frame.copy()
    img = img[y: y + h, x: x + w]
    kernel = (9, 9)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel))
    while np.sum(img[0]) == 0:
        img = img[1:]

    while np.sum(img[:, 0]) == 0:
        img = np.delete(img, 0, 1)

    while np.sum(img[-1]) == 0:
        img = img[:-1]

    while np.sum(img[:, -1]) == 0:
        img = np.delete(img, -1, 1)

    rows, cols = img.shape
    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
        img = cv2.resize(img, (cols, rows))
    else:
        factor = 28.0 / cols
        cols = 20
        rows = int(round(rows * factor))
        img = cv2.resize(img, (cols, rows))

    img = img/255.0
    colsPadding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
    rowsPadding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
    img = np.lib.pad(img, (rowsPadding, colsPadding), 'constant')
    shiftx, shifty = getBestShift(img)
    img = shift(img, shiftx, shifty)
    x_test = np.expand_dims(img, axis=0)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    pred, prob = compile_evaluate_trained_model_(model, x_test)
    return pred, prob


def getBestShift(img):
    cy, cx = measurements.center_of_mass(img)
    rows, cols = img.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)
    return shiftx, shifty


def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted


def show_img(img):
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def predict_live(input_shape, n_classes, x_test):
    trained_model = load_trained_model(input_shape, n_classes)
    compile_evaluate_trained_model(trained_model, x_test, y_test)


def capture_image():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, "Press Q to capture", (50, 50), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite('capture.jpg', frame)
            break
        else:
            continue
    cap.release()
    cv2.destroyAllWindows()


def process_img():
    kernel = (3, 3)
    frame = cv2.imread('capture.jpg')
    grayscaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 115, 1)
    img = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel))
    output = cv2.connectedComponentsWithStats(
        img, 8, cv2.CV_32S)
    for i in range(output[0]):
        val1 = output[2][i][4]
        val2 = output[2][i][4]
        if val1 >= 11000 and val2 <= 35000:
            x = output[2][i][0]
            y = output[2][i][1]
            w = output[2][i][2]
            h = output[2][i][3]
            rect = (x, y, w, h)
            aspect_ratio = w/h
            print(aspect_ratio)
            if aspect_ratio <= 0.9 and aspect_ratio >= 0.5:
                pred, prob = test_rects(frame, rect, trained_model)
                if prob >= 70:
                    prediction = str(pred)
                    cv2.putText(frame, prediction, (x + int(w / 3), y + int(h / 2)), font, 1, (0, 255, 0), 1,
                                cv2.LINE_AA)
                    prob = str(prob)
                    cv2.putText(frame, prob + "%", (x, y - 2), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
    show_img(frame)
    cv2.imwrite("part2.jpg", frame)


def process_img_():
    kernel = (3, 3)
    frame = cv2.imread('capture.jpg')
    grayscaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 115, 1)
    img = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel))
    output = cv2.connectedComponentsWithStats(
        img, 8, cv2.CV_32S)
    pred_probs = []
    for i in range(output[0]):
        val1 = output[2][i][4]
        val2 = output[2][i][4]
        if val1 >= 200 and val2 <= 2000:
            x = output[2][i][0]
            y = output[2][i][1]
            w = output[2][i][2]
            h = output[2][i][3]
            rect = (x, y, w, h)
            aspect_ratio = w/h
            if aspect_ratio <= 0.9 and aspect_ratio >= 0.5:
                pred, prob = test_rects(frame, rect, trained_model)
                if prob >= 70:
                    pred_probs.append(prob/10.0)
                    prediction = str(pred)
                    cv2.putText(frame, prediction, (x + int(w/3), y + int(h/2)), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
                    prob = str(prob)
                    cv2.putText(frame, prob, (x, y-2), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
    avg_pred_prob = float("%.1f" % np.sum(pred_probs))
    cv2.putText(frame, "% AVG Classification Accuracy - " + str(avg_pred_prob) + "%", (50, 60), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
    show_img(frame)
    cv2.imwrite("part3.jpg", frame)


# get mnist dataset from keras
checkpointer = ModelCheckpoint(filepath='mnist.model.best.hd5', verbose=1, save_best_only=True)
x_train, y_train, x_test, y_test, input_shape, n_classes = load_mnist_get_data_info()

# PART 1:
# epoch 10 is chosen by running epochs till 50, and from loss vs epoch and accuracy vs epoch graph,
# chose epoch 10 which has a good enough training accuracy and testing accuracy without overfitting.
train_test_model_offline(input_shape, n_classes, checkpointer, x_test, y_test, batch_size=300, epochs=10)

# PART 2:
# loads model saved in file mnist.model.best.hd5 to classify new captures.
trained_model = load_trained_model(input_shape, n_classes)
capture_image()
process_img()

# PART 3:
# load new img data embedded in another image, localize and identify the handwritten digits
trained_model = load_trained_model(input_shape, n_classes)
capture_image()
process_img_()
