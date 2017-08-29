from darkflow.net.build import TFNet
import cv2
from time import time as timer
import sys
import argparse
import numpy as np
import os

def main():
    # read video from file
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", help="path to input file", type=str)
    parser.add_argument("-o", "--output_path", help="path to output file", type=str)
    parser.add_argument("-t", "--threshold", help="Confidence level (Thershold)", type=float)
    parser.add_argument("-v", "--verbose", help="Show video in progress", action="store_true")
    parser.add_argument("-c", "--camera", help="read input from WebCam", action="store_true")
    args = parser.parse_args()

    input_file = args.input_path
    output_file = args.output_path
    threshold = args.threshold

    if not args.threshold:
        threshold = 0.25

    # options of darkflow - model, weights
    options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": threshold}
    # create a tfnet object
    tfnet = TFNet(options)
    global tfnet

    if args.camera:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(input_file)

    # save parameters of video
    width = int(cap.get(3))

    height = int(cap.get(4))

    frame_rate = int(cap.get(5))

    # initializing videocodec and video writer
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(output_file, fourcc, frame_rate, (width, height))

    elapsed = int()
    start = timer()
    # reading video file an image by image while there is images
    # to see images in realtimework you can uncomment string cv2.imshow('',preprocess)
    i = 0
    while cap.isOpened():
        i += 1
        if i % 10:
            continue

        ret, frame = cap.read()
        if not ret:
            break
        preprocessed = tfnet.framework.preprocess(frame)
        feed_dict = {tfnet.inp: [preprocessed]}
        net_out = tfnet.sess.run(tfnet.out, feed_dict)[0]
        processed = postprocess(net_out, frame, False)
        out.write(processed)
        if args.verbose:
            cv2.imshow('', processed)
        elapsed += 1
        if elapsed % 5 == 0:
            sys.stdout.write('\r')
            sys.stdout.write('{0:3.3f} FPS'.format(
                elapsed / (timer() - start)))
            sys.stdout.flush()
        choice = cv2.waitKey(1)
        if choice == 27:
            break
    # writing video to file
    sys.stdout.write('\n')
    cap.release()
    out.release()
    cv2.destroyAllWindows()


# calculation of rectangle area

def area(a, b, c, d):

    return (a-b)*(c-d)

# posprocess that detect only persons


def postprocess(net_out, im, save=True):

    meta, FLAGS = tfnet.framework.meta, tfnet.framework.FLAGS
    threshold = FLAGS.threshold
    boxes = tfnet.framework.findboxes(net_out)
    if type(im) is not np.ndarray:
        imgcv = cv2.imread(im)
    else:
        imgcv = im
    h, w, _ = imgcv.shape
    for b in boxes:
        boxResults = tfnet.framework.process_box(b, h, w, threshold)
        if boxResults is None:
            continue
        left, right, top, bot, label, max_indx, confidence = boxResults
        thick = int((h + w) // 300)

        if label == "person" and area(right, left, bot, top) < 40000:
            cv2.rectangle(imgcv,
                          (left, top), (right, bot), meta['colors'][max_indx], thick)
            cv2.putText(imgcv, label, (left, top - 12),
                        0, 1e-3 * h, meta['colors'][max_indx],
                        thick // 3)
    if not save:
        return imgcv

    outfolder = os.path.join(FLAGS.imgdir, 'out')
    img_name = os.path.join(outfolder, os.path.basename(im))

    cv2.imwrite(img_name, imgcv)


if __name__ == "__main__":
    main()
