import os
import sys
import os.path as osp
import cv2


def create_image_folder(path):
    os.mkdir(f"{path}-frames")


def check_frame_folder_exists(path):
    return osp.exists(f"{path}-frames")


def process_video(path):
    if not check_frame_folder_exists(path):
        vidcap = cv2.VideoCapture(path)
        create_image_folder(path)
        count = 0
        while vidcap.isOpened():
            ret, frame = vidcap.read()

            if ret:
                cv2.imwrite(f'{path}-frames/frame-{count}.png', frame)
                count += 5
                vidcap.set(1, count)
            else:
                vidcap.release()
                break


if __name__=="__main__":
    for f in os.listdir(sys.argv[1]):
        if not osp.isdir(osp.join(sys.argv[1],f)):
            process_video(osp.join(sys.argv[1], f))