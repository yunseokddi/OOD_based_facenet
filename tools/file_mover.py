import os
import shutil

if __name__ == "__main__":
    src_path = "/home/dorosee/yunseok/data/lfw-deepfunneled"
    dst_path = "/home/dorosee/yunseok/data/lfw-samples"

    for directory in os.listdir(src_path):
        for file in os.listdir(os.path.join(src_path, directory)):
            shutil.copy(os.path.join(src_path, directory, file), os.path.join(dst_path, file))
