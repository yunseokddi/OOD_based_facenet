import os
import shutil

if __name__ == "__main__":
    src_path = "/Users/yunseok/Downloads/안면인식 영상/OOD_test"

    for file_name in os.listdir(src_path):
        new_dir_name = file_name.split('_')[0]
        if not os.path.exists(os.path.join(src_path, new_dir_name)):
            os.makedirs(os.path.join(src_path, new_dir_name))

        shutil.move(os.path.join(src_path, file_name), os.path.join(src_path, new_dir_name, file_name))
