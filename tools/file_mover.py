import os
import shutil

if __name__ == "__main__":
    src_path = "/Users/yunseok/Downloads/안면인식 영상/Validation/2407-2847"
    dst_path = "/Users/yunseok/Downloads/안면인식 영상/OOD_test"

    for dir in os.listdir(src_path):
        file_dir = os.path.join(src_path, dir, 'GOPRO/Light_02_Mid/attack_01_print_none_flat/color/crop')

        for file in os.listdir(file_dir):
            # print(os.path.join(file_dir, file))
            # print(dir)
            shutil.move(os.path.join(file_dir, file), os.path.join(dst_path,dir+'_'+file))