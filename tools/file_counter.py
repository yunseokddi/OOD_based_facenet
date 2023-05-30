import os

if __name__ == "__main__":
    src_path = "/Users/yunseok/Downloads/VGGface_2"

    idx = 0

    for dir in os.listdir(src_path):
        for file in os.listdir(os.path.join(src_path, dir)):
            idx += 1

    print("Total num : {}".format(idx))
