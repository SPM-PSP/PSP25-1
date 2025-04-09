import os
import cv2
import uuid

def resize_and_move_images(source_dir, destination_dir):
    """
    读取源目录下的所有文件夹，依次读取里面的图片，
    用opencv调整大小为512*512，然后转移到目标文件夹，
    用uid重新命名并生成uid.txt，内容是文件夹的名字。

    Args:
        source_dir (str): 源目录路径。
        destination_dir (str): 目标目录路径。
    """

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for folder_name in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder_name)
        if os.path.isdir(folder_path):
            print(f"正在处理文件夹: {folder_name}")
            process_folder(folder_path, destination_dir, folder_name)

def process_folder(folder_path, destination_dir, original_folder_name):
    """
    处理单个文件夹中的图片。

    Args:
        folder_path (str): 当前文件夹的路径。
        destination_dir (str): 目标目录路径。
        original_folder_name (str): 原始文件夹的名字，用于写入uid.txt。
    """
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                img = cv2.imread(file_path)
                if img is None:
                    print(f"无法读取图片: {filename}, 已跳过")
                    continue

                resized_img = cv2.resize(img, (512, 512))
                uid = uuid.uuid4()
                new_filename = str(uid) + ".jpg" # 保留原始图片后缀
                if not os.path.exists(destination_dir+"/"+original_folder_name):
                    os.makedirs(destination_dir+"/"+original_folder_name)
                new_file_path = os.path.join(destination_dir, original_folder_name,new_filename)
                cv2.imwrite(new_file_path, resized_img)
                print(f"图片: {filename} 已调整大小并保存为: {new_filename}")

            except Exception as e:
                print(f"处理图片 {filename} 时发生错误: {e}")



if __name__ == "__main__":
    source_directory = "D:\\dachuan\\dataset1\\img_new\\imgs"  # 替换为你的源目录
    destination_directory = "D:\\dachuan\\dataset1\\seperate\\diffusion_new_1" # 替换为你的目标目录

    resize_and_move_images(source_directory, destination_directory)
    print("所有文件夹处理完成！")