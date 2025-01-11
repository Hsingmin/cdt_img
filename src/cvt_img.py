import sys
import os
from PIL import Image

def convert_jpeg_to_png_in_directory(directory):
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        # 检查文件扩展名是否为.jpeg或.jpg（不区分大小写）
        if filename.lower().endswith(('.jpeg', '.jpg')):
            # 构建完整的文件路径
            input_path = os.path.join(directory, filename)
            # 构建输出文件的路径，将扩展名改为.png
            output_path = os.path.splitext(input_path)[0] + '.png'
            
            try:
                # 打开JPEG图片
                with Image.open(input_path) as img:
                    # 将图片保存为PNG格式
                    img.save(output_path, "PNG")
                    print(f"Successfully converted {input_path} to {output_path}")
            except Exception as e:
                print(f"Failed to convert {input_path}. Error: {e}")

# 示例用法
# directory_path = "path/to/your/jpeg/directory"  # 替换为你的JPEG图片所在的目录路径

# convert_jpeg_to_png_in_directory(directory_path)


if __name__ == "__main__":
    dir = sys.argv[1]
    # test input time
    convert_jpeg_to_png_in_directory(dir)
