from calendar import TUESDAY
import subprocess
import json
import sys
from tracemalloc import start
from unittest import TestSuite, result
import uuid
import os
import time


# curl -X POST http://127.0.0.1:5000/get_img_score -H "Content-Type: application/json" -d '{"request_id": "0000", "hour": 2, "minute": 10, "image": "/home/xingmin/applications/cdt_img/new_test/8.png"}'
# curl http://127.0.0.1:5000/get_score_result/0000


def request(request_data):
    data_json = json.dumps(request_data)

    # 定义curl命令及其参数
    curl_command = [
    'curl',
    '-X', 'POST',
    '-H', 'Content-Type: application/json',
    '-d', data_json,
    'http://127.0.0.1:5000/get_img_score'
    ]
    # 执行curl命令并捕获输出和错误
    result = subprocess.run(curl_command, capture_output=True, text=True)

    # 检查命令是否成功执行
    if result.returncode == 0:
        print("POST executed successfully!")
        print("Standard Output:")
        print(result.stdout)
    else:
        print(f"POST failed with return code {result.returncode}")
        print("Standard Error:")
        print(result.stderr)


def query_result(request_id):
    # 定义curl命令及其参数
    curl_command = [
    'curl',
    '-X', 'GET',
    'http://127.0.0.1:5000/get_score_result/' + request_id
    ]
    # 执行curl命令并捕获输出和错误
    result = subprocess.run(curl_command, capture_output=True, text=True)
    # 检查命令是否成功执行
    if result.returncode == 0:
        print("GET executed successfully!")
        print("Standard Output:")
        print(result.stdout)
    else:
        print(f"GET failed with return code {result.returncode}")
        print("Standard Error:")
        print(result.stderr)

    try:
        json_data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        # 如果输出不是有效的JSON格式，捕获异常并处理
        print(f"Error decoding JSON: {e}")
        # 这里可以返回一个错误消息或者抛出一个异常
        return None  # 或者 raise Exception("Output is not valid JSON")
 
    # 返回转换后的JSON数据
    return json_data


def polling_untill_done(request_id, data):
    timeout = 300
    start_time = time.time()
    while True:
        jsonresult = query_result(request_id)
        if "error" not in jsonresult:
            with open('./tmp/data.json', 'r', encoding='utf-8') as f:
                history_data = json.load(f)
            # data_list = [dict()]
            # data_list.append(history_data)
            # 修改数据（向列表中添加新元素）
            data["result"] = jsonresult["result"]
            print("append data {} to history_data = {}".format(data, history_data))
            history_data.append(data)

            with open('./tmp/data.json', 'w', encoding='utf-8') as json_file:
                json.dump(history_data, json_file, ensure_ascii=False, indent=4)
                print("data.json文件已被创建（或覆盖）并写入数据。")
            return True
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            return False
        time.sleep(50)


# if __name__ == "__main__":
#     img_dir = sys.argv[1]
#     hour = int(sys.argv[2])
#     minute = int(sys.argv[3])
#     request_id = str(uuid.uuid4())
#     data = {
#         "request_id": request_id,
#         "image": img_dir,
#         "hour": hour,
#         "minute": minute
#     }

#     request(data)
#     polling_untill_done(request_id, data)


if __name__ == "__main__":
    dir = sys.argv[1]

    for parent, dirnames, filenames in os.walk(dir):
        for img_name in filenames:
            img_path = os.path.join(parent, img_name)
            print("Request computing score for {}".format(img_path))
            request_id = str(uuid.uuid4())
            data = {
                "request_id": request_id,
                "image": img_path,
                "hour": 5,
                "minute": 20
            }

            request(data)
            polling_untill_done(request_id, data)
