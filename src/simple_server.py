from flask import Flask, request, jsonify
import threading
import time
import uuid
from image_score import score_circle_and_hands


app = Flask(__name__)

# 用于存储计算结果的字典
results = {}
# 用于存储计算任务的锁
results_lock = threading.Lock()

def running_task(request_id, image_data, hour, minute):
    score_circle, score_hands = score_circle_and_hands(image_data, hour, minute)
    # 使用锁来安全地更新结果字典
    with results_lock:
        results[request_id] = {"score_circle": score_circle, "score_hands": score_hands}

@app.route('/get_img_score', methods=['POST'])
def get_img_score():
    input_json = request.get_json()
    request_id = input_json.get("request_id", "111")
    image_data = input_json.get("image")
    hour = int(input_json.get("hour", 9))
    minute = int(input_json.get("minute", 30))
    request_ip = request.remote_addr

    if not image_data:
        return jsonify({'error': 'No data provided'}), 400

    # 启动后台线程进行长时间计算
    threading.Thread(target=running_task, args=(request_id, image_data, hour, minute)).start()

    score, score_detail = -1, {}

    result = {
        "score": score,
        "score_detail": score_detail
    }

    return jsonify(result)


@app.route('/get_score_result/<request_id>', methods=['GET'])
def get_score_result(request_id):
    # 使用锁来安全地访问结果字典
    with results_lock:
        if request_id in results:
            result = results[request_id]
            # 移除已返回的结果
            del results[request_id]
            return jsonify({'request_id': request_id, 'result': result})
        else:
            return jsonify({'request_id': request_id, 'error': 'Result not found'}), 404

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=7890, debug=True)