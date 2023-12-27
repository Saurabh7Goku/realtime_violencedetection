import cv2
from flask import Flask, render_template, Response, request, jsonify
import numpy as np
from skimage.transform import resize
import tensorflow as tf
from fun.invisibleMinds import pred_fight, videoFightModel2

app = Flask(__name__)
model = videoFightModel2(tf, wight='invisibleminds.hdfs')

cap = None
video_file_path = None
url = None
violence_detections = []

def initialize_video_capture(file_path):
    global cap
    cap = cv2.VideoCapture(file_path)

def process_frame(frame, frames):
    font = cv2.FONT_HERSHEY_SIMPLEX
    frm = resize(frame, (160, 160, 3))
    frm = np.expand_dims(frm, axis=0)
    if np.max(frm) > 1:
        frm = frm / 255.

    ysdatav2 = np.zeros((1, 30, 160, 160, 3), dtype=np.float16)
    ysdatav2[0][:][:] = frames
    predaction = pred_fight(model, ysdatav2, acuracy=0.96)

    if predaction[0]:
        cv2.putText(frame,
                    'Violence Detected!!!',
                    (100, 200),
                    font, 3,
                    (238, 75, 43),
                    5,
                    cv2.LINE_4)

        # Store violence detection information
        detection_info = {
            'timestamp': str(cap.get(cv2.CAP_PROP_POS_MSEC)),
            'video_url': f'/videos/output-{len(violence_detections)}.mp4'
        }
        violence_detections.append(detection_info)

        # Update the sidebar dynamically
        update_sidebar()

    return frame

def update_sidebar():
    global violence_detections
    data = {
        'violence_detections': violence_detections
    }
    # Broadcast the updated data to all connected clients
    socketio.emit('update_sidebar', data)

def gen():
    global cap, violence_detections
    i = 0
    frames = np.zeros((30, 160, 160, 3), dtype=np.float16)
    old = []
    j = 0

    while cap.isOpened():
        font = cv2.FONT_HERSHEY_SIMPLEX
        ret, frame = cap.read()
        if ret:
            # Process every third frame
            if i % 3 == 0:
                processed_frame = process_frame(frame.copy(), frames)
                ret, jpeg = cv2.imencode('.jpg', processed_frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

            if i > 29:
                ysdatav2 = np.zeros((1, 30, 160, 160, 3), dtype=np.float16)
                ysdatav2[0][:][:] = frames
                predaction = pred_fight(model, ysdatav2, acuracy=0.96)
                if predaction[0]:
                    cv2.imshow('video', frame)
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    vio = cv2.VideoWriter(f"./videos/output-{j}.avi", fourcc, 10.0, (frame.shape[1], frame.shape[0]))
                    vio = cv2.VideoWriter(f"./videos/output-{j}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10,
                                          (frame.shape[1], frame.shape[0]))
                    print('Violence detected here ...')
                    print("timestamp is: ", str(cap.get(cv2.CAP_PROP_POS_MSEC)))
                    for x in old:
                        vio.write(x)
                    vio.release
                    cv2.putText(frame,
                                'Violence Detected!!!',
                                (100, 200),
                                font, 3,
                                (238, 75, 43),
                                5,
                                cv2.LINE_4)

                    # Store violence detection information
                    detection_info = {
                        'timestamp': str(cap.get(cv2.CAP_PROP_POS_MSEC)),
                        'video_url': f'/videos/output-{j}.mp4'
                    }
                    violence_detections.append(detection_info)

                i = 0
                j += 1
                frames = np.zeros((30, 160, 160, 3), dtype=np.float16)
                old = []
            else:
                frm = resize(frame, (160, 160, 3))
                old.append(frame)
                fshape = frame.shape
                fheight = fshape[0]
                fwidth = fshape[1]
                frm = np.expand_dims(frm, axis=0)
                if np.max(frm) > 1:
                    frm = frm / 255.
                frames[i][:] = frm
                i += 1
        else:
            break

    cap.release()

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html', violence_detections=violence_detections)

@app.route('/process_video', methods=['POST'])
def process_video():
    global video_file_path, violence_detections, cap
    video_file = request.files.get('video_file')
    video_url = request.form.get('video_url')

    # Clear the sidebar when selecting a new video
    violence_detections = []

    if video_file:
        video_file_path = video_file.filename
        video_file.save(video_file_path)
        initialize_video_capture(video_file_path)

    elif video_url:
        url = video_url
        # video_url.save(url)
        initialize_video_capture(url)

    return render_template('index.html', violence_detections=violence_detections)

@app.route('/end_video', methods=['POST'])
def end_video():
    global cap
    if cap is not None:
        cap.release()
    return jsonify(success=True)

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    from flask_socketio import SocketIO

    socketio = SocketIO(app)

    @socketio.on('disconnect')
    def handle_disconnect():
        global cap
        if cap is not None:
            cap.release()

    socketio.run(app, debug=True)
