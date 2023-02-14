import cv2 
from flask import Flask, render_template, Response
import numpy as np
from skimage.transform import resize
from fun.invisibleMinds import *
import tensorflow as tf


app = Flask(__name__)
model = videoFightModel2(tf,wight='invisibleMinds.hdfs')


cap = cv2.VideoCapture('vi.avi')
cap2 = cv2.VideoCapture('v1.mp4')


def gen():
    
    i = 0
    frames = np.zeros((30, 160, 160, 3), dtype=np.float16)
    old = []
    j = 0

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            font = cv2.FONT_HERSHEY_SIMPLEX
            if i > 29:
                ysdatav2 = np.zeros((1, 30, 160, 160, 3), dtype=np.float16)
                ysdatav2[0][:][:] = frames
                predaction = pred_fight(model,ysdatav2,acuracy=0.96)
                if predaction[0] == True:
                    cv2.imshow('video', frame)
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    vio = cv2.VideoWriter("./videos/output-"+str(j)+".avi", fourcc, 10.0, (fwidth,fheight))
                    vio = cv2.VideoWriter("./videos/output-"+str(j)+".mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, (300, 400))
                    print('Violance detected here ...')
                    print("timestamp is: ", str(cap.get(cv2.CAP_PROP_POS_MSEC)))
                    for x in old:
                        vio.write(x)
                        #print(x) will print the frame which contains Violence...
                    vio.release
                    cv2.putText(frame, 
                        'Violence Detected!!!', 
                        (100, 200), 
                        font, 3, 
                        (238, 75, 43), 
                        5, 
                        cv2.LINE_4)            
                i = 0
                j += 1
                frames = np.zeros((30, 160, 160, 3), dtype=np.float16)
                old = []
            else:
                frm = resize(frame,(160,160,3))
                old.append(frame)
                fshape = frame.shape
                fheight = fshape[0]
                fwidth = fshape[1]
                frm = np.expand_dims(frm,axis=0)
                if(np.max(frm)>1):
                    frm = frm/255.
                frames[i][:] = frm
                i+=1
        else:
            break         
            #cv2.imshow("countours", image)
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        #time.sleep(0.1)
        key = cv2.waitKey(20)
        if key == 27:
           break

def gen1():
    
    i = 0
    frames = np.zeros((30, 160, 160, 3), dtype=np.float16)
    old = []
    j = 0

    while(cap2.isOpened()):
        ret, frame = cap2.read()
        if ret == True:
            font = cv2.FONT_HERSHEY_SIMPLEX
            if i > 29:
                ysdatav2 = np.zeros((1, 30, 160, 160, 3), dtype=np.float16)
                ysdatav2[0][:][:] = frames
                predaction = pred_fight(model,ysdatav2,acuracy=0.96)
                if predaction[0] == True:
                    print('Violance detected here ...')
                    print("timestamp is: ", str(cap.get(cv2.CAP_PROP_POS_MSEC)))
                    cv2.imshow('video', frame)
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    vio = cv2.VideoWriter("./videos/output-"+str(j)+".avi", fourcc, 10.0, (fwidth,fheight))
                    vio = cv2.VideoWriter("./videos/output-"+str(j)+".mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, (300, 400))
                    for x in old:
                        vio.write(x)
                        #print(x) will print the frame which contains Violence...
                    vio.release
                    cv2.putText(frame, 
                        'Violence Detected!!!', 
                        (100, 200), 
                        font, 3, 
                        (238, 75, 43), 
                        5, 
                        cv2.LINE_4)            
                i = 0
                j += 1
                frames = np.zeros((30, 160, 160, 3), dtype=np.float16)
                old = []
            else:
                frm = resize(frame,(160,160,3))
                old.append(frame)
                fshape = frame.shape
                fheight = fshape[0]
                fwidth = fshape[1]
                frm = np.expand_dims(frm,axis=0)
                if(np.max(frm)>1):
                    frm = frm/255.
                frames[i][:] = frm
                i+=1
        else:
            break         
            #cv2.imshow("countours", image)
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        #time.sleep(0.1)
        key = cv2.waitKey(20)
        if key == 27:
           break

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed2')
def video_feed2():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen1(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)
