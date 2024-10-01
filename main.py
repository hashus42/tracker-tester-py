import cv2 as cv

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

drawBox = [0, 0, 0, 0]
mouseDown = False
mouseUp = False
start_x = 0
start_y = 0
roi_nano = None
initialize_tracker = False
init = False

tracker_nano = None

cap = cv.VideoCapture("data/19028313-hd_1920_1080_30fps.mp4")


def on_mouse(event, x, y, flags, param):
    global mouseDown, mouseUp, start_x, start_y, roi_nano, initialize_tracker, drawBox
    if event == cv.EVENT_LBUTTONDOWN:
        mouseDown = True
        mouseUp = False
        start_x = x
        start_y = y
        drawBox[0] = start_x
        drawBox[1] = start_y
    elif mouseDown and event == cv.EVENT_MOUSEMOVE:
        if x > start_x:
            drawBox[2] = x - start_x
        else:
            drawBox[2] = start_x - x
            drawBox[0] = x
        if y > start_y:
            drawBox[3] = y - start_y
        else:
            drawBox[3] = start_y - y
            drawBox[1] = y
    elif event == cv.EVENT_LBUTTONUP:
        mouseDown = False
        mouseUp = True
        initialize_tracker = True


while True:
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)

    if frame is not None:
        frame_prc_nano = frame.copy()
        frame_prc_boosting = frame.copy()
        frame_prc_mil = frame.copy()
        frame_prc_kcf = frame.copy()
        frame_prc_tld = frame.copy()
        frame_prc_medianflow = frame.copy()
        frame_prc_goturn = frame.copy()
        frame_prc_mosse = frame.copy()
        frame_prc_csrt = frame.copy()
    else:
        print("Frame is None")
        break

    start = cv.getTickCount()

    if mouseDown:
        pt1 = (drawBox[0], drawBox[1])
        pt2 = (drawBox[0] + drawBox[2], drawBox[1] + drawBox[3])
        cv.rectangle(frame, pt1, pt2, (0, 255, 0), 2)

    if initialize_tracker:
        tracker_nano = cv.TrackerNano.create()
        tracker_mil = cv.TrackerMIL.create()
        tracker_kcf = cv.TrackerKCF.create()
        # tracker_tld = cv.TrackerTLD.create()
        # tracker_medianflow = cv.TrackerMedianFlow.create()
        # tracker_goturn = cv.TrackerGOTURN.create()
        # tracker_mosse = cv.TrackerMOSSE.create()
        tracker_csrt = cv.TrackerCSRT.create()
        if tracker_nano is not None:
            tracker_nano.init(frame_prc_nano, drawBox)
            tracker_mil.init(frame_prc_mil, drawBox)
            tracker_kcf.init(frame_prc_kcf, drawBox)
            # tracker_tld.init(frame_prc_tld, drawBox)
            # tracker_medianflow.init(frame_prc_medianflow, drawBox)
            # tracker_goturn.init(frame_prc_goturn, drawBox)
            # tracker_mosse.init(frame_prc_mosse, drawBox)
            tracker_csrt.init(frame_prc_csrt, drawBox)
            initialize_tracker = False
            init = True
        else:
            print("Tracker is None")
            break

    if init:
        success_nano, roi_nano = tracker_nano.update(frame_prc_nano)
        if success_nano:
            cv.rectangle(frame, roi_nano, (0, 255, 0), 2)
        success_mil, roi_mil = tracker_mil.update(frame_prc_mil)
        if success_mil:
            cv.rectangle(frame, roi_mil, (255, 0, 0), 2)
        success_kcf, roi_kcf = tracker_kcf.update(frame_prc_kcf)
        if success_kcf:
            cv.rectangle(frame, roi_kcf, (120, 12, 80), 2)
        # success_tld, roi_tld = tracker_tld.update(frame_prc_tld)
        # if success_tld:
        #     cv.rectangle(frame, roi_tld, (0, 255, 0), 2)
        # success_medianflow, roi_medianflow = tracker_medianflow.update(frame_prc_medianflow)
        # if success_medianflow:
        #     cv.rectangle(frame, roi_medianflow, (0, 255, 0), 2)
        # success_goturn, roi_goturn = tracker_goturn.update(frame_prc_goturn)
        # if success_goturn:
        #     cv.rectangle(frame, roi_goturn, (0, 255, 0), 2)
        # success_mosse, roi_mosse = tracker_mosse.update(frame_prc_mosse)
        # if success_mosse:
        #     cv.rectangle(frame, roi_mosse, (0, 255, 0), 2)
        success_csrt, roi_csrt = tracker_csrt.update(frame_prc_csrt)
        if success_csrt:
            cv.rectangle(frame, roi_csrt, (0, 0, 255), 2)

    fps = cv.getTickFrequency() / (cv.getTickCount() - start)

    # Show FPS
    cv.putText(frame, "FPS: " + str(int(fps)), (100, 50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    # Display the resulting frames
    cv.namedWindow("Nano_Tracker", cv.WINDOW_NORMAL)
    cv.resizeWindow("Nano_Tracker", 640, 480)
    cv.moveWindow("Nano_Tracker", SCREEN_WIDTH//2 - 640//2, SCREEN_HEIGHT//2 - 480//2)
    cv.imshow("Nano_Tracker", frame)
    cv.setMouseCallback("Nano_Tracker", on_mouse)

    key = cv.waitKey(1) & 0xFF
    if key == ord("q"):
        break
