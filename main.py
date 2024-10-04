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
paused = False

tracker_nano = None

cap = cv.VideoCapture("data/amman-stunt.mp4")

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
    start = cv.getTickCount()
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)

    if frame is not None:
        # Process frames
        frame_prc_nano = frame.copy()
        frame_prc_boosting = frame.copy()
        frame_prc_mil = frame.copy()
        frame_prc_kcf = frame.copy()
        # frame_prc_tld = frame.copy()
        frame_prc_medianflow = frame.copy()
        frame_prc_goturn = frame.copy()
        # frame_prc_mosse = frame.copy()
        frame_prc_csrt = frame.copy()
        # Show frames
        frame_nano = frame.copy()
        frame_boosting = frame.copy()
        frame_mil = frame.copy()
        frame_kcf = frame.copy()
        # frame_tld = frame.copy()
        frame_medianflow = frame.copy()
        frame_goturn = frame.copy()
        # frame_mosse = frame.copy()
        frame_csrt = frame.copy()
    else:
        print("Frame is None")
        break

    if mouseDown:
        pt1 = (drawBox[0], drawBox[1])
        pt2 = (drawBox[0] + drawBox[2], drawBox[1] + drawBox[3])
        cv.rectangle(frame, pt1, pt2, (0, 255, 0), 2)

    if initialize_tracker:
        tracker_nano = cv.TrackerNano.create()
        tracker_mil = cv.TrackerMIL.create()
        tracker_kcf = cv.TrackerKCF.create()
        # tracker_tld = cv.TrackerTLD.create()
        tracker_medianflow = cv.legacy.TrackerMedianFlow.create()
        tracker_goturn = cv.TrackerGOTURN.create()
        # tracker_mosse = cv.TrackerMOSSE.create()
        tracker_csrt = cv.TrackerCSRT.create()
        if tracker_nano is not None:
            tracker_nano.init(frame_prc_nano, drawBox)
            tracker_mil.init(frame_prc_mil, drawBox)
            tracker_kcf.init(frame_prc_kcf, drawBox)
            # tracker_tld.init(frame_prc_tld, drawBox)
            tracker_medianflow.init(frame_prc_medianflow, drawBox)
            tracker_goturn.init(frame_prc_goturn, drawBox)
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
            cv.rectangle(frame_nano, roi_nano, (0, 200, 0), 2)
        success_mil, roi_mil = tracker_mil.update(frame_prc_mil)
        if success_mil:
            cv.rectangle(frame_mil, roi_mil, (255, 0, 0), 2)
        success_kcf, roi_kcf = tracker_kcf.update(frame_prc_kcf)
        if success_kcf:
            cv.rectangle(frame_kcf, roi_kcf, (120, 12, 80), 2)
        # success_tld, roi_tld = tracker_tld.update(frame_prc_tld)
        # if success_tld:
        #     cv.rectangle(frame, roi_tld, (0, 255, 0), 2)
        success_medianflow, roi_medianflow = tracker_medianflow.update(frame_prc_medianflow)
        if success_medianflow:
            roi_medianflow = [int(i) for i in roi_medianflow]
            x, y, w, h = roi_medianflow
            cv.rectangle(frame_medianflow, (x, y), (x + w, y + h), (120, 12, 160), 2)
        success_goturn, roi_goturn = tracker_goturn.update(frame_prc_goturn)
        if success_goturn:
            cv.rectangle(frame_goturn, roi_goturn, (80, 12, 160), 2)
        # success_mosse, roi_mosse = tracker_mosse.update(frame_prc_mosse)
        # if success_mosse:
        #     cv.rectangle(frame, roi_mosse, (0, 255, 0), 2)
        success_csrt, roi_csrt = tracker_csrt.update(frame_prc_csrt)
        if success_csrt:
            cv.rectangle(frame_csrt, roi_csrt, (0, 0, 255), 2)

    fps = cv.getTickFrequency() / (cv.getTickCount() - start)

    # Show FPS
    cv.putText(frame_nano, "NANO Tracker ", (100, 70), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 0), 2)
    cv.putText(frame_nano, "FPS: " + str(int(fps)), (100, 90), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 40, 50), 2)
    cv.putText(frame_mil, "MIL Tracker ", (100, 110), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
    cv.putText(frame_mil, "FPS: " + str(int(fps)), (100, 130), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 40, 50), 2)
    cv.putText(frame_kcf, "KCF Tracker ", (100, 150), cv.FONT_HERSHEY_SIMPLEX, 0.75, (120, 12, 80), 2)
    cv.putText(frame_kcf, "FPS: " + str(int(fps)), (100, 170), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 40, 50), 2)
    cv.putText(frame_medianflow, "Medianflow Tracker ", (100, 190), cv.FONT_HERSHEY_SIMPLEX, 0.75, (120, 12, 160), 2)
    cv.putText(frame_medianflow, "FPS: " + str(int(fps)), (100, 210), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 40, 50), 2)
    cv.putText(frame_goturn, "Goturn Tracker ", (100, 230), cv.FONT_HERSHEY_SIMPLEX, 0.75, (80, 12, 160), 2)
    cv.putText(frame_goturn, "FPS: " + str(int(fps)), (100, 250), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 40, 50), 2)
    cv.putText(frame_csrt, "CSRT Tracker ", (100, 270), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv.putText(frame_csrt, "FPS: " + str(int(fps)), (100, 290), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 40, 50), 2)

    # Display the resulting frames
    cv.namedWindow("Nano_Tracker", cv.WINDOW_NORMAL)
    cv.resizeWindow("Nano_Tracker", 640, 480)
    cv.moveWindow("Nano_Tracker", 0, 0)
    cv.imshow("Nano_Tracker", frame_nano)

    cv.namedWindow("MIL_Tracker", cv.WINDOW_NORMAL)
    cv.resizeWindow("MIL_Tracker", 640, 480)
    cv.moveWindow("MIL_Tracker", 640, 0)
    cv.imshow("MIL_Tracker", frame_mil)

    cv.namedWindow("KCF_Tracker", cv.WINDOW_NORMAL)
    cv.resizeWindow("KCF_Tracker", 640, 480)
    cv.moveWindow("KCF_Tracker", 1280, 0)
    cv.imshow("KCF_Tracker", frame_kcf)

    cv.namedWindow("Medianflow_Tracker", cv.WINDOW_NORMAL)
    cv.resizeWindow("Medianflow_Tracker", 640, 480)
    cv.moveWindow("Medianflow_Tracker", 0, 530)
    cv.imshow("Medianflow_Tracker", frame_medianflow)

    cv.namedWindow("Goturn_Tracker", cv.WINDOW_NORMAL)
    cv.resizeWindow("Goturn_Tracker", 640, 480)
    cv.moveWindow("Goturn_Tracker", 640, 530)
    cv.imshow("Goturn_Tracker", frame_goturn)

    cv.namedWindow("CSRT_Tracker", cv.WINDOW_NORMAL)
    cv.resizeWindow("CSRT_Tracker", 640, 480)
    cv.moveWindow("CSRT_Tracker", 1280, 530)
    cv.imshow("CSRT_Tracker", frame_csrt)

    cv.setMouseCallback("Nano_Tracker", on_mouse)

    key = cv.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord(" "):
        paused = not paused
        while paused:
            key = cv.waitKey(1) & 0xFF
            if key == ord(" "):
                paused = False
            elif key == ord("q"):
                break

cap.release()
cv.destroyAllWindows()
