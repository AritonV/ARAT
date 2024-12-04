import json
import os
import tempfile
import time

import cv2
import numpy as np
import pygame
import supervision as sv
from flask import Flask, jsonify, render_template, request
from gtts import gTTS
from ultralytics import YOLO

app = Flask(__name__)

BALL = 0
BLUE_BLOCK = 1
GREEN_BLOCK = 2
MARBLE = 3
RED_BLOCK = 4
SHORT_TUBE = 5
STONE = 6
TALL_TUBE = 7
WASHER = 8
YELLOW_BLOCK = 9

scores = []
x1, y1 = 973, 695
x2, y2 = 1175, 839
x3, y3 = 363, 145
x4, y4 = 670, 306
pt1 = (x1, y1)
pt2 = (x2, y2)
pt3 = (x3, y3)
pt4 = (x4, y4)
start = False
end = False
drawing = False
ZONE_IN_POLYGON = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
ZONE_OUT_POLYGON = np.array([[x3, y3], [x4, y3], [x4, y4], [x3, y4]])


def say_instruction(text):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_file_path = temp_file.name
    tts = gTTS(text=text, lang="en")
    tts.save(temp_file_path)
    pygame.mixer.init()
    sound = pygame.mixer.Sound(temp_file_path)
    sound.play()
    os.remove(temp_file_path)


def draw_box(event, x, y, flags, param):
    global pt1, pt2, pt3, pt4, start, end, drawing
    if drawing:
        if event == cv2.EVENT_LBUTTONDOWN:
            pt1 = (x, y)
            start = True
        if event == cv2.EVENT_RBUTTONDOWN:
            pt3 = (x, y)
            end = True
        if event == cv2.EVENT_LBUTTONUP:
            pt2 = (x, y)
            start = False
        if event == cv2.EVENT_RBUTTONUP:
            pt4 = (x, y)
            end = False
        if event == cv2.EVENT_MOUSEMOVE:
            if start:
                pt2 = (x, y)
            if end:
                pt4 = (x, y)


def main(
    cid: int,
    obj: str = "object",
    msg: str = "",
    poly_zones=[ZONE_IN_POLYGON, ZONE_OUT_POLYGON],
):
    global drawing
    if msg == "":
        msg = f"Please move the {obj}"
    text1 = False
    speak = True
    zones = True
    redefine = False
    timer = True
    time_taken = 0
    score = 0
    cap = cv2.VideoCapture(1)
    model = YOLO("ARAT.pt")
    model.to("cuda")
    color = sv.ColorPalette.from_hex(
        [
            "#bbbbbb",
            "#0000ff",
            "#00ff00",
            "#ff3bfc",
            "#ff0000",
            "#9191ff",
            "#00d99b",
            "#ff9500",
            "#737373",
            "#ffff00",
        ]
    )
    box_annotator = sv.BoxAnnotator(color=color)
    while True:
        ret, frame = cap.read()
        if text1:
            frame = cv2.putText(
                frame,
                "Drag Left click to draw the starting zone, and drag right click to draw the ending zone",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            frame = cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
            frame = cv2.rectangle(frame, pt3, pt4, (0, 0, 255), 2)
            redefine = True
            zones = True
        else:
            if zones:
                zones = False
                if redefine:
                    poly_in = np.array(
                        [
                            [pt1[0], pt1[1]],
                            [pt2[0], pt1[1]],
                            [pt2[0], pt2[1]],
                            [pt1[0], pt2[1]],
                        ]
                    )
                else:
                    poly_in = poly_zones[0]
                zone_in = sv.PolygonZone(
                    polygon=poly_in,
                    frame_resolution_wh=(1920, 1080),
                    triggering_position=sv.Position.CENTER,
                )
                zone_in_annotator = sv.PolygonZoneAnnotator(
                    zone=zone_in,
                    color=sv.Color.GREEN,
                    thickness=2,
                    text_thickness=2,
                    text_scale=1,
                )
                if redefine:
                    poly_out = np.array(
                        [
                            [pt3[0], pt3[1]],
                            [pt4[0], pt3[1]],
                            [pt4[0], pt4[1]],
                            [pt3[0], pt4[1]],
                        ]
                    )
                    redefine = False
                else:
                    poly_out = poly_zones[1]
                zone_out = sv.PolygonZone(
                    polygon=poly_out,
                    frame_resolution_wh=(1920, 1080),
                    triggering_position=sv.Position.CENTER,
                )
                zone_out_annotator = sv.PolygonZoneAnnotator(
                    zone=zone_out,
                    color=sv.Color.RED,
                    thickness=2,
                    text_thickness=2,
                    text_scale=1,
                )
            else:
                result = model(frame, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(result)
                detections = detections[detections.class_id == cid]
                detections = detections[detections.confidence > 0.8]
                labels = [
                    f"{model.model.names[class_id]} {confidence:0.2f}"
                    for _, _, confidence, class_id, _, _, in detections
                ]
                frame = box_annotator.annotate(
                    scene=frame, detections=detections, labels=labels
                )
                mask_in = zone_in.trigger(detections=detections)
                detection_in = detections[mask_in]
                if len(detection_in) > 0 and cid in detection_in.class_id:
                    if timer:
                        start_time = time.time()
                        time_taken = 0
                        timer = False
                mask_out = zone_out.trigger(detections=detections)
                detection_out = detections[mask_out]
                if len(detection_out) > 0 and cid in detection_out.class_id:
                    if not timer:
                        end_time = time.time()
                        time_taken = round(end_time - start_time, 1)
                        if time_taken > 0 and time_taken < 5:
                            score = 3
                        elif time_taken >= 5 and time_taken < 20:
                            score = 2
                        elif time_taken >= 20 and time_taken < 60:
                            score = 1
                        else:
                            score = 0
                        say_instruction("Great job!")
                        scores.append({"score": score, "object": obj})
                        cap.release()
                        break
                frame = cv2.putText(
                    frame,
                    f"Time:{time_taken if timer or time_taken > 0 else round(time.time() - start_time, 1)}s",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2,
                )
                frame = cv2.putText(
                    frame,
                    f"Please move the {obj}",
                    (200, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2,
                )
                if speak:
                    say_instruction(msg)
                    speak = False
                if not timer:
                    end_time = time.time()
                    time_taken = round(end_time - start_time, 1)
                if time_taken > 60:
                    score = 0
                    say_instruction("Amazing try!")
                    scores.append({"score": score, "object": obj})
                    cap.release()
                    break
                frame = zone_in_annotator.annotate(scene=frame)
                frame = zone_out_annotator.annotate(scene=frame)
        cv2.namedWindow("ARAT", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("ARAT", draw_box)
        cv2.imshow("ARAT", frame)
        if cv2.waitKey(1) & 0xFF == ord("e"):
            text1 = not text1
            drawing = not drawing


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run_analysis", methods=["POST"])
def start():
    global scores
    scores = []
    total_score = 0
    name = request.form["name"]
    main(
        RED_BLOCK,
        "red cube",
        f"Hello {name}, welcome to the ARAT test! For each of the objects you see in front of you, please move them to the right side of the box, and then you will have to move the objects from the starting zone (colored green) to the ending zone (colored red). Good luck! Now please move the red cube.",
    )
    main(YELLOW_BLOCK, "yellow cube")
    main(GREEN_BLOCK, "green cube")
    main(BLUE_BLOCK, "blue cube")
    main(BALL, "tennis ball")
    main(STONE, "stone")
    p1 = np.array([[1216, 532], [1287, 532], [1287, 770], [1216, 770]])
    p2 = np.array([[952, 509], [1030, 509], [1030, 779], [952, 779]])
    main(
        TALL_TUBE,
        "tall orange tube",
        "Now please move the wooden board with the pegs, so that it's positioned vertically in the middle of the box, facing from the shortest to the longest peg. Now please move the tall orange tube.",
        [p1, p2],
    )
    p1 = np.array([[1096, 644], [1151, 644], [1151, 780], [1096, 780]])
    p2 = np.array([[835, 573], [912, 573], [912, 735], [835, 735]])
    main(SHORT_TUBE, "short blue tube", "", [p1, p2])
    p1 = np.array([[1293, 787], [1379, 787], [1379, 827], [1293, 827]])
    p2 = np.array([[1210, 717], [1283, 717], [1283, 735], [1210, 735]])
    main(
        WASHER,
        "washer",
        "You will have to move the washer from the bottom left corner of the board to the middle of the shortest peg.",
        [p1, p2],
    )
    main(
        MARBLE,
        "marble with index finger",
        "Now please put back the wooden board and place the metal tins on the box, one on the top and one on the bottom. Now please move the marble with index finger.",
    )
    main(MARBLE, "marble with middle finger")
    main(MARBLE, "marble with ring finger")
    say_instruction(
        "Congratulations! You have completed the ARAT test. Thank you for participating!"
    )
    cv2.destroyAllWindows()
    if os.path.isfile(f"./tests/{name}_tests.json"):
        with open(f"./tests/{name}_tests.json", "r") as f:
            all_tests = json.load(f)
    else:
        all_tests = []
    for score in scores:
        total_score += score["score"]
    new_test = {
        "scores": scores,
        "total_score": total_score,
        "date": time.strftime("%H:%M %d-%m-%Y", time.localtime(time.time())),
    }
    all_tests.append(new_test)
    new_all_tests = json.dumps(all_tests, indent=2)
    with open(f"./tests/{name}_tests.json", "w") as f:
        f.write(new_all_tests)
    return jsonify(all_tests)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
