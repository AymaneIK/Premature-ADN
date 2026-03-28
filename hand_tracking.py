import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

FINGER_NUMBERS = {
    "Thumb":  5,
    "Index":  1,
    "Middle": 2,
    "Ring":   3,
    "Pinky":  4,
}

COLOR_LEFT  = (255, 180,  50)
COLOR_RIGHT = ( 50, 220, 255)
COLOR_UP    = ( 50, 255, 120)
COLOR_DOWN  = ( 80,  80, 255)
COLOR_BG    = (0, 0, 0)


def draw_badge(img, text, pos, color, font_scale=0.55, thickness=1):
    x, y = pos
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX,
                                         font_scale, thickness)
    pad = 5
    cv2.rectangle(img, (x - pad, y - th - pad),
                  (x + tw + pad, y + baseline + pad), COLOR_BG, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX,
                font_scale, color, thickness, cv2.LINE_AA)


def fingers_up(hand_landmarks, handedness_label):
    lm = hand_landmarks.landmark
    extended = []

    # ✅ FIXED THUMB (stable with mirror)
    thumb_tip = lm[4]
    thumb_ip  = lm[3]

    if handedness_label == "Right":
        extended.append(thumb_tip.x > thumb_ip.x)
    else:
        extended.append(thumb_tip.x < thumb_ip.x)

    # Other fingers
    for tip_id, pip_id in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        extended.append(lm[tip_id].y < lm[pip_id].y)

    return extended


def main():
    cap = cv2.VideoCapture(0)

    cap.set(3, 1280)
    cap.set(4, 720)

    cv2.namedWindow("Hand Detection", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Hand Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    ) as hands:

        while True:
            success, frame = cap.read()
            if not success:
                break

            # ✅ Mirror FIRST (important)
            frame = cv2.flip(frame, 1)

            img_h, img_w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_lm, hand_info in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness,
                ):

                    label = hand_info.classification[0].label
                    hand_color = COLOR_LEFT if label == "Left" else COLOR_RIGHT

                    mp_draw.draw_landmarks(
                        frame, hand_lm,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

                    up_flags = fingers_up(hand_lm, label)
                    count_up = sum(up_flags)

                    lm = hand_lm.landmark

                    for i, (tip_id, name) in enumerate(zip(FINGER_TIPS, FINGER_NAMES)):
                        cx = int(lm[tip_id].x * img_w)
                        cy = int(lm[tip_id].y * img_h)

                        num    = FINGER_NUMBERS[name]
                        is_up  = up_flags[i]
                        color  = COLOR_UP if is_up else COLOR_DOWN
                        status = "UP" if is_up else "down"

                        text = f"#{num} {name} ({status})"

                        cv2.circle(frame, (cx, cy), 8, color, -1)
                        draw_badge(frame, text, (cx + 12, cy + 5), color)

                    wrist_x = int(lm[0].x * img_w)
                    wrist_y = int(lm[0].y * img_h)

                    draw_badge(
                        frame,
                        f"{label} | {count_up} finger(s)",
                        (wrist_x - 60, wrist_y + 30),
                        hand_color,
                        font_scale=0.65,
                        thickness=2,
                    )

            cv2.imshow("Hand Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()