import cv2
import os
import numpy as np
from ultralytics import YOLO
from collections import deque

def calculate_angle(v1, v2):
    """Calculate the angle between two vectors."""
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_v1, unit_v2)
    # Clip to avoid numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle = np.arccos(dot_product) * (180 / np.pi)  # Convert to degrees
    return angle


def is_hand_raised(wrist, elbow, shoulder, angle_threshold=130):
    """Check if the hand is raised based on keypoints."""
    if wrist is None or elbow is None or shoulder is None:
        return False

    v1 = np.array([elbow[0] - wrist[0], elbow[1] - wrist[1]])
    v2 = np.array([elbow[0] - shoulder[0], elbow[1] - shoulder[1]])
    angle = calculate_angle(v1, v2)
    # Less restrictive angle threshold and stronger height requirement
    return angle <= angle_threshold and wrist[1] < shoulder[1]


def is_standing(hip, knee, ankle, angle_threshold=130):
    """Check if a person is standing based on keypoints."""
    if hip is None or knee is None or ankle is None:
        return False

    # Vertical alignment check with tolerance
    if not (hip[1] < knee[1] and knee[1] < ankle[1]):
        return False

    v1 = np.array([hip[0] - knee[0], hip[1] - knee[1]])
    v2 = np.array([ankle[0] - knee[0], ankle[1] - knee[1]])
    angle = calculate_angle(v1, v2)
    return angle > angle_threshold


def detect_interaction(hand, phones, overlap_threshold=0.3):
    """Check if a hand is interacting with a phone based on bounding box overlap."""
    hx1, hy1, hx2, hy2 = hand
    hand_area = (hx2 - hx1) * (hy2 - hy1)

    for px1, py1, px2, py2 in phones:
        # Calculate intersection
        ix1 = max(hx1, px1)
        iy1 = max(hy1, py1)
        ix2 = min(hx2, px2)
        iy2 = min(hy2, py2)

        if ix2 > ix1 and iy2 > iy1:
            intersection_area = (ix2 - ix1) * (iy2 - iy1)
            overlap_ratio = intersection_area / hand_area
            if overlap_ratio > overlap_threshold:
                return True
    return False


def map_hands_to_people(hands, people, keypoints, threshold=80):
    """Map each detected hand to the nearest person based on wrist position and distance."""
    hand_to_person = {}

    for hand in hands:
        hx1, hy1, hx2, hy2 = hand
        hand_center = np.array([(hx1 + hx2) / 2, (hy1 + hy2) / 2])
        min_distance = float('inf')
        mapped_person = None

        for i, keypoint in enumerate(keypoints):
            left_wrist, right_wrist = keypoint[9], keypoint[10]
            for wrist in [left_wrist, right_wrist]:
                if wrist is None:
                    continue
                wrist_pos = np.array(wrist)
                distance = np.linalg.norm(hand_center - wrist_pos)
                if distance < min_distance and distance < threshold:
                    min_distance = distance
                    mapped_person = i

        if mapped_person is not None:
            if mapped_person in hand_to_person:
                hand_to_person[mapped_person].append(hand)
            else:
                hand_to_person[mapped_person] = [hand]

    return hand_to_person


def is_hand_waving(wrist_positions, min_frames=10, min_lateral_movement=35, min_vertical_std=10,
                   min_direction_changes=2):
    """
    Determine if a hand is waving based on wrist position history with improved thresholds.

    Args:
        wrist_positions: Deque of wrist positions over time
        min_frames: Minimum number of frames to consider (increased for more stability)
        min_lateral_movement: Minimum total horizontal movement to qualify as waving
        min_vertical_std: Minimum standard deviation for vertical movement
        min_direction_changes: Minimum number of direction changes required

    Returns:
        Boolean indicating if waving is detected
    """
    if len(wrist_positions) < min_frames:
        return False

    # Extract x and y coordinates
    wrist_array = np.array(wrist_positions)
    x_coords = wrist_array[:, 0]
    y_coords = wrist_array[:, 1]

    # Calculate horizontal movement
    x_diff = np.abs(np.diff(x_coords))
    total_lateral_movement = np.sum(x_diff)

    # Calculate vertical movement variability
    y_std = np.std(y_coords)

    # Check for oscillatory pattern in x direction (lateral movement)
    direction_changes = 0
    for i in range(1, len(x_diff)):
        if (x_coords[i + 1] - x_coords[i]) * (x_coords[i] - x_coords[i - 1]) < 0:
            direction_changes += 1

    # Hand waving involves lateral movement with some vertical variation and direction changes
    is_waving = (total_lateral_movement > min_lateral_movement and
                 y_std > min_vertical_std and
                 direction_changes >= min_direction_changes)

    return is_waving


def is_walking(ankle_positions, hip_positions, min_frames=15, min_movement=25, min_direction_changes=3):
    """
    Determine if a person is walking based on ankle and hip position history with improved thresholds.

    Args:
        ankle_positions: Deque of ankle positions over time
        hip_positions: Deque of hip positions over time
        min_frames: Minimum number of frames to consider (increased)
        min_movement: Minimum movement to qualify as walking
        min_direction_changes: Minimum number of direction changes required

    Returns:
        Boolean indicating if walking is detected
    """
    if len(ankle_positions) < min_frames or len(hip_positions) < min_frames:
        return False

    # Extract ankle coordinates
    ankle_array = np.array(ankle_positions)
    hip_array = np.array(hip_positions)

    # Calculate horizontal and vertical movement of ankles
    ankle_diffs = np.diff(ankle_array, axis=0)
    ankle_movement = np.sum(np.sqrt(np.sum(ankle_diffs ** 2, axis=1)))

    # Calculate hip movement (should be less than ankle movement for walking)
    hip_diffs = np.diff(hip_array, axis=0)
    hip_movement = np.sum(np.sqrt(np.sum(hip_diffs ** 2, axis=1)))

    # Calculate alternating pattern in ankle movements (typical of walking)
    ankle_x_diffs = ankle_diffs[:, 0]
    direction_changes = np.sum(ankle_x_diffs[1:] * ankle_x_diffs[:-1] < 0)

    # Walking involves forward movement with some alternating pattern
    is_walking = (ankle_movement > min_movement and
                  direction_changes >= min_direction_changes and
                  ankle_movement > hip_movement * 1.2)  # Ankles move significantly more than hips

    return is_walking


# New class for action state management with temporal smoothing
class ActionStateManager:
    def __init__(self, smoothing_window=10):
        self.states = {}  # Format: {person_id: {action: confidence}}
        self.history = {}  # Format: {person_id: {action: deque of booleans}}
        self.smoothing_window = smoothing_window

    def initialize_person(self, person_id):
        """Initialize tracking for a new person"""
        if person_id not in self.states:
            self.states[person_id] = {}
            self.history[person_id] = {}

    def update_state(self, person_id, action, is_active):
        """Update action state with temporal smoothing"""
        self.initialize_person(person_id)

        # Initialize history for this action if needed
        if action not in self.history[person_id]:
            self.history[person_id][action] = deque(maxlen=self.smoothing_window)

        # Add current detection to history
        self.history[person_id][action].append(is_active)

        # Calculate confidence as percentage of positive detections
        if len(self.history[person_id][action]) > 0:
            confidence = sum(self.history[person_id][action]) / len(self.history[person_id][action])
            self.states[person_id][action] = confidence

    def get_active_actions(self, person_id, confidence_threshold=0.6):
        """Get actions that exceed the confidence threshold"""
        if person_id not in self.states:
            return set()

        active_actions = set()
        for action, confidence in self.states[person_id].items():
            if confidence >= confidence_threshold:
                active_actions.add(action)

        return active_actions


def main(pose_model="models/yolo pose/yolo11n-pose.pt",
         object_detection_model="models/yolo phone hand detection/best.pt",
         cam_input="temp_videos/56f16b86-2a43-4faa-aa22-2158bc544ac8.mp4",
         env=None,
         history_frames=20,
         smoothing_window=15):

    results = []

    display_count = 0
    if env:
        os.environ["QT_QPA_PLATFORM"] = env

    pose_model = YOLO(pose_model)
    object_model = YOLO(object_detection_model)

    input_src = 0 if not cam_input else cam_input

    cap = cv2.VideoCapture(input_src)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    # Initialize previous frame and optical flow parameters
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to grab first frame")
        exit()

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Initialize keypoint history for each potential person
    # Using dictionaries with person ID as key
    max_history = history_frames
    wrist_history = {}  # For tracking hand movement
    ankle_history = {}  # For tracking walking
    hip_history = {}  # For tracking walking

    # For tracking person IDs across frames (simple tracking)
    last_person_positions = {}

    # Initialize action state manager for temporal smoothing
    action_manager = ActionStateManager(smoothing_window=smoothing_window)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video")
            break

        frame_result = []

        # Convert current frame to grayscale for optical flow
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        pose_results = pose_model(frame, verbose=False)
        object_results = object_model(frame, verbose=False)

        hands, phones, people = [], [], []
        keypoints_list = []

        for result in object_results:
            for box, label in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box)
                if int(label) == 0:
                    phones.append((x1, y1, x2, y2))
                elif int(label) == 1:
                    hands.append((x1, y1, x2, y2))

        # Person tracking and action detection
        current_person_positions = {}

        for result in pose_results:
            keypoints = result.keypoints.xy.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                people.append((x1, y1, x2, y2))
                keypoints_list.append([(int(x), int(y)) if x > 0 and y > 0 else None for x, y in keypoints[i]])

                # Use center of bounding box to track people
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                current_person_positions[i] = center

        # Map current detections to previous detections
        person_id_mapping = {}
        for curr_id, curr_pos in current_person_positions.items():
            best_match = None
            best_dist = float('inf')
            for prev_id, prev_pos in last_person_positions.items():
                dist = np.sqrt((curr_pos[0] - prev_pos[0]) ** 2 + (curr_pos[1] - prev_pos[1]) ** 2)
                if dist < best_dist and dist < 100:  # Threshold distance for same person
                    best_dist = dist
                    best_match = prev_id

            if best_match is not None:
                person_id_mapping[curr_id] = best_match
            else:
                # New person
                person_id_mapping[curr_id] = max(list(last_person_positions.keys()) + [-1]) + 1

        # Update last_person_positions for next frame
        last_person_positions = {person_id_mapping[curr_id]: pos
                                 for curr_id, pos in current_person_positions.items()}

        hand_to_person = map_hands_to_people(hands, people, keypoints_list)

        for i, (x1, y1, x2, y2) in enumerate(people):
            # Get stable person ID
            person_id = person_id_mapping[i]

            # Initialize history for new people
            if person_id not in wrist_history:
                wrist_history[person_id] = {'left': deque(maxlen=max_history),
                                            'right': deque(maxlen=max_history)}

            if person_id not in ankle_history:
                ankle_history[person_id] = {'left': deque(maxlen=max_history),
                                            'right': deque(maxlen=max_history)}

            if person_id not in hip_history:
                hip_history[person_id] = {'left': deque(maxlen=max_history),
                                          'right': deque(maxlen=max_history)}

            # Extract keypoints for this person
            left_wrist, left_elbow, left_shoulder = keypoints_list[i][9], keypoints_list[i][7], keypoints_list[i][5]
            right_wrist, right_elbow, right_shoulder = keypoints_list[i][10], keypoints_list[i][8], keypoints_list[i][6]
            left_hip, left_knee, left_ankle = keypoints_list[i][11], keypoints_list[i][13], keypoints_list[i][15]
            right_hip, right_knee, right_ankle = keypoints_list[i][12], keypoints_list[i][14], keypoints_list[i][16]

            # Update position history for motion analysis
            if left_wrist is not None:
                wrist_history[person_id]['left'].append(left_wrist)
            if right_wrist is not None:
                wrist_history[person_id]['right'].append(right_wrist)
            if left_ankle is not None:
                ankle_history[person_id]['left'].append(left_ankle)
            if right_ankle is not None:
                ankle_history[person_id]['right'].append(right_ankle)
            if left_hip is not None:
                hip_history[person_id]['left'].append(left_hip)
            if right_hip is not None:
                hip_history[person_id]['right'].append(right_hip)

            # Check phone interaction
            phone_interaction = False
            if i in hand_to_person:
                for hand in hand_to_person[i]:
                    if detect_interaction(hand, phones):
                        phone_interaction = True
                        break

            action_manager.update_state(person_id, "Holding Phone", phone_interaction)

            # Check hand raised and waving
            left_hand_raised = is_hand_raised(left_wrist, left_elbow, left_shoulder)
            right_hand_raised = is_hand_raised(right_wrist, right_elbow, right_shoulder)

            # Update hand raised state
            action_manager.update_state(person_id, "Hand Raised", left_hand_raised or right_hand_raised)

            # Check for waving with raised hand (only if hand is actually raised)
            left_waving = left_hand_raised and len(wrist_history[person_id]['left']) > 0 and \
                          is_hand_waving(wrist_history[person_id]['left'])
            right_waving = right_hand_raised and len(wrist_history[person_id]['right']) > 0 and \
                           is_hand_waving(wrist_history[person_id]['right'])

            # Update hand waving state
            action_manager.update_state(person_id, "Hand Waving", left_waving or right_waving)

            # Check standing and walking
            standing_left = is_standing(left_hip, left_knee, left_ankle)
            standing_right = is_standing(right_hip, right_knee, right_ankle)

            # Update standing state
            action_manager.update_state(person_id, "Standing", standing_left or standing_right)

            # Check for walking (only if person is standing)
            walking_detected = False
            if standing_left or standing_right:
                if len(ankle_history[person_id]['left']) > 0 and len(hip_history[person_id]['left']) > 0:
                    walking_detected = walking_detected or is_walking(
                        ankle_history[person_id]['left'], hip_history[person_id]['left'])

                if len(ankle_history[person_id]['right']) > 0 and len(hip_history[person_id]['right']) > 0:
                    walking_detected = walking_detected or is_walking(
                        ankle_history[person_id]['right'], hip_history[person_id]['right'])

            # Update walking state
            action_manager.update_state(person_id, "Walking", walking_detected)

            # Get temporally smoothed actions
            active_actions = action_manager.get_active_actions(person_id)

            # Handle contradictory states (prioritize more specific actions)
            if "Hand Waving" in active_actions and "Hand Raised" in active_actions:
                active_actions.remove("Hand Raised")  # Waving implies raised, so remove the less specific one

            if "Walking" in active_actions and "Standing" in active_actions:
                active_actions.remove("Standing")  # Walking implies standing, so remove the less specific one

            if active_actions:
                # Add confidence values to display
                display_actions = []
                for action in active_actions:
                    confidence = action_manager.states[person_id].get(action, 0) * 100
                    display_actions.append(f"{action}")

                frame_result.append({
                    "person_id": i,
                    "actions": display_actions
                })

                # Draw bounding box and actions
                cv2.putText(frame, ", ".join(display_actions), (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Draw trajectory of wrists for visualization
                if "Hand Waving" in active_actions or "Hand Raised" in active_actions:
                    if left_hand_raised and len(wrist_history[person_id]['left']) > 1:
                        points = np.array(list(wrist_history[person_id]['left']), dtype=np.int32)
                        for j in range(len(points) - 1):
                            color = (0, 255, 0) if "Hand Waving" in active_actions else (0, 0, 255)
                            cv2.line(frame, tuple(points[j]), tuple(points[j + 1]), color, 2)

                    if right_hand_raised and len(wrist_history[person_id]['right']) > 1:
                        points = np.array(list(wrist_history[person_id]['right']), dtype=np.int32)
                        for j in range(len(points) - 1):
                            color = (0, 255, 0) if "Hand Waving" in active_actions else (0, 0, 255)
                            cv2.line(frame, tuple(points[j]), tuple(points[j + 1]), color, 2)
            else:
                frame_result.append({
                    "person_id": i,
                    "actions": []
                })

        # Update previous frame
        prev_gray = gray.copy()

        # cv2.imshow("YOLO Pose & Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        display_count+=1
        results.append(frame_result)

    return results

if __name__ == "__main__":
    main()