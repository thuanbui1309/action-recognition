def active_marking(actions):
    action_weights = {
        "Hand Raised": 1.0,
        "Hand Waving": -0.25,
        "Holding Phone": -1.0,
        "Walking": -0.5,
        "Standing": -0.25,
    }

    valid_actions = {a for a, w in action_weights.items() if w > 0}
    invalid_actions = {a for a, w in action_weights.items() if w < 0}

    frame_states = []

    for frame in actions:
        total = len(frame)
        if total == 0:
            frame_states.append("neutral")
            continue

        valid_score = sum(
            (sum(1 for person in frame if action in person['actions']) / total) * action_weights[action]
            for action in valid_actions
        )

        invalid_score = sum(
            (sum(1 for person in frame if action in person['actions']) / total) * abs(action_weights[action])
            for action in invalid_actions
        )

        score = valid_score - invalid_score

        if score > 0:
            frame_states.append("positive")
        elif score < 0:
            frame_states.append("negative")
        else:
            frame_states.append("neutral")

    # Compute class-level score
    n_pos = frame_states.count("positive")
    n_neg = frame_states.count("negative")
    n_neu = frame_states.count("neutral")
    n_total = len(frame_states)

    if n_total == 0:
        class_score = 0
    else:
        class_score = n_pos / (n_total - n_neu)

    print(class_score)

    # Determine overall label
    if class_score > 0.5:
        class_label = "Positive"
    elif class_score < 0.5:
        class_label = "Negative"

    return {
        "activeness_score": round(class_score, 4),
        "activeness_label": class_label,
        "total_frames": n_total,
        "positive_frames": n_pos,
        "negative_frames": n_neg,
        "neutral_frames": n_neu,
    }
