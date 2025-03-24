import cv2
import numpy as np
import os
import argparse
import tensorflow as tf
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model

# Top-k predictions
def get_top_k(probs, labels, k=3):
    top_predictions = tf.argsort(probs, axis=-1, direction="DESCENDING")[0, :k]
    top_labels = labels[top_predictions.numpy()]
    top_probs = tf.gather(probs[0], top_predictions).numpy()
    return [(label, prob) for label, prob in zip(top_labels, top_probs)]

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Real-time Action Recognition with MoviNet")
    parser.add_argument("--input", type=str, default=0, help="Video file path or '0' for webcam")
    parser.add_argument("--augmented", type=str, default=False, help="Model trained with augmented data or not")
    parser.add_argument("--labels", type=str, default="models/movinet/labels.npy", help="labels_old annotation")
    parser.add_argument("--env", type=str, default=None, help="QPA_Platform")
    args = parser.parse_args()

    # Run this if using Linux Wayland
    if args.env:
        os.environ["QT_QPA_PLATFORM"] = args.env

    # Load labels_old
    labels = np.load(args.labels, allow_pickle=True)

    # Create backbone
    backbone = movinet.Movinet(
        model_id="a0",
        causal=True,
        conv_type='2plus1d',
        se_type='2plus3d',
        activation='hard_swish',
        gating_activation='hard_sigmoid',
        use_positional_encoding=False,
        use_external_states=True,
    )

    # Load model
    model = movinet_model.MovinetClassifier(
        backbone,
        num_classes=len(labels),
        output_states=True)

    # Init movinet state
    input_shape = (1, 1, 172, 172, 3)  # (batch_size, frames, height, width, channels)
    states = model.init_states(input_shape)

    # Load weights from the checkpoint to the rebuilt model
    if args.augmented:
        weights = "models/movinet/a0/a0_augmented.weights.h5"
    else:
        weights = "models/movinet/a0/a0.weights.h5"
    model.load_weights(weights)

    print(args.input)
    # Input source
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        exit()

    # Original video resolution
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # New video resolution
    display_width = 640
    display_height = int(original_height * (display_width / original_width))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to 172x172
        img = cv2.resize(frame, (172, 172))
        img = img.astype(np.float32) / 255.0
        img_tensor = tf.convert_to_tensor(img)
        img_tensor = tf.expand_dims(img_tensor, axis=0)
        img_tensor = tf.expand_dims(img_tensor, axis=0)

        # Action prediction
        logits, states = model({**states, "image": img_tensor})
        probs = tf.nn.softmax(logits, axis=-1)
        predictions = get_top_k(probs, labels, k=3)

        frame = cv2.resize(frame, (display_width, display_height))

        # Display top-3 predictions
        for i, pred in enumerate(predictions):
            text = f"Top {i + 1}: {pred[0]}"
            cv2.putText(frame, text, (10, 50 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Action Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
