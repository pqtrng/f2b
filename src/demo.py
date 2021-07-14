import click
import cv2
import dlib
import helper
import numpy as np

from src.config import Config


@click.command()
@click.argument("training_type")
@click.argument("dataset")
@click.argument("output_network_type")
def demo(training_type, dataset, output_network_type):
    """Run local demo.

    Args:
        training_type (str): type of training
        dataset (str): name of dataset
        output_network_type (str): type of output network
    """

    # Load model
    model, _ = helper.get_trained_model(
        training_type=training_type,
        dataset=dataset,
        output_network_type=output_network_type,
    )

    detector = dlib.get_frontal_face_detector()

    for img in helper.yield_images_from_camera():
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)

        detected = detector(input_img, 1)

        faces = np.empty(
            (len(detected), Config.image_default_size, Config.image_default_size, 3)
        )

        if len(detected) > 0:

            # Crop face
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = (
                    d.left(),
                    d.top(),
                    d.right() + 1,
                    d.bottom() + 1,
                    d.width(),
                    d.height(),
                )
                xw1 = max(int(x1 - Config.margin * w), 0)
                yw1 = max(int(y1 - Config.margin * h), 0)
                xw2 = min(int(x2 + Config.margin * w), img_w - 1)
                yw2 = min(int(y2 + Config.margin * h), img_h - 1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                faces[i, :, :, :] = cv2.resize(
                    img[yw1 : yw2 + 1, xw1 : xw2 + 1, :],
                    (Config.image_default_size, Config.image_default_size),
                )

            predictions = model.predict(faces)

            for i, d in enumerate(detected):
                label = f"BMI: {predictions[i][0]:.2f}"
                helper.draw_label(image=img, point=(d.left(), d.top()), label=label)

        cv2.imshow("result", img)
        key = cv2.waitKey(30)

        if key == 27:  # ESC
            break


if __name__ == "__main__":
    demo()
