import cv2
import numpy as np
import torch

from evo_science.entities.utils.nms import NonMaxSuppression


@torch.no_grad()
def demo(input_size: int, model):
    camera = cv2.VideoCapture(0)
    # Check if camera opened successfully
    if not camera.isOpened():
        print("Error opening video stream or file")
    # Read until video is completed
    nms = NonMaxSuppression(conf_threshold=0.25, iou_threshold=0.7)
    while camera.isOpened():
        # Capture frame-by-frame
        success, frame = camera.read()
        if success:
            image = frame.copy()
            shape = image.shape[:2]

            r = input_size / max(shape[0], shape[1])
            if r != 1:
                resample = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
                image = cv2.resize(image, dsize=(int(shape[1] * r), int(shape[0] * r)), interpolation=resample)
            height, width = image.shape[:2]

            # Scale ratio (new / old)
            r = min(1.0, input_size / height, input_size / width)

            # Compute padding
            pad = int(round(width * r)), int(round(height * r))
            w = np.mod((input_size - pad[0]), 32) / 2
            h = np.mod((input_size - pad[1]), 32) / 2

            if (width, height) != pad:  # resize
                image = cv2.resize(image, pad, interpolation=cv2.INTER_LINEAR)
            top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
            left, right = int(round(w - 0.1)), int(round(w + 0.1))
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)  # add border

            # Convert HWC to CHW, BGR to RGB
            x = image.transpose((2, 0, 1))[::-1]
            x = np.ascontiguousarray(x)
            x = torch.from_numpy(x)
            x = x.unsqueeze(dim=0)
            x = x.cuda()
            x = x.half()
            x = x / 255
            # Inference
            outputs = model(x)
            # NMS
            outputs = nms(outputs=outputs)
            for output in outputs:
                output[:, [0, 2]] -= w  # x padding
                output[:, [1, 3]] -= h  # y padding
                output[:, :4] /= min(height / shape[0], width / shape[1])

                output[:, 0].clamp_(0, shape[1])  # x1
                output[:, 1].clamp_(0, shape[0])  # y1
                output[:, 2].clamp_(0, shape[1])  # x2
                output[:, 3].clamp_(0, shape[0])  # y2

                for box in output:
                    box = box.cpu().numpy()
                    x1, y1, x2, y2, score, index = box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            cv2.imshow("Frame", frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
        # Break the loop
        else:
            break
    # When everything done, release the video capture object
    camera.release()

    # Closes all the frames
    cv2.destroyAllWindows()
