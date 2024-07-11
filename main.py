import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import os
import numpy as np
from keras.models import load_model
import imutils
import psutil
import time

def weapon_detect(path):
    # Read the model and configuration
    net = cv2.dnn.readNet("D:\Omkar\Fourth Year\Major Project 2\Main\Intelligent_video_surveillance-main\Intelligent_video_surveillance-main\yolov3_training_2000.weights", "D:\Omkar\Fourth Year\Major Project 2\Main\Intelligent_video_surveillance-main\Intelligent_video_surveillance-main\yolov3_testing.cfg")

    # Set preferable backend and target to GPU
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    classes = ["Weapon"]
    output_layer_names = net.getUnconnectedOutLayersNames()
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    cap = cv2.VideoCapture(path)
    skip_frames = 15  # Adjust the number of frames to skip

    frame_count = 0
    temp = 0  # Initialize temp to 0

    # Create the directory to save frames if it doesn't exist
    extracted_frames_dir = "D:/Omkar/Fourth Year/Major Project 2/Main/Intelligent_video_surveillance-main/Intelligent_video_surveillance-main/Extracted frames"
    if not os.path.exists(extracted_frames_dir):
        os.makedirs(extracted_frames_dir)

    # Define the constant filename
    frame_filename = os.path.join(extracted_frames_dir, "detected_frame.jpg")

    while True:
        ret, img = cap.read()
        if not ret:
            print("Failed to read a frame")
            break
        frame_count += 1
        if frame_count % skip_frames != 0:
            continue  # Skip frames

        height, width, channels = img.shape

        # Resize input image
        blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layer_names)

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        if len(indexes) > 0:
            print("Weapon detected in frame")
            # Overwrite the frame with weapon detection
            cv2.imwrite(frame_filename, img)
            temp = 1  # Set temp to 1 if weapon is detected
            break  # Exit the loop if weapon is detected

        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

        # cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    return temp  # Return the value of temp

def abnormal_action(path):
    def mean_squared_loss(x1, x2):
        difference = x1 - x2
        a, b, c, d, e = difference.shape
        n_samples = a * b * c * d * e
        sq_difference = difference**2
        Sum = sq_difference.sum()
        distance = np.sqrt(Sum)
        mean_distance = distance / n_samples
        return mean_distance

    # Load the model
    model = load_model("D:/Omkar/Fourth Year/Major Project 2/Main/Intelligent_video_surveillance-main/Intelligent_video_surveillance-main/saved_model.keras")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error: Unable to open video capture.")
        return

    # Variables for resource monitoring
    cpu_percent_list = []
    memory_percent_list = []
    start_time = time.time()

    # Create the directory to save frames if it doesn't exist
    extracted_frames_dir = "D:/Omkar/Fourth Year/Major Project 2/Main/Intelligent_video_surveillance-main/Intelligent_video_surveillance-main/Extracted frames"
    if not os.path.exists(extracted_frames_dir):
        os.makedirs(extracted_frames_dir)

    # Define the constant filename
    frame_filename = os.path.join(extracted_frames_dir, "detected_frame.jpg")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error while reading frames.")
            break

        imagedump = []
        for i in range(10):
            ret, frame = cap.read()
            if not ret:
                print("End of video or error while reading frames.")
                break

            image = imutils.resize(frame, width=700, height=600)
            frame = cv2.resize(frame, (227, 227), interpolation=cv2.INTER_AREA)
            gray = 0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] + 0.1140 * frame[:, :, 2]
            gray = (gray - gray.mean()) / gray.std()
            gray = np.clip(gray, 0, 1)
            imagedump.append(gray)

        if not ret:
            break

        imagedump = np.array(imagedump)
        imagedump.resize(227, 227, 10)
        imagedump = np.expand_dims(imagedump, axis=0)
        imagedump = np.expand_dims(imagedump, axis=4)

        output = model.predict(imagedump)

        loss = mean_squared_loss(imagedump, output)
        print(loss)

        if loss > 0.000335:
            print('Abnormal Event Detected')
            cv2.imwrite(frame_filename, image)
            cv2.putText(image, "Abnormal Event", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            # Overwrite the frame with abnormal event detection
            break

        # cv2.imshow("video", image)
        
        # Collect CPU and memory utilization
        cpu_percent_list.append(psutil.cpu_percent())
        memory_percent_list.append(psutil.virtual_memory().percent)

        print("Current Memory Usage:", (psutil.virtual_memory().percent)/2, "%")

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time

    # Print average CPU and memory utilization
    if cpu_percent_list:
        average_cpu_percent = sum(cpu_percent_list) / len(cpu_percent_list)
    else:
        average_cpu_percent = 0
    if memory_percent_list:
        average_memory_percent = sum(memory_percent_list) / len(memory_percent_list)
    else:
        average_memory_percent = 0
    print("Average CPU Utilization:", average_cpu_percent, "%")
    print("Average Memory Utilization:", average_memory_percent/2, "%")
    print("Execution Time:", execution_time, "seconds")

    cap.release()
    cv2.destroyAllWindows()


# Function to browse and select a video file
def browse_file():
    filename = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
    entry_path.delete(0, tk.END)
    entry_path.insert(0, filename)

# Function to detect weapon and abnormal action
def detect_action():
    path = entry_path.get()
    if path:
        val = weapon_detect(path)
        if val == 0:
            abnormal_action(path)
            print("Weapon not detected but abnormal action detected")
            status_label.config(text="Weapon not detected but abnormal action detected", fg="red")
            # Display the extracted frame in the UI
            img = Image.open("D:/Omkar/Fourth Year/Major Project 2/Main/Intelligent_video_surveillance-main/Intelligent_video_surveillance-main/Extracted frames/detected_frame.jpg")
            img.thumbnail((300, 300))
            img = ImageTk.PhotoImage(img)
            label_image.config(image=img)
            label_image.image = img
        else:
            print("Weapon detected")
            status_label.config(text="Weapon detected", fg="red")
            # Display the extracted frame in the UI
            img = Image.open("D:/Omkar/Fourth Year/Major Project 2/Main/Intelligent_video_surveillance-main/Intelligent_video_surveillance-main/Extracted frames/detected_frame.jpg")
            img.thumbnail((300, 300))
            img = ImageTk.PhotoImage(img)
            label_image.config(image=img)
            label_image.image = img

# Create the main window
root = tk.Tk()
root.title("Video Survillance Detection")
root.geometry("500x450")

# Create and place input fields and buttons
frame_input = tk.Frame(root, bg="#f0f0f0", padx=10, pady=10)
frame_input.pack(pady=10)

label_path = tk.Label(frame_input, text="Select Video:", bg="#f0f0f0")
label_path.grid(row=0, column=0, padx=5, pady=5)

entry_path = tk.Entry(frame_input, width=30)
entry_path.grid(row=0, column=1, padx=5, pady=5)

button_browse = tk.Button(frame_input, text="Browse", command=browse_file)
button_browse.grid(row=0, column=2, padx=5, pady=5)

button_detect = tk.Button(root, text="Detect", command=detect_action, bg="#4CAF50", fg="white")
button_detect.pack(pady=10)

# Create and place output field for extracted frame and status message
frame_output = tk.Frame(root, bg="#f0f0f0")
frame_output.pack()

label_image = tk.Label(frame_output, bg="#f0f0f0")
label_image.pack(pady=5)

status_label = tk.Label(frame_output, bg="#f0f0f0", fg="black")
status_label.pack(pady=5)

root.mainloop()