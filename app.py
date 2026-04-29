from flask import Flask, request, render_template
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model (downloads automatically first time)
model = YOLO("yolov8n.pt")

def predict_image(img):
    results = model(img)

    names = results[0].names
    boxes = results[0].boxes

    detected = []

    if boxes is not None:
        for cls in boxes.cls:
            label = names[int(cls)]
            detected.append(label)

    if not detected:
        return "No animal detected"

    return ", ".join(set(detected))


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]

        if file:
            img = Image.open(file)

            result = predict_image(img)

            return render_template("index.html", result=result)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)