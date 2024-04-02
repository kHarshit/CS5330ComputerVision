"""
@author: Khushi Neema, Harshit Kumar
@brief: Predict on new handwritten digit images using a trained MNIST model on a live webcam feed.
"""

import cv2
import torch
import torchvision.transforms as transforms
from models.MyNetwork import MyNetwork

# load the trained model
model = MyNetwork()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply the transformations
    tensor = transform(gray)
    tensor = tensor.unsqueeze(0)

    # Make a prediction
    output = model(tensor)
    _, predicted = torch.max(output.data, 1)

    # Display the resulting frame
    cv2.putText(frame, 'Detected Digit: ' + str(predicted.item()), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.imshow('Live Digit Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
