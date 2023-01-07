from PIL import Image, ImageDraw,ImageFont # Load Images & Draw Rectangles.
import torchvision.transforms as transforms # Tensor Transformation.
import torchvision.models as models # ResNet-18 PyTorch Model.
from torch import nn # Neural Network Layers
import torch # YOLO v5 Model
import time # Benchmark extraction
import cv2
import numpy as np
from deepface import DeepFace

def main(folder_path, img, transform, FaceDetector, FaceClassifier):

    def extractFace(IMG, FaceDetector, threshold=0.50, returnFace=True):
        extractedFaces = []
        extractedBoxes = []
        FaceDetections = FaceDetector(IMG).pandas().xyxy[0]
        for detection in FaceDetections.values:
            xmin, ymin, xmax, ymax, confidence = detection[:5]
            if confidence >= threshold:
                bb = [(xmin, ymin), (xmax, ymax)]
                if returnFace:
                    w, h = xmax - xmin, ymax - ymin
                    currentFace = IMG.crop((xmin, ymin, w+xmin, h+ymin))
                    extractedFaces.append(currentFace)
                extractedBoxes.append(bb)

        return extractedFaces, extractedBoxes

    def readImage(IMG):
        IMG = IMG.convert('RGB')
        IMG = IMG.resize((200, 200))
        tensorIMG = transform(IMG).unsqueeze(0)
        return tensorIMG

    def extractInfo(MyModel, tensorIMG):
        tensorIMG = tensorIMG.to(runOn)
        tensorLabels = MyModel(tensorIMG)[0]

        Age = torch.argmax(tensorLabels[:Classes])
        Gender = int(torch.argmax(tensorLabels[Classes:]))
        Gender = 'Male' if Gender == 0 else 'Female'

        C1 = float(torch.max(tensorLabels[:Classes]))
        C2 = float(torch.max(tensorLabels[Classes:]))

        return Groups[Age], Gender, [round(C1, 3), round(C2, 3)]

    def returnAnalysis(img):
        t0 = time.time()

        IMG = Image.fromarray(img)
        IMG.save("example.jpg") # INI
        IMG_ = ImageDraw.Draw(IMG)
        faces, bbs = extractFace(IMG, FaceDetector, 0.7)
        tt1 = time.time() - t0

        tt2 = 0
        for face, bb in zip(faces, bbs):
            IMG_.rectangle(bb, outline ="Red", width=4)
            tensorIMG = readImage(face)
            t0 = time.time()
            Age, Gender, C = extractInfo(FaceClassifier, tensorIMG)
            obj = DeepFace.analyze(img_path = "example.jpg", 
                                     actions = ['emotion'],enforce_detection=False)
            Emoji = obj["dominant_emotion"]
            tt = time.time() - t0
            tt2 += tt
            textBox = f'{Age} {Gender} {Emoji}'
            Text = ImageDraw.Draw(IMG)
            font = ImageFont.truetype(folder_path + 'FontsFree-Net-arial-bold.ttf', 24) 
            Text.text((bb[0][0], bb[0][1]-25), 
                        textBox, 
                        font=font,
                        fill=(255, 0, 0))
        return IMG, tt1+tt2

    IMG, tt = returnAnalysis(img)
    print(f"Extraction Time: {round(tt, 3)}")
    return IMG

if __name__ == '__main__':
    torch.hub.set_dir(r'C:\Users\bryan\Downloads\Project\model')

    Classes = 9
    Groups = ['00-10', '11-20', '21-30', 
            '31-40', '41-50', '51-60', 
            '61-70', '71-80', '81-90']
    runOn = "cpu"

    def returnDetector():
        FaceDetector = torch.hub.load('ultralytics/yolov5', 
                                'yolov5s', 
                                folder_path + 'Best.onnx', 
                                _verbose=False)
        FaceDetector.eval()
        FaceDetector.to(runOn)
        return FaceDetector

    def returnClassifier():
        FaceClassifier = models.resnet18(pretrained=True)
        FaceClassifier.fc = nn.Linear(512, Classes+2)
        FaceClassifier = nn.Sequential(FaceClassifier, nn.Sigmoid())

        FaceClassifier.load_state_dict(torch.load(ClassificationModel2, map_location=torch.device('cpu')))
        FaceClassifier.eval()
        FaceClassifier.to(runOn)
        return FaceClassifier

    folder_path = 'C:/Users/bryan/Downloads/Project/'
    ClassificationModel2 = folder_path + 'ResNet-18 Age 0.60 + Gender 93.pt'

    transform = transforms.Compose([transforms.ToTensor()])
    FaceDetector = returnDetector()
    # torch.save(FaceDetector.state_dict(), "Detector.pt")
    FaceClassifier = returnClassifier()

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280) # width
    cap.set(4, 720) # height

    while True:
        ret, img = cap.read()
        img = main(folder_path,img, transform, FaceDetector, FaceClassifier)
        img = np.array(img) 

        # img.show()

        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):   
            break

    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()