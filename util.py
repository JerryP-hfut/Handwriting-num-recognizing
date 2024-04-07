import torch
import torch.nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

from model import lenet

model = lenet.Lenet()
model.load_state_dict(torch.load("saved_model.pth"))
model.eval()

test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

def predict_image(image_path, model):
    print("recognizing...")
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # 添加一个维度作为批处理
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return predicted.item()

while True:
    order = input("Please enter the MODE(predict/evaluate):")
    if order == "predict" or order == "evaluate":
        break
    else:
        print("Unrecognized order.PLEASE REINPUT!")
        
        
if order == "evaluate":
    print("evaluating...")
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Test Accuracy: {:.2f}%'.format(100 * correct / total))
else:
    image_path = 'E:\BianCheng\MACHINE_LEARNING\input\\test_7.png'
    try:
        predicted_label = predict_image(image_path, model)
        print("I think this figure is:", predicted_label)
    except Exception as e:
        print("Something went wrong:", e)
