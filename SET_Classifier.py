import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision


# import time
# from PIL import Image

# classifier is still not perfect because it is not trained on the full dataset
class MultiClassCNN(nn.Module):
    """
    CNN model for classifying images into 81 classes
    """
    def __init__(self):
        super(MultiClassCNN, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 20 * 32, 128)
        self.fc2 = nn.Linear(128, 82)
        self.relu = nn.ReLU()

    def forward(self, x):
        layer1 = self.relu(self.conv2(self.pool1(self.relu(self.conv1(x)))))
        layer2 = self.pool3(self.relu(self.conv3(self.pool2(layer1))))
        layer3 = self.fc2(self.relu(self.fc1(layer2.view(-1, 64 * 20 * 32))))
        return layer3


def classifier_model_train(model=None):
    """
    Trains the classifier model
    :param model: if not None, then the model is trained on the given model
    :return: None
    """
    dir_list = os.listdir("Train_Images/")
    if model is not None:
        classifier_model = model
    else:
        classifier_model = torch.load("model.pt")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier_model.parameters(), lr=1e-4)

    for epoch in range(10):
        cum_loss = 0
        for i in range(1, 82):
            for card in dir_list:
                sub = card[:card.find("_")].strip(".pn")
                if sub == str(i):
                    try:
                        X = torchvision.io.read_image("Train_Images/" + card).unsqueeze(dim=0).float()
                    except:
                        continue
                    classifier_model.train()
                    y = torch.tensor(i).unsqueeze(dim=0)
                    y_hat = classifier_model(X)
                    loss = loss_fn(y_hat, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # print("Epoch: ", epoch, " Image: ", i, " Loss: ", loss.item())
                    cum_loss += loss.item()
        print("Epoch: ", epoch, " Cumulative Loss: ", cum_loss)
    torch.save(classifier_model, "model.pt")


def classifier_model_test():
    """
    Tests the classifier model
    :return: None
    """
    classifier_model = torch.load("model.pt")
    labels = pd.read_csv("labels.csv", index_col=0)
    with torch.inference_mode():
        classifier_model.eval()
        for i in range(12):
            img = torchvision.io.read_image("Test_Images/" + str(i) + ".png").unsqueeze(dim=0).float()
            # show = Image.open("Test_Images/" + str(i) + ".png")
            # show.show()
            # time.sleep(2)
            output = classifier_model(img)
            result = torch.argmax(output, dim=1).item()
            print("Image: ", i, "  Result: ", result)


def main():
    condition = 2
    if condition == 0:
        # 0 to create a model from scratch
        classifier_model = MultiClassCNN()
        classifier_model_train(classifier_model)
    elif condition == 1:
        # 1 to further train a model
        classifier_model_train()
    # 2 to just test the model
    classifier_model_test()


if __name__ == '__main__':
    main()
