import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision

# global variables
verbose = False

# Classification reference
# number = {1, 2, 3}
#          [0, 1, 2]
# color = {red, purple,  green}
#         [0  ,     1,      2]
# shape = {squiggle, diamond, oval}
#         [0       ,       1,    2]
# fill = {full, half, empty}
#        [0   ,    1,    2]
# visit https://www.setgame.com/set/puzzle to scrape clear images of all 81 cards

class MultiClassCNN(nn.Module):
    """
    CNN model for classifying images into 3 classes
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
        self.fc2 = nn.Linear(128, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        layer1 = self.relu(self.conv2(self.pool1(self.relu(self.conv1(x)))))
        layer2 = self.pool3(self.relu(self.conv3(self.pool2(layer1))))
        layer3 = self.fc2(self.relu(self.fc1(layer2.view(-1, 64 * 20 * 32))))
        return layer3


def classifier_model_train(model, k, learning_rate=1e-4):
    """
    Trains the classifier model
    :param model: if not None, then the model is trained on the given model
    :return: None
    """
    dir_list = os.listdir("Train_Images/")
    labels = pd.read_csv("labels.csv", index_col=0)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    losses = [0 for i in range(101)]
    for epoch in range(1, 101):
        cum_loss = 0
        for card in dir_list:
            name = card.split("_")[0].strip(".png")
            X = torchvision.io.read_image("Train_Images/" + card).unsqueeze(dim=0).float()
            model.train()
            y = labels.iloc[int(name) - 1, k].item()
            y = torch.tensor([y])
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if verbose:
                print("Epoch: ", epoch, " Image: ", card, " Name:", name, "Guess: ", 
                      y_hat.argmax(dim=1).item(), " Actual: ", y)
            cum_loss += loss.item()
        losses[epoch] = cum_loss                
        print("Epoch: ", epoch, " Cumulative Loss: ", cum_loss)
        # Early stopping methods to prevent overfitting, however since this classification should be quite 
        # accurate given the simplicity of the task, overfitting thresholds are set to be quite high
        if cum_loss < 0.05:
            break
    name = "model" + str(k) + ".pt"
    torch.save(model, name)

def number(Improve):
    print("Number")
    if not Improve:
        number_model = MultiClassCNN()
        classifier_model_train(number_model, 0)
    else:
        number_model = torch.load("model0.pt")
        classifier_model_train(number_model, 0)
        
def color(Improve):
    print("Color")
    if not Improve:
        color_model = MultiClassCNN()
        classifier_model_train(color_model, 1)
    else:
        color_model = torch.load("model1.pt")
        classifier_model_train(color_model, 1)

def shape(Improve):
    print("Shape")
    if not Improve:
        shape_model = MultiClassCNN()
        classifier_model_train(shape_model, 2)
    else:
        shape_model = torch.load("model2.pt")
        classifier_model_train(shape_model, 2)
        
def fill(Improve):
    print("Fill")
    if not Improve:
        fill_model = MultiClassCNN()
        classifier_model_train(fill_model, 3, 1e-6)
    else:
        fill_model = torch.load("model3.pt")
        classifier_model_train(fill_model, 3, 1e-6)
    
    
def main():
    Improve = False  # Improve determines if new models should be trained or if the existing models should be improved
    number(Improve)
    color(Improve)
    shape(Improve)
    fill(Improve)

if __name__ == '__main__':
    main()
