import pandas as pd
import os
import torch
import torchvision
from PIL import Image, ImageDraw
from SET_Classifier import MultiClassCNN

# global variables
condition = False
verbose = True
reset = True

def split(img):
    """
    Turns a single image of current game state into 12 images of individual images for classification
    :param img:
    :return: None
    """
    images = []
    w = img.size[0]
    h = img.size[1]
    for i in range(12):
        x = (i % 4) * w / 4
        y = (i // 4) * h / 3
        img_i = img.crop((x, y, x + w / 4, y + h / 3))
        img_i = img_i.resize((258, 167))
        images.append(img_i)
    for i in range(12):
        images[i].save("Test_Images/" + str(i) + ".png")


def classify():
    """
    Classifies each image in Test_Images and returns the results
    :return output: list of lists of card features
    """
    number_model = torch.load("model0.pt")
    color_model = torch.load("model1.pt")
    shape_model = torch.load("model2.pt")
    fill_model = torch.load("model3.pt")
    output = []
    for i in range(12):
        img = torchvision.io.read_image("Test_Images/" + str(i) + ".png").unsqueeze(dim=0).float()
        if img.shape[1] == 3:
            img = torch.cat((img, torch.zeros(1, 1, 167, 258)), dim=1)
        
        number = number_model(img).argmax(dim=1).item()
        color = color_model(img).argmax(dim=1).item()     
        shape = shape_model(img).argmax(dim=1).item()
        fill = fill_model(img).argmax(dim=1).item()        
        label = [number, color, shape, fill]
        if verbose:
            if i == 0:
                print("Classifying: ")
            print("Image: ", i, "  Result: ", label)
        output.append(label)
    return output


def find(results):
    """
    Finds all sets in the current game state
    :param results: a list of lists of card features
    :return sets: list of sets
    """
    if verbose:
        print("Finding sets: ")
    sets = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            for k in range(j + 1, len(results)):
                if is_set(results[i], results[j], results[k]):
                    if verbose:
                        print("Set found: |", results[i], results[j], results[k], "|")
                    sets.append([i, j, k])
    return sets


def is_set(card1, card2, card3):
    """
    Determines if the three cards are a set
    :param card1:
    :param card2:
    :param card3:
    :return boolean: True if the three cards are a set, False otherwise
    """
    for i in range(len(card1)):
        if card1[i] == card2[i] == card3[i]:
            continue
        elif card1[i] != card2[i] != card3[i] and card1[i] != card3[i]:
            continue
        else:
            return False
    return True


def show(output, image_name):
    """
    Displays the current game state with the sets highlighted
    :param image_name:
    :param output:
    :return: None
    """
    color_dict = {0: "red", 1: "green", 2: "purple", 3: "yellow",
                  4: "blue", 5: "orange", 6: "pink", 7: "black",
                  8: "white", 9: "grey", 10: "brown", 11: "cyan"}
    # building the image from composites
    img = Image.new("RGB", (258 * 4, 167 * 3))
    for i in range(12):
        img_i = Image.open("Test_Images/" + str(i) + ".png")
        x = (i % 4) * 258
        y = (i // 4) * 167
        img.paste(img_i, (x, y))
        if reset:
            os.remove("Test_Images/" + str(i) + ".png")
    for SET in output:
        k = 0
        for i in range(3):
            for j in range(i + 1, 3):
                k += 1
                if k > 2:
                    continue
                x1 = (SET[i] % 4) * 258 + 129
                x2 = (SET[j] % 4) * 258 + 129
                y1 = (SET[i] // 4) * 167 + 83
                y2 = (SET[j] // 4) * 167 + 83
                draw = ImageDraw.Draw(img)
                draw.line((x1, y1, x2, y2), fill=color_dict[output.index(SET)], width=5)
    if reset:
        img.save(image_name[:image_name.find("_")] + "_solved.png")
    else:
        img.show()


def main(image):
    if verbose:
        print("Starting", '-' * 41)
    # check if folder test images is empty
    if len(os.listdir("Test_Images")) == 0:
        img1 = Image.open(image)
        split(img1)
    results = classify()
    output = find(results)
    show(output, image)
    if verbose:
        print("Done", '-' * 45)


if __name__ == "__main__":
    condition = True
    main("feb27_game.png")
