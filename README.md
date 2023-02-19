# SET
## Intro
This project is a potentially naive attempt to create a program that will take an image of a given instantiation of the game SET and output an image of the solved game.
This is completed by getting a webcam capture in SET_Continuous, feeding that image to SET_Solver which will utilize the CNN from SET_Classifier to classify equal partitions of the initial image, translate those classifications into a list, find the sets possible, and convert the indexes of the sets into locations in a grid of images that are marked up, combined, and returned as output.
## The Game
Visit https://en.wikipedia.org/wiki/Set_(card_game) for more info. Essentially 12 cards are lined up in a four by 3 grid and players attempt to find patterns among three cards in which each card is all the same or all different for each of the four features. [number, symbol, shading, color]
## Example
<img src= "https://user-images.githubusercontent.com/93213444/219936687-29db76bf-fcf1-4dbf-8026-37876a8bf7db.png" width="512" height="334" />
--------------------------------------------------------------------------------

This image is split into 12 cards, the program is run using these new cards as input, and in turn the below image is created where every seperately colored line represents connections between the three cards that create a set.

<img src= "https://user-images.githubusercontent.com/93213444/219936704-74efd67e-19f8-44f4-83ad-cb894ce59558.png" width="512" height="334" />
