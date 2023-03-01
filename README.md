# SET
## Intro
This project is a potentially naive attempt to create a program that will take an image of a given instantiation of the game SET and output an image of the solved game.
This is completed by getting a webcam capture in SET_Continuous, feeding that image to SET_Solver which will utilize the CNN from SET_Classifier to classify equal partitions of the initial image, find the sets possible given the predicted facets of the card, and convert the indexes of the sets into locations in a grid of images that are marked up, combined, and returned as output.
## The Game
Visit https://en.wikipedia.org/wiki/Set_(card_game) for more info. Essentially 12 cards are lined up in a four by 3 grid and players attempt to find patterns among three cards in which each card is all the same or all different for each of the four features. [number, symbol, shading, color]
## Example
<img src= "https://user-images.githubusercontent.com/93213444/222033025-1065399c-6c36-43cd-93b5-a41a1f6432c1.png" width="512" height="250" />
--------------------------------------------------------------------------------

This image is split into 12 cards, the program is run using these new cards as input, and in turn the below image is created where every seperately colored line represents connections between the three cards that create a set.

<img src= "https://user-images.githubusercontent.com/93213444/222033120-2553f7d0-659d-4937-8b79-50e29cc64428.png" width="512" height="250" />

