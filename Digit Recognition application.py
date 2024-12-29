import pygame
import sys
from pygame.locals import *
import numpy as np
import cv2
from keras.models import load_model

# Initialize pygame
pygame.init()

# Window settings
WINDOWSIZEX = 640
WINDOWSIZEY = 480
Boundary = 5

# Colors
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)

# Model and labels
model = load_model("best_model.h5.keras")
labels = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
          6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}

# Initialize pygame display
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
pygame.display.set_caption("Digit Board")
DISPLAYSURF.fill(black)

# Font for displaying text
font = pygame.font.Font(None, 36)

# Variables
iswriting = False
Number_xcord = []
Number_ycord = []
imagesave = False
predict = True
img_count = 1

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, white, (xcord, ycord), 4, 0)
            Number_xcord.append(xcord)
            Number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            if Number_xcord and Number_ycord:
                # Calculate bounding box
                rect_min_x, rect_max_x = max(min(Number_xcord) - Boundary, 0), min(max(Number_xcord) + Boundary, WINDOWSIZEX)
                rect_min_y, rect_max_y = max(min(Number_ycord) - Boundary, 0), min(max(Number_ycord) + Boundary, WINDOWSIZEY)

                Number_xcord = []
                Number_ycord = []

                # Extract the drawn region
                img_arr = pygame.surfarray.array3d(DISPLAYSURF)[rect_min_x:rect_max_x, rect_min_y:rect_max_y]
                img_arr = np.transpose(img_arr, (1, 0, 2))  # Correct axes for OpenCV (height, width, channels)

                if img_arr.size == 0:
                    print("Empty image array!")
                    continue

                 # Convert to grayscale
                img_arr = cv2.resize(img_arr, (32, 32))  # Resize to 32x32
                img_arr = img_arr / 255.0  # Normalize to [0, 1]
               
                img_arr = np.expand_dims(img_arr, axis=0)  # Add batch dimension

                if imagesave:
                    cv2.imwrite(f"image_{img_count}.png", img_arr.squeeze() * 255)  # Save normalized image
                    img_count += 1

                if predict:
                    # Predict the label
                    if img_arr.shape == (1, 32, 32, 3):
                        prediction = model.predict(img_arr)
                        label = str(labels[np.argmax(prediction)])

                        # Display the label
                        text_surface = font.render(label, True, red, white)
                        text_rect = text_surface.get_rect()
                        text_rect.left, text_rect.top = rect_min_x, rect_min_y
                        DISPLAYSURF.blit(text_surface, text_rect)
                    else:
                        print("Image shape mismatch. Expected (1, 32, 32, 3).")

        if event.type == KEYDOWN:
            if event.unicode == 'n':
                DISPLAYSURF.fill(black)

    pygame.display.update()
