import pygame
import sys
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

STOP_PROGRAM = False
WHITE = (255,255,255)
BLACK = (0,0,0)
GREY = (120,120,125)
WIDTH = 300
HEIGHT = 200
pygame.init()
clock = pygame.time.Clock()
clock.tick(80)
font = pygame.font.Font(None,20)
number_font = pygame.font.Font(None,110)
with open('svc_model.pkl','rb') as file:
    svc = pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

def draw_circle(x,y,surface):
    pygame.draw.circle(surface,BLACK,(x,y),6)
    screen.blit(draw_surface,(0,0))
    pygame.display.flip()

def reset_draw_surface():
    draw_surface.fill(WHITE)
    screen.blit(draw_surface,(0,0))
    pygame.display.flip()

def draw_clear_button():
    pygame.draw.rect(output_surface, WHITE, (20, 10, 60, 20))
    button_text = font.render("CLEAR",True,BLACK)
    output_surface.blit(button_text,(28,14))
    screen.blit(output_surface,(100,200))
    pygame.display.flip()

def draw_predict_button():
    pygame.draw.rect(output_surface, WHITE, (20, 40, 60, 20))
    button_text = font.render("PREDICT", True, BLACK)
    output_surface.blit(button_text, (22, 44))
    screen.blit(output_surface, (100, 200))
    pygame.display.flip()

def draw_output(number):
    pygame.draw.rect(output_surface,WHITE,(20,80,60,100))
    output_text = number_font.render(f"{number[0]}",True,BLACK)
    output_surface.blit(output_text,(28,92))
    screen.blit(output_text,(100,200))
    pygame.display.flip()

def compute_image(img):
    img = np.rot90(img, k=1)
    img = np.flipud(img)
    img = np.where(img == 16777215, 255, img)
    img = np.where(img == 0, 1, img)
    img = np.where(img == 255, 0, img)
    img = img * 255
    img = Image.fromarray(img)
    img = img.resize((28, 28))
    img = np.array(img)
    img = np.where(img < 0, img * -1, img)
    img = np.where(img > 255, 255, img)
    img = img.reshape(784)
    img = scaler.transform([img])
    return img

screen = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("Digit Recognition")
draw_surface = pygame.Surface((200,200))
output_surface = pygame.Surface((100,200))
draw_surface.fill(WHITE)
output_surface.fill(GREY)
draw_clear_button()
draw_predict_button()
screen.blit(draw_surface,(0,0))
screen.blit(output_surface,(200,0))
pygame.display.flip()
while not(STOP_PROGRAM):
    for event in pygame.event.get():
        if pygame.mouse.get_pressed()[0] == True:
            x,y = pygame.mouse.get_pos()
            draw_circle(x,y,draw_surface)
        if 220<=pygame.mouse.get_pos()[0]<=280 and 10<=pygame.mouse.get_pos()[1]<=30 and event.type == pygame.MOUSEBUTTONDOWN:
            reset_draw_surface()
            output_surface.fill(GREY)
            draw_clear_button()
            draw_predict_button()
            screen.blit(output_surface,(200,0))
            pygame.display.flip()
        if 220<=pygame.mouse.get_pos()[0]<=280 and 40<=pygame.mouse.get_pos()[1]<=60 and event.type == pygame.MOUSEBUTTONDOWN:
            img = pygame.surfarray.array2d(draw_surface)
            img = compute_image(img)
            number = svc.predict(img)
            output_surface.fill(GREY)
            draw_clear_button()
            draw_predict_button()
            draw_output(number)
            screen.blit(output_surface,(200,0))
            pygame.display.flip()
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()