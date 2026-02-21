import pygame
import time
import random

pygame.init()
# Set up the display and colors
white = (255, 255, 255)
yellow = (255, 255, 102)
black = (0, 0, 0)
red = (213, 50, 80)
green = (0, 255, 0)
blue = (50, 153, 213)
dis_width = 600
dis_height = 400
dis = pygame.display.set_mode((dis_width,dis_width))
pygame.display.set_caption("Snake Game")
clock = pygame.time.Clock()
snake_block = 10
snake_speed = 5
font_style = pygame.font.SysFont("Arial",25)
score_font = pygame.font.SysFont("comicsansms",25)

def disp_score(score):
    value = score_font.render("Your score:" + str(score), True, yellow)
    dis.blit(value,[0,0])
def snake(snake_block, snake_list):
    for x in snake_list:
        pygame.draw.rect(dis,black,[x[0],x[1],snake_block,snake_block])
def message(msg,color):
    m = font_style.render(msg,True,color)
    dis.blit(m,[dis_width/2,dis_height/2])
def GameLoop():
    game_over, game_close = False, False
    