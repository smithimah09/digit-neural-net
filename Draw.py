import pygame
import pygame_widgets
from pygame_widgets.button import Button
import math
from NeuralNet import MLP
import numpy as np

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BACKGROUND = (100, 100, 255)
UPDATE = 0.01
SLEEP = 0.5

pygame.init()
pygame.font.init()

label_font = pygame.font.SysFont('Comic Sans MS', 25)
prob_font = pygame.font.SysFont('Comic Sans MS', 18)

clock = pygame.time.Clock()
width, height = 720, 500
screen = pygame.display.set_mode((width, height))
screen.fill(BACKGROUND)

drawing_rect = pygame.Rect(40, 40, 380, 380)
drawing_pane = pygame.Surface((380, 380))
drawing_pane.fill(WHITE)

pixel_pane = pygame.Surface((380, 380))
pixel_pane.fill(WHITE)

predictions = pygame.Surface((220, 380))
predictions.fill(WHITE)

pixelated = False

reset = Button(
    screen, 60, 430, 160, 40,
    text='Clear',
    fontSize=25,
    inactiveColour=(255, 255, 255),
    hoverColour=(200, 200, 200),
    pressedColour=(100, 100, 100),
    radius=5,
    onClick=lambda: clear()
)

toggle = Button(
    screen, 240, 430, 160, 40,
    text='Pixelated',
    fontSize=25,
    inactiveColour=(255, 255, 255),
    hoverColour=(200, 200, 200),
    pressedColour=(100, 100, 100),
    radius=5,
    onClick=lambda: toggle_view()
)

net = MLP.load("Models/0.9811.pickle")

def main():
    running = True
    dt = 0
    stop_time = 0

    draw_labels()

    while running:
        eventList = pygame.event.get()
        for event in eventList:
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEMOTION and not pixelated:
                if pygame.mouse.get_pressed()[0]:
                    last_pos = (event.pos[0] - event.rel[0], event.pos[1] - event.rel[1])
                    draw_line(last_pos, pygame.mouse.get_pos())
                    stop_time = 0

        if len(eventList) == 0:
            stop_time += dt

        if dt > UPDATE and stop_time < SLEEP and not pixelated:
            draw_chart(net.run(classify()))
            dt = 0

        screen.blit(pixel_pane if pixelated else drawing_pane, (40, 40))
        screen.blit(predictions, (460, 40))

        pygame_widgets.update(eventList)
        pygame.display.flip()
        dt += clock.tick(60) / 1000

    pygame.quit()

def draw_labels():
    for i in range(10):
        screen.blit(label_font.render(str(i), True, BLACK), (440, 40 + i * 38))

def draw_chart(probs):
    pygame.draw.rect(screen, BACKGROUND, pygame.Rect(680, 40, 40, 380))
    predictions.fill(WHITE)
    pred = np.argmax(probs, axis=1)[0]
    for i in range(10):
        pygame.draw.rect(predictions, (50, 50, 200), (0, 8 + i * 38, probs[0][i] * 220, 20))
        color = WHITE if i == pred else BLACK
        prob_text = prob_font.render(percent(probs[0][i]), True, color)
        screen.blit(prob_text, (685, 40 + i * 38))

def percent(prob):
    return f"{round(prob * 100)}%"

def draw_line(start, end):
    dy, dx = end[1] - start[1], end[0] - start[0]
    distance = round(math.hypot(dx, dy))
    for i in range(distance):
        x = start[0] + i / distance * dx
        y = start[1] + i / distance * dy
        if drawing_rect.collidepoint(x, y):
            pygame.draw.circle(drawing_pane, BLACK, (x - 40, y - 40), 12)

def classify():
    scaled = pygame.transform.smoothscale(drawing_pane, (28, 28))
    image = pygame.surfarray.array3d(scaled)
    image = abs(1 - image / 253)
    image = np.mean(image, axis=2)
    return image.transpose().ravel()

def clear():
    drawing_pane.fill(WHITE)
    global pixelated
    if pixelated:
        pixelated = False
        toggle.setText("Pixelated")

def toggle_view():
    global pixelated
    if toggle.string == "Pixelated":
        toggle.setText("Draw")
        pixelated = True
        image = classify().reshape(28, 28)
        image = 255 - image * 255
        for i in range(28):
            for j in range(28):
                val = int(image[i][j])
                pygame.draw.rect(pixel_pane, (val, val, val), (int(380 / 28 * j), int(380 / 28 * i), int(380 / 28), int(380 / 28)))
    else:
        toggle.setText("Pixelated")
        pixelated = False

if __name__ == "__main__":
    main()
