"""Simple Pymunk example with a single joint hanging from one end"""
import pygame
import pymunk
import pymunk.pygame_util

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((600, 600))
pygame.display.set_caption("Hanging Joint Example")
clock = pygame.time.Clock()

# Initialize Pymunk
space = pymunk.Space()
space.gravity = (0, 900)  # Gravity pointing downward

# Create drawing options
draw_options = pymunk.pygame_util.DrawOptions(screen)

# Create a static body (fixed point at the top)
static_body = space.static_body
anchor_pos = (300, 100)  # Position where the joint is attached

# Create a dynamic body (the hanging object) - just a point with mass
mass = 10
moment = 10
body = pymunk.Body(mass, moment)
body.position = (400, 300)  # Starting position offset to the right so it swings


# Add just the body to the space (no shape, so it won't be visible)
space.add(body)

# Create a pin joint connecting the static body to the dynamic body
# PinJoint(body_a, body_b, anchor_a, anchor_b)
# anchor_a and anchor_b are in local coordinates relative to each body
joint = pymunk.PinJoint(static_body, body, anchor_pos, (0, 0))
space.add(joint)

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_SPACE:
                # Apply a velocity to make it swing
                # body.velocity = (500, 0)
                pass

    # Clear screen
    screen.fill((255, 255, 255))
    
    # Draw physics objects
    space.debug_draw(draw_options)
    
    # Update physics
    space.step(1/60.0)
    
    # Update display
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
