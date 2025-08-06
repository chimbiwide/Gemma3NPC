import pygame
import random
import sys
import os

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

#sprites for actual game
class Player(pygame.sprite.Sprite):
    def __init__(self, groups):
        super().__init__(groups)
        self.image = pygame.image.load(resource_path('Space_Shooter/images/player.png')).convert_alpha()
        self.rect = self.image.get_frect(center=(WINDOW_WIDTH/2,WINDOW_HEIGHT/2))
        self.direction = pygame.math.Vector2(0,0)
        self.speed = 500

        #cooldown
        self.can_shoot = True
        self.laser_shoot_time = 0
        self.cooldown_duration = 400

        #mask
        self.mask = pygame.mask.from_surface(self.image)

    def laser_timer(self):
        if not self.can_shoot:
            current_time = pygame.time.get_ticks()
            if current_time - self.laser_shoot_time >= self.cooldown_duration:
                self.can_shoot = True

    def update(self,dt):
        keys = pygame.key.get_pressed()
        recent_keys = pygame.key.get_just_pressed()
        self.direction.x = int(keys[pygame.K_d]) - int(keys[pygame.K_a])
        self.direction.y = int(keys[pygame.K_s]) - int(keys[pygame.K_w])
        self.direction = self.direction.normalize() if self.direction else self.direction
        self.rect.center += self.direction * self.speed * dt
        if recent_keys[pygame.K_SPACE] and self.can_shoot:
            Laser(self.rect.midtop, (all_sprites, laser_sprites), laser_surf)
            self.can_shoot = False
            self.laser_shoot_time = pygame.time.get_ticks()
            laser_sound.play()
        self.laser_timer()

class Star(pygame.sprite.Sprite):
    def __init__(self, groups, star_surf):
        super().__init__(groups)
        self.image = star_surf
        self.rect = self.image.get_frect(center = (random.randint(0, WINDOW_WIDTH),random.randint(0, WINDOW_HEIGHT)))

class Laser(pygame.sprite.Sprite):
    def __init__(self, pos, groups, laser_surf):
        super().__init__(groups)
        self.image = laser_surf
        self.rect = self.image.get_frect(midbottom = pos)

    def update(self,dt):
        self.rect.centery -= 500 * dt
        if self.rect.bottom < 0:
            self.kill()

class Meteor(pygame.sprite.Sprite):
    def __init__(self, surf, pos, groups):
        super().__init__(groups)
        self.original_surf = surf
        self.image = surf
        self.rect = self.image.get_frect(center = pos)
        self.start_time = pygame.time.get_ticks()
        self.life_time = 3000
        self.direction = pygame.math.Vector2(random.uniform(-0.5, 0.5),1)
        self.speed = random.randint(300, 500)
        self.rotation_speed = random.randint(40, 80)
        self.rotation = 0
    def update(self,dt):
        self.rect.center += self.direction * self.speed * dt
        if pygame.time.get_ticks() - self.start_time >= self.life_time:
            self.kill()
        self.rotation += self.rotation_speed * dt
        self.image = pygame.transform.rotozoom(self.original_surf, self.rotation, 1)
        self.rect = self.image.get_frect(center = self.rect.center)

class AnimatedExplosion(pygame.sprite.Sprite):
    def __init__(self, frames, pos, groups):
        super().__init__(groups)
        self.frames = frames
        self.frames_index = 0
        self.image = self.frames[self.frames_index]
        self.rect = self.image.get_frect(center = pos)

    def update(self,dt):
        self.frames_index += 30 * dt
        if self.frames_index < len(self.frames):
            self.image = self.frames[int(self.frames_index)]
        else:
            self.kill()

#sprites for the hub (spaceship background and the dog)
class spaceship(pygame.sprite.Sprite):
    def __init__(self, surf, pos, groups):
        super().__init__(groups)
        self.image = surf
        self.rect = self.image.get_frect(center = pos)

    def update(self,in_hub):
        if not in_hub:
            self.kill()

class dog(pygame.sprite.Sprite):
    def __init__(self, surf, pos, groups):
        super().__init__(groups)
        self.image = surf
        self.rect = self.image.get_frect(center = pos)

    def update(self,in_hub):
        if not in_hub:
            self.kill()

#functions
def collision():
    global running, in_game, in_hub, in_died, current_score, scores, score_saved

    collision_sprites = pygame.sprite.spritecollide(player, meteor_sprites, True, pygame.sprite.collide_mask)
    if collision_sprites:
        in_game = False
        in_died = True
        game_music.stop()
        # Save score once when player dies
        if not score_saved:
            scores.append(current_score)
            write_score()
            score_saved = True

    for laser in laser_sprites:
        collided_sprites = pygame.sprite.spritecollide(laser, meteor_sprites, True)
        if collided_sprites:
            laser.kill()
            AnimatedExplosion(explosion_frames, laser.rect.midtop, all_sprites)
            explosion_sound.play()

def display_score():
    global current_score
    current_score = (pygame.time.get_ticks() - game_start_time) // 500
    text_surf = font.render(str(current_score), True, '#07b50d')
    text_rect = text_surf.get_frect(midbottom = (WINDOW_WIDTH/2,WINDOW_HEIGHT- 50))
    screen.blit(text_surf, text_rect)
    pygame.draw.rect(screen,'#07b50d', text_rect.inflate(20,10).move(0,-8), 5, 10)

def display_death():
    global death_sound_played, clicked, in_game, running, score_clock, in_died, current_score, scores
    #you died words
    you_died_surf = you_died_font.render("You Died", True, '#ff4444')
    you_died_rect = you_died_surf.get_frect(center = (480, 300))
    screen.blit(you_died_surf, you_died_rect)
    if not death_sound_played:
        death_sound.play(1)
        death_sound_played = True

    #continue?
    continue_surf = game_choices_font.render("Return to hub", True, '#07b50d')
    continue_rect = continue_surf.get_rect(center = (280, 450))
    button_rect = continue_rect.inflate(40,20)
    if button_rect.collidepoint(mouse_pos):
        pygame.draw.rect(screen, '#2d5a2d', button_rect)  # Lighter green when hovering
        pygame.draw.rect(screen, '#0aff0a', button_rect, 3)  # Brighter border
    else:
        pygame.draw.rect(screen, '#1a3d1a', button_rect)  # Normal background
        pygame.draw.rect(screen, '#07b50d', button_rect, 3)  # Normal border
    screen.blit(continue_surf, continue_rect)

    #quit
    quit_surf = game_choices_font.render("Click to quit", True, '#ff6b6b')
    quit_rect = quit_surf.get_rect(center = (680, 450))
    quit_button_rect = quit_rect.inflate(40,20)
    if quit_button_rect.collidepoint(mouse_pos):
        pygame.draw.rect(screen, '#5a2d2d', quit_button_rect)  # Lighter red when hovering
        pygame.draw.rect(screen, '#ff8888', quit_button_rect, 3)  # Brighter border
    else:
        pygame.draw.rect(screen, '#3d1a1a', quit_button_rect)  # Normal background
        pygame.draw.rect(screen, '#ff4444', quit_button_rect, 3)
    screen.blit(quit_surf, quit_rect)

def display_hub():
    play_game_surf = game_choices_font.render("Play Game", True, '#07b50d')
    play_game_rect = play_game_surf.get_rect(center = (WINDOW_WIDTH//2, 450))
    button_rect = play_game_rect.inflate(40,20)
    if button_rect.collidepoint(mouse_pos):
        pygame.draw.rect(screen, '#2d5a2d', button_rect)  # Lighter green when hovering
        pygame.draw.rect(screen, '#0aff0a', button_rect, 3)  # Brighter border
    else:
        pygame.draw.rect(screen, '#1a3d1a', button_rect)  # Normal background
        pygame.draw.rect(screen, '#07b50d', button_rect, 3)  # Normal border
    screen.blit(play_game_surf, play_game_rect)
    return button_rect

def reset_game():
    global death_sound_played, game_start_time, in_game, player, in_died, in_hub, death_sound, scores, current_score, score_saved

    # Stop current music
    game_music.stop()
    death_sound.stop()

    # Clear all sprites
    meteor_sprites.empty()
    laser_sprites.empty()
    all_sprites.empty()

    # Recreate everything
    for i in range(20):
        Star(all_sprites, star_surf)

    player = Player(all_sprites)

    # Reset state
    current_score = 0
    score_saved = False
    death_sound_played = False
    game_start_time = pygame.time.get_ticks()
    in_game = False
    in_died = False
    in_hub = True
    game_music.play(loops=-1)

def write_score():
    global scores, current_score
    with open('scores.txt', 'w') as score_file:
        score_file.write("All Previous scores: ")
        score_file.write(str(scores))
        score_file.write('\n')
        score_file.write("Score for the current run: ")
        score_file.write(str(current_score))
        score_file.write('\n')

def clear_score():
    with open('scores.txt', 'w') as score_file:
        score_file.write('')

#setup
pygame.init()

WINDOW_WIDTH = 960
WINDOW_HEIGHT = 720
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Space shooter")

#three diffrent game states --in hub, in game and died
running = True
in_hub = True
in_game = False
in_died = False

death_sound_played = False
clicked = False

scores = []
current_score = 0
score_saved = False

#time
game_start_time = pygame.time.get_ticks()
clock = pygame.time.Clock()

#import for spaceship(hub/base)
dog_surf = pygame.image.load(resource_path('Space_Shooter/images/dog.svg')).convert_alpha()
spaceship_surf = pygame.image.load(resource_path('Space_Shooter/images/spaceship.png')).convert_alpha()

#import for main game
star_surf = pygame.image.load(resource_path('Space_Shooter/images/star.png')).convert_alpha()
meteor_surf = pygame.image.load(resource_path('Space_Shooter/images/meteor.png')).convert_alpha()
laser_surf = pygame.image.load(resource_path('Space_Shooter/images/laser.png')).convert_alpha()
font = pygame.font.Font(resource_path('Space_Shooter/images/Oxanium-Bold.ttf'), 40)
explosion_frames = [pygame.image.load(resource_path(f'Space_Shooter/images/explosion/{i}.png')).convert_alpha() for i in range(21)]
laser_sound = pygame.mixer.Sound(resource_path('Space_Shooter/audio/laser.wav'))
laser_sound.set_volume(0.5)
explosion_sound = pygame.mixer.Sound(resource_path('Space_Shooter/audio/explosion.wav'))
explosion_sound.set_volume(0.5)
game_music = pygame.mixer.Sound(resource_path('Space_Shooter/audio/game_music.wav'))
game_music.set_volume(0.3)
game_music.play(loops=-1)

#import for death Screen
you_died_font = pygame.font.Font(resource_path('Space_Shooter/images/Oxanium-Bold.ttf'), 75)
game_choices_font = pygame.font.Font(resource_path('Space_Shooter/images/Oxanium-Bold.ttf'), 48)
death_sound = pygame.mixer.Sound(resource_path('Space_Shooter/audio/You_Died.ogg'))
death_sound.set_volume(1)

#sprites for hub
hub_sprites = pygame.sprite.Group()
spaceship = spaceship(spaceship_surf, (WINDOW_WIDTH//2, WINDOW_HEIGHT//2), hub_sprites)
dog = dog(dog_surf, (480, 340), hub_sprites)

#sprites for main game
all_sprites = pygame.sprite.Group()
meteor_sprites = pygame.sprite.Group()
laser_sprites = pygame.sprite.Group()
for i in range(20):
    Star(all_sprites, star_surf)
player = Player(all_sprites)

#custom events: Metoer Event
meteor_event = pygame.event.custom_type()
pygame.time.set_timer(meteor_event, 500)

while running:
    dt =  (clock.tick()) / 1000
    mouse_pos = pygame.mouse.get_pos()
    #event loop
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN and not in_game:
            if in_died:
                continue_rect = pygame.Rect(220, 420, 160, 60)  # Match button display area
                quit_rect = pygame.Rect(620, 420, 160, 60)
                if continue_rect.collidepoint(event.pos):
                    reset_game()
                elif quit_rect.collidepoint(event.pos):
                    running = False
            elif in_hub:
                # Get button rect from display_hub function
                play_game_surf = game_choices_font.render("Play Game", True, '#07b50d')
                play_game_rect = play_game_surf.get_rect(center = (WINDOW_WIDTH//2, 450))
                button_rect = play_game_rect.inflate(40,20)
                if button_rect.collidepoint(event.pos):
                    in_hub = False
                    in_game = True
                    game_start_time = pygame.time.get_ticks()  # Reset timer when game actually starts
        if event.type == meteor_event and in_game:
            x, y = random.randint(0,WINDOW_WIDTH), random.randint(-200,-100)
            Meteor(meteor_surf, (x, y), (all_sprites, meteor_sprites))
    if in_hub:
        screen.fill('#023c61')
        hub_sprites.update(in_hub)
        hub_sprites.draw(screen)
        display_hub()
    if in_game:
        all_sprites.update(dt)
        collision()
        #draw the game
        screen.fill('#023c61')
        all_sprites.draw(screen)
        display_score()
    if in_died:
        screen.fill('#000000')
        display_death()
    pygame.display.update()
clear_score()
pygame.quit()