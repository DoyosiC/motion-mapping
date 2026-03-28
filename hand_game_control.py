import cv2
import mediapipe as mp
import pygame
import os
import random

# TensorFlowの警告を抑制
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# =====================
# 設定用の変数
# =====================

# ゲーム全体の設定
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
FPS = 60

# プレイヤーの設定
PLAYER_SPEED = 5           # 左右の移動速度
PLAYER_JUMP_HEIGHT = 15    # ジャンプの高さ
GRAVITY = 0.75             # 重力の強さ

# 敵の設定
ENEMY_SPAWN_RATE = 60     # 敵の生成頻度（数値が小さいほど頻度が高い）
ENEMY_MIN_SPEED = 3        # 敵の最小速度
ENEMY_MAX_SPEED = 7        # 敵の最大速度

# 難易度の調整
HAND_DETECTION_CONFIDENCE = 0.7  # 手の検出信頼度
HAND_TRACKING_CONFIDENCE = 0.7   # 手の追跡信頼度

# =====================
# Pygameの初期化
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Hand Tracking Avoidance Game')
clock = pygame.time.Clock()

# MediaPipe Handsの初期化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=HAND_DETECTION_CONFIDENCE,
    min_tracking_confidence=HAND_TRACKING_CONFIDENCE
)
mp_draw = mp.solutions.drawing_utils

# プレイヤーのクラス
class Player:
    def __init__(self):
        self.image = pygame.Surface((50, 50))
        self.image.fill((0, 128, 255))
        self.rect = self.image.get_rect()
        self.rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 60)
        self.velocity_y = 0
        self.is_jumping = False

    def update(self, moving_left, moving_right, jump):
        dx = 0

        # 左右の移動
        if moving_left:
            dx = -PLAYER_SPEED
        if moving_right:
            dx = PLAYER_SPEED

        # ジャンプ処理
        if jump and not self.is_jumping:
            self.velocity_y = -PLAYER_JUMP_HEIGHT
            self.is_jumping = True

        # 重力を適用
        self.velocity_y += GRAVITY
        if self.velocity_y > 10:
            self.velocity_y = 10

        self.rect.y += self.velocity_y

        # 地面に着地したらジャンプ終了
        if self.rect.bottom > SCREEN_HEIGHT:
            self.rect.bottom = SCREEN_HEIGHT
            self.is_jumping = False

        self.rect.x += dx

        # 画面外に出ないように制限
        self.rect.x = max(0, min(self.rect.x, SCREEN_WIDTH - self.rect.width))

    def draw(self):
        screen.blit(self.image, self.rect)

# 敵のクラス
class Enemy:
    def __init__(self, direction):
        self.image = pygame.Surface((30, 30))
        self.image.fill((255, 0, 0))
        self.rect = self.image.get_rect()
        self.speed = random.randint(ENEMY_MIN_SPEED, ENEMY_MAX_SPEED)
        self.direction = direction

        # 左右または上から物体を生成
        if direction == 'left':
            self.rect.x = 0
            self.rect.y = random.randint(0, SCREEN_HEIGHT - 50)
        elif direction == 'right':
            self.rect.x = SCREEN_WIDTH - 30
            self.rect.y = random.randint(0, SCREEN_HEIGHT - 50)
        elif direction == 'top':
            self.rect.x = random.randint(0, SCREEN_WIDTH - 50)
            self.rect.y = 0

    def update(self):
        if self.direction == 'left':
            self.rect.x += self.speed
        elif self.direction == 'right':
            self.rect.x -= self.speed
        elif self.direction == 'top':
            self.rect.y += self.speed

    def draw(self):
        screen.blit(self.image, self.rect)

    def off_screen(self):
        return (self.rect.x < -50 or self.rect.x > SCREEN_WIDTH + 50 or
                self.rect.y > SCREEN_HEIGHT + 50)


# プレイヤーの初期化
player = Player()
enemies = []
cap = cv2.VideoCapture(1)

def get_hand_position(y, screen_height):
    if y < screen_height / 3:
        return '上'
    elif y < (2 * screen_height / 3):
        return '中'
    else:
        return '下'

def convert_opencv_to_pygame(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.transpose(img)
    return pygame.surfarray.make_surface(img)

# ゲームのメインループ
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    success, img = cap.read()
    if not success:
        break

    # OpenCVでカメラ映像を取得し、RGBに変換
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # プレイヤーの動きの初期化
    moving_left = moving_right = jump = False

    # 手の検出と動きの判定
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label
            index_finger_tip_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * SCREEN_HEIGHT)
            position = get_hand_position(index_finger_tip_y, SCREEN_HEIGHT)

            if hand_label == 'Left' and position == '上':
                jump = True
            elif hand_label == 'Right':
                if position == '上':
                    moving_right = True
                elif position == '下':
                    moving_left = True

    # プレイヤーの動きを更新
    player.update(moving_left, moving_right, jump)

    # 敵の生成
    if random.randint(1, ENEMY_SPAWN_RATE) == 1:
        direction = random.choice(['left', 'right', 'top'])
        enemies.append(Enemy(direction))

    # 敵の更新と当たり判定のチェック
    for enemy in enemies[:]:
        enemy.update()
        if player.rect.colliderect(enemy.rect):
            print("ゲームオーバー")
            font = pygame.font.SysFont('Arial', 48)
            text = font.render('ゲームオーバー', True, (255, 0, 0))
            screen.blit(text, (SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 - 24))
            pygame.display.update()
            pygame.time.wait(2000)
            running = False

        if enemy.off_screen():
            enemies.remove(enemy)

    # OpenCVの画像をPygameのSurfaceに変換して描画
    frame_surface = convert_opencv_to_pygame(img)
    screen.blit(frame_surface, (0, 0))

    # プレイヤーと敵の描画
    player.draw()
    for enemy in enemies:
        enemy.draw()

    # 画面更新
    pygame.display.update()
    clock.tick(FPS)

cap.release()
cv2.destroyAllWindows()
pygame.quit()
