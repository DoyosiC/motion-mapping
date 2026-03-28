import cv2
import mediapipe as mp
import pygame
import os

# TensorFlowの警告を抑制
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# MediaPipe Handsの初期化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Pygameの初期化
pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Hand Tracking Game Control')

# カメラの設定
cap = cv2.VideoCapture(0)

# 手の高さの状態を保持する変数
left_hand_position = '中'
right_hand_position = '中'

def convert_opencv_to_pygame(img):
    """OpenCVの画像をPygameのSurfaceに変換"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGRからRGBに変換
    img = cv2.transpose(img)  # 軸を回転
    return pygame.surfarray.make_surface(img)

def get_hand_position(y, screen_height):
    """手のy座標に基づいて上・中・下を判定"""
    if y < screen_height / 3:
        return '上'
    elif y < (2 * screen_height / 3):
        return '中'
    else:
        return '下'

# メインループ
running = True

try:
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        success, img = cap.read()
        if not success:
            print("カメラから映像を取得できませんでした。")
            break

        # 画像をRGBに変換
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 画像を処理して手を検出
        results = hands.process(img_rgb)

        # 手のランドマークを描画
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # 手の種類（左手または右手）を取得
                hand_label = handedness.classification[0].label

                # ランドマークの座標を取得 (画面の高さにスケーリング)
                index_finger_tip_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * SCREEN_HEIGHT)

                # 手の高さを3分割して判定
                position = get_hand_position(index_finger_tip_y, SCREEN_HEIGHT)

                # 左右の手の位置を更新
                if hand_label == 'Left':
                    left_hand_position = position
                else:
                    right_hand_position = position

                # ランドマークを描画
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # OpenCVの画像をPygameのSurfaceに変換して描画
        frame_surface = convert_opencv_to_pygame(img)
        screen.blit(frame_surface, (0, 0))

        # フォントの初期化（freesansboldを使用）
        font = pygame.font.SysFont('Arial', 36)

        # 手の高さを表示
        text = font.render(f'左手: {left_hand_position}  右手: {right_hand_position}', True, (255, 255, 255))
        screen.blit(text, (20, 20))

        # 画面を3分割してガイドラインを描画
        pygame.draw.line(screen, (0, 255, 0), (0, SCREEN_HEIGHT // 3), (SCREEN_WIDTH, SCREEN_HEIGHT // 3), 2)
        pygame.draw.line(screen, (0, 255, 0), (0, 2 * SCREEN_HEIGHT // 3), (SCREEN_WIDTH, 2 * SCREEN_HEIGHT // 3), 2)

        # Pygameの画面を更新
        pygame.display.update()

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False

except (KeyboardInterrupt, SystemExit):
    pass

finally:
    # リソースの解放
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()