import cv2
import numpy as np
import keyboard

# トラックバーのコールバック関数
def nothing(x):
    pass

# トラックバー用のウィンドウを生成
cv2.namedWindow("OpenCV Window")
cv2.namedWindow("Settings")
cv2.resizeWindow("Settings", 800, 300)  # トラックバーのウィンドウを横に広げ、高さも調整

# 色検出用のトラックバー（HSV値）
cv2.createTrackbar("H_min", "Settings", 0, 179, nothing)
cv2.createTrackbar("H_max", "Settings", 9, 179, nothing)
cv2.createTrackbar("S_min", "Settings", 128, 255, nothing)
cv2.createTrackbar("S_max", "Settings", 255, 255, nothing)
cv2.createTrackbar("V_min", "Settings", 128, 255, nothing)
cv2.createTrackbar("V_max", "Settings", 255, 255, nothing)

# 面積フィルタ用のトラックバー
cv2.createTrackbar("Min Area", "Settings", 500, 5000, nothing)
cv2.createTrackbar("Max Area", "Settings", 50000, 100000, nothing)

# カメラの初期設定（PCカメラ使用）
cap = cv2.VideoCapture(0)

try:
    while True:
        if keyboard.is_pressed('esc'):
            break
        else:
            ret, frame = cap.read()
            if not ret:
                break

            # カメラ映像をBGRからHSVに変換
            hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # HSVトラックバーの値を取得
            h_min = cv2.getTrackbarPos("H_min", "Settings")
            h_max = cv2.getTrackbarPos("H_max", "Settings")
            s_min = cv2.getTrackbarPos("S_min", "Settings")
            s_max = cv2.getTrackbarPos("S_max", "Settings")
            v_min = cv2.getTrackbarPos("V_min", "Settings")
            v_max = cv2.getTrackbarPos("V_max", "Settings")

            # 面積フィルタの値を取得
            min_area = cv2.getTrackbarPos("Min Area", "Settings")
            max_area = cv2.getTrackbarPos("Max Area", "Settings")

            # inRange関数で範囲指定し二値化
            bin_image = cv2.inRange(hsv_image, (h_min, s_min, v_min), (h_max, s_max, v_max))

            # bitwise_andで元画像にマスクをかける -> マスクされた部分の色だけ残る
            masked_image = cv2.bitwise_and(frame, frame, mask=bin_image)

            # 面積・重心計算付きのラベリング処理を行う
            num_labels, label_image, stats, center = cv2.connectedComponentsWithStats(bin_image)

            # 最大のラベルは画面全体を覆う黒なので不要．データを削除
            num_labels -= 1
            stats = np.delete(stats, 0, 0)
            center = np.delete(center, 0, 0)

            detected_count = 0
            if num_labels >= 1:
                for i in range(num_labels):
                    # 各ラベルのx,y,w,h,面積s,重心位置mx,myを取得
                    x = stats[i][0]
                    y = stats[i][1]
                    w = stats[i][2]
                    h = stats[i][3]
                    s = stats[i][4]
                    mx = int(center[i][0])
                    my = int(center[i][1])

                    # 面積フィルタリング
                    if s < min_area or s > max_area:
                        continue

                    detected_count += 1

                    # ラベルを囲うバウンディングボックスを描画
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)

                    # 重心位置の座標と面積を表示
                    cv2.putText(frame, f"Area: {s}", (x, y + h + 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0))
                    cv2.putText(frame, f"Center: ({mx},{my})", (x, y + h + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))

            # 検出数を表示
            cv2.putText(frame, f"Detected: {detected_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # ウィンドウに画像を表示
            cv2.imshow('OpenCV Window', frame)
            cv2.imshow('Masked Image', masked_image)

            # 'q'キーが押されたら終了
            if keyboard.is_pressed('q'):
                break

        cv2.waitKey(1)

    # リソースを解放
    cap.release()
    cv2.destroyAllWindows()

except (KeyboardInterrupt, SystemExit):
    cap.release()
    cv2.destroyAllWindows()
