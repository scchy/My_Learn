
import cv2
import argparse
import numpy as np
# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--img_file', type=str, default='./imgs/truck.jpg', help='image file')
parser.add_argument('--metric_cofig_file', type=str, default='./config/metric_cofig.txt', help='metric config file')
args = parser.parse_args()


img_file = args.img_file
metric_cofig_file = args.metric_cofig_file
config_points = []

def on_mouse_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONUP: # 左键点击
        if len(config_points) < 2:
            config_points.append((x,y))
            cv2.circle(image, (x,y), 10, (0, 255, 0), -1)
        else:
            print('config points is full')


image = cv2.imread(img_file)
cv2.namedWindow('image')
cv2.setMouseCallback('image',on_mouse_event) # 设置鼠标事件

while(True):
    # 显示配置点
    if len(config_points) == 2:
        cv2.line(image, config_points[0], config_points[1], (0, 0, 255), 5)
    
    cv2.imshow('image',image)
    # 退出ESC
    if cv2.waitKey(20) & 0xFF == 27:
        break
    # enter键保存配置文件
    elif cv2.waitKey(20) & 0xFF == 13:
        cv2.putText( image , 'Saved config', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        with open(metric_cofig_file, 'w', encoding='utf8') as fw:
            # 写入配置文件，每行一个点：x,y
            for pt in config_points:
                write_line = str(pt[0]) + ',' + str(pt[1]) + '\n'
                fw.write(write_line)
            # 计算点之间的像素距离
            distance = np.linalg.norm(np.array(config_points[0]) - np.array(config_points[1]))
            print('distance is ', distance)

        print('save config file')
        


cv2.destroyAllWindows()



