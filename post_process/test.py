import cv2
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser(description='Get directions')
parser.add_argument('--image_path', type=str, help='Path to image', required=True)
parser.add_argument('--pred_file', type=str, help="path to prediction file", required=True)
args = parser.parse_args()


file = open(args.pred_file, 'r')
lines = file.readlines()
lines = list(map(lambda x: x.strip().split(","), lines))

groups = defaultdict(list)
for line in lines:
    frame, cls, tlwh = int(line[0]), int(line[1]), list(map(float, line[2:]))
    groups[frame].append((frame, cls, tlwh))


print(groups)
vidcap = cv2.VideoCapture(args.image_path)        
            
success,image = vidcap.read()        
idx = 0

while True:
    # display the image and wait for a keypress
    
    success, image = vidcap.read()
    if(not success):
        break
    
    cv2.rectangle(image, (2239, 1), (2695, 572), (255, 0, 0), 2)
    for data in groups[idx]:
        x1, y1, w, h = list(map(int, data[-1]))
        cv2.rectangle(image, (x1, y1), (x1+w, y1+h), (0, 0, 255), 2)
    
    cv2.imshow("image", image)
    
    idx += 1
    key = cv2.waitKey(1) & 0xFF
    # if the 'r' key is pressed, reset the cropping region
    # if the 'c' key is pressed, break from the loop
    if key == ord("c"):
        break