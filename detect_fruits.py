import cv2
import json
import click
import numpy as np

from glob import glob
from tqdm import tqdm

from typing import Dict


def detect_fruits(img_path: str) -> Dict[str, int]:

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    apple = 0
    banana = 0
    orange = 0

    # changing size of picture
    img = cv2.resize(img, None, fx=0.2, fy=0.2)

    # from rgb to hsv
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # define fruits range
    bottom_banana = np.array([23, 100, 105])
    top_banana = np.array([36, 255, 255])
    bottom_orange = np.array([0, 218, 122])
    top_orange = np.array([16, 255, 255])
    bottom_every_fruit = np.array([0, 100, 63])
    top_every_fruit = np.array([255, 255, 255])

    # image masking and filtering
    mask_banana = cv2.inRange(img_hsv, bottom_banana, top_banana)
    mask_banana = cv2.medianBlur(mask_banana, 57)
    mask_orange = cv2.inRange(img_hsv, bottom_orange, top_orange)
    mask_orange = cv2.medianBlur(mask_orange, 57)
    mask_every_fruit = cv2.inRange(
        img_hsv, bottom_every_fruit, top_every_fruit)
    mask_every_fruit = cv2.medianBlur(mask_every_fruit, 57)

    # finding borders
    borders_banana = cv2.findContours(
        mask_banana, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    borders_orange = cv2.findContours(
        mask_orange, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    borders_every_fruit = cv2.findContours(
        mask_every_fruit, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

    # counting number of fruits
    s_bottom = 1
    s_top = 20000000
    banana_cnts = []
    orange_cnts = []
    every_fruit_cnts = []

    for i in borders_banana:
        if s_bottom < cv2.contourArea(i) < s_top:
            banana_cnts.append(i)

    for i in borders_orange:
        if s_bottom < cv2.contourArea(i) < s_top:
            orange_cnts.append(i)

    for i in borders_every_fruit:
        if s_bottom < cv2.contourArea(i) < s_top:
            every_fruit_cnts.append(i)

    banana = len(banana_cnts)
    orange = len(orange_cnts)
    apple = len(every_fruit_cnts)-len(banana_cnts)-len(orange_cnts)

    return {'apple': apple, 'banana': banana, 'orange': orange}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory')
@click.option('-o', '--output_file_path', help='Path to output file')
def main(data_path, output_file_path):

    img_list = glob(f'{data_path}/*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        fruits = detect_fruits(img_path)

        filename = img_path.split('/')[-1]

        results[filename] = fruits

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)

    print(results)


if __name__ == '__main__':
    main()
