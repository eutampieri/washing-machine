# -*- coding: utf-8 -*-
"""ProgettoVA_E_R_TAMPIERI_PULIGHE.ipynb

Original file is located at
    https://colab.research.google.com/drive/1Q2O4vVdsHupAAqJZQbwieGtZK8L9o1JE

"""

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses
import cv2
import numpy as np

digits_model = tf.keras.models.load_model('digitsp.keras')

template_display = cv2.imread('/content/display_template.png',cv2.IMREAD_GRAYSCALE)
template_bar = cv2.imread('/content/template_bar.jpg',cv2.IMREAD_GRAYSCALE)

def find_region(image, template):
  result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
  _, _, _, top_left = cv2.minMaxLoc(result)
  bottom_right = (top_left[0]+template.shape[1], top_left[1]+template.shape[0])
  return (top_left, bottom_right)

def find_regions(image):
  display_region = find_region(image[:, :image.shape[1]//2], template_display)
  x_offset = display_region[1][0]+300
  im = (image[display_region[0][1]:,x_offset:x_offset+250]).copy()
  if im.shape[0] < 80 or im.shape[1] < 80:
    raise ValueError('Out of bounds.')
  im[im > np.average(im)] = np.average(im)
  bar_region = find_region(im, template_bar)
  bar_region = ((bar_region[0][0]+x_offset, bar_region[0][1]+ display_region[0][1]), (bar_region[1][0]+x_offset, bar_region[1][1]+display_region[0][1]))
  return (display_region, bar_region)

def extract_region(image, region):
  return image[region[0][1]:region[1][1], region[0][0]:region[1][0]]

def pad_and_resize_image(image):
  width = 64
  left = (width - image.shape[1])//2
  right = width-left-image.shape[1]
  top = (width - image.shape[0])//2
  bottom = width-top-image.shape[0]
  return cv2.resize(cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT), (32,32), interpolation=cv2.INTER_LINEAR)

def extract_digits(display):
   regions = [ ((0, 0), (21, 53)), ((18, 0), (32, 53)), ((30, 0), (46, 53)) ]
   return [pad_and_resize_image(extract_region(display, r)) for r in regions]

def digit_recognition(digit, m=10):
  digit = np.expand_dims(digit, axis=0)
  digit = np.expand_dims(digit, axis=-1)
  predictions = digits_model.predict(digit, verbose=0)
  predictions[m+1:9]=0
  return np.argmax(predictions) % 10

def minutes_calculation(digits):
  d = digits
  d[0] = 0 if d[0] != 1 else 1
  return digits[0]*60 + digits[1] * 10 + digits[2]

def get_bands(image):
  step = image.shape[0] // 6
  start = [step * i for i in range(6)]
  regions = [((0, i), (image.shape[1], i + step)) for i in start]
  return [np.average(extract_region(image, r)) for r in regions]

def phase_extraction(bar):
  img_avg = np.average(bar)
  img_var = np.std(bar)
  img_thr = img_avg - 460 / img_var
  _, img_bin = cv2.threshold(bar, img_thr, 255, cv2.THRESH_BINARY)
  img_bin = 255 - img_bin
  closing_structuring_el = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
  opened_img = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, closing_structuring_el)
  bands = get_bands(opened_img)
  phases = ["Ammollo", "Lavaggio", "Risciacquo", "Stop con acqua", "Centrifuga", "Fase antipiega/Fine"]
  return next((p for x, p in zip(bands, phases) if x <= 20), None)

def recognise(image):
  display_region, bar_region = find_regions(image)
  display = extract_region(image, display_region)
  bar = extract_region(image, bar_region)
  digits = extract_digits(display)
  max_digit=[2,5,9]
  digits = [digit_recognition(d,m) for d, m in zip(digits,max_digit)]
  minutes = minutes_calculation(digits)
  phase = phase_extraction(bar)
  on_off = 1 if phase is not None else 0
  return on_off, phase, minutes

img_off = cv2.imread('/content/1733739730.jpg', cv2.IMREAD_GRAYSCALE)
print(recognise(img_off))

img_on = cv2.imread('/content/1733757429L75.jpg', cv2.IMREAD_GRAYSCALE)
print(recognise(img_on))

digits_model.summary()

import glob
dataset = glob.glob("dataset/*")

def extract_label(f):
  filename = f.split('/')[1].split('.')[0][10:]
  if len(filename) == 0:
    return (0, None, None)
  else:
    return (1, filename[0], int(filename[1:]))

acc_s = 0
acc_m = 0
err_m= 0
acc_on_off = 0
count=0
count_off= 0
for i in dataset:
  count += 1
  l_on_off, l_status, l_minutes = extract_label(i)
  image = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
  try:
    on_off_status, r_status, r_minutes = recognise(image)
    if on_off_status == l_on_off:
       acc_on_off += 1
    if on_off_status == 0:
       count_off += 1
       continue
    r_status = r_status[0]
    acc_s += 1 if r_status == l_status else 0
    n=0
    if l_minutes is None or r_minutes is None:
      pass
    elif l_minutes == 0:
      n += 1 if r_minutes == l_minutes else 0
    else:
      n += 1- (abs(l_minutes - r_minutes)/l_minutes)
    err_m += abs(l_minutes - r_minutes)
    acc_m += n
    #print (i,l_minutes, r_minutes, n)
  except Exception as e:
    print(i)
    print( on_off_status, r_status, r_minutes)
    print(e, image.shape)
    continue

acc_s /= (count-count_off)
acc_m /= (count-count_off)
err_m /= (count-count_off)
acc_on_off /= count
print(count, 'images:',acc_s, acc_m,acc_on_off)

acc_syestem = (acc_s+acc_m+acc_on_off)/3
print(acc_syestem)
print(f'L\'Errore medio sui minuti è {err_m}')
print(f'L\'accuratezza totale del sistema è {acc_syestem}')
