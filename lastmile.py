import cv2 
from craft_text_detector import Craft
import numpy as np
import re
import easyocr
import pandas as pd
import haversine as hs


class LastMile():

  def __init__(self, text_localization_object, text_recognition_object, delivery_dataset, distance_threshold):
    """ initializes text localization model, text recognition model, delivery dataset and distance threshold
    * "text_localization_object" is based on CRAFT
    * "text_recognition_object" is based on EasyOCR
    * "delivery_dataset" is a dictionary that maps the stops numbers to their related info: actual long and lat coordinates, 
      long and lat coordinates captured by the driver responsible for delivery, images of unit number(s), address of the customer,
      type of address (building or house)
    * "distance threshold" is the distance below which a delivery is considered to delivered in the right place. Meaning that the distance 
      between actual and captured coordinates must be less than or equal to the distance threshold. 
    """
    self.tl_model = text_localization_object
    self.tr_model = text_recognition_object
    self.dds = delivery_dataset
    self.distance_threshold = distance_threshold

  
  def img_read(self, image, color = cv2.COLOR_BGR2RGB):
     """ reads a single input image with default RGB color spance using OpenCV
    * argument "image": is an image captured by the driver containing the unit numbers of a building/apartments and houses depending on the type of the address. 
    * argument "color": is the conversion from the default BGR color value in OpenCV to RGB
    * returns "img": is an image returned according to "image" and "color" settings
    """
    img = cv2.imread(image, color)
    return img

  def img_sat(self, img, sat_val):
    """ updates color saturation of an image
    * argument "img": is the image to be color saturation adjusted.
    * argument "sat_val": is the color saturation value, it can be both integer and float.
    * returns "rbg_img": which is an RGB image with adjusted saturation is returned.
    """
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype("float32")
    (h, s, v) = cv2.split(hsv_img)
    s = np.clip(s*sat_val,0,255)
    hsv_img = cv2.merge([h,s,v])
    rgb_img = cv2.cvtColor(hsv_img.astype("uint8"), cv2.COLOR_HSV2RGB)
    return rgb_img

  def img_contrast(self, img, clipLimit, tileGridSize=(8,8)):
    """ updates cotrast of an image using the CLAHE (Contrast Limiting Adaptive Histogram Equalization)
    * argument "img": is the image to be contrast adjusted.
    * argument "clipLimit": is the contrast limit for localized changes in contrast.
    * argument "tileGridSize": the grid size the image is divided into to apply CLAHE.
    * returns "img": which is an RGB image with adjusted contrast.
    """
    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    h, s, v = cv2.split(lab_img)    
    clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    cl = clahe.apply(h)
    lab_img = cv2.merge((cl,s,v))
    img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)
    return img

  def img_brightness(self, img, bright_val):
    """ updates brightness of an image
    * argument "img": is the image to be brigthness adjusted.
    * argument "bright_val": is the brightness value, the higher it is the higher the brightness.
    * returns "rgb_img": which is an RGB image with adjusted brightness.
    """
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_img)
    lim = 255 - bright_val
    v[v > lim] = 255
    v[v <= lim] += bright_val
    hsv_img = cv2.merge((h, s, v))
    rgb_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    return rgb_img

  def text_localization_crops(self, img, expand_per=0.01):
    """ localizes texts in an image 
    * argument "img": is the image whose text info will be localized.
    * argument "expand_per": is a factor by which a localized text is increase. This is necessary since localized texts in cropped a image often gets missing 
      pixels from the left side of the most left character and missing pixels in the right most character at the end of the cropped image.  
    * returns "text_crops": is a list of cropped images containing detected texts in image "img".
    """
    texts = self.tl_model.detect_text(img)
    bbox = texts['boxes']
    k=0
    text_crops=[]
    for n in range(len(bbox)):
      s00 = int(bbox[n][1][1]*(1-expand_per))
      s01 = int(bbox[n][2][1]*(1+expand_per))
      s10 = int(bbox[n][0][0]*(1-expand_per))
      s11 = int(bbox[n][1][0]*(1+expand_per))
      text_crops.append(img[s00:s01,s10:s11])
      k+=1
    return text_crops

  def crops_text_recognition(self, text_crops):
    """ recognizes texts in in the cropped images from the "text_localization_crops" function
    * argument "text_crops": is the image whose text info will be localized.
    * returns "recognized_texts": is a list of recognized texts from the cropped images images in "text_crops".
    """
    recognized_texts = []
    for text_crop in text_crops:
      recognized_texts.append(self.tr_model.recognize(text_crop, detail=0)[0])
    return recognized_texts

  def distance_difference(self, stop_num):
    """ calculates the difference between the acutal coordinates of the customer and the captured coordinates by the driver. the coordinates values 
        are obtained from the database of stops in a route.
    * argument "stop_num": is the stop number where distance must be calculated ex ('stop_0', 'stop_1' etc).
    * returns "distance": distance in meters between the mentioned coordinates.
    """
    stop = self.dds[stop_num]
    distance = 1000*hs.haversine(stop['actual_lat_long'],stop['captured_lat_long']) # in meters
    return distance

  def units_from_address(self, stop_num):
    """ extracts actual units numbers from a given address
    * argument "stop_num": is the stop number where units numbers are to be extracted.
    * returns "units_nums": a list of extracted unit numbers of an address.
    """
    stop = self.dds[stop_num]
    address = stop['address']
    units_nums = re.findall(r"\d+",address)
    return units_nums

  # match predicted and acutual units
  def match_units(self, predicted_units,actual_units):
    """ matches predicted unit numbers from images and actual units numbers extracted from an address
    * argument "predicted_units": are units numbers recognized from images.
    * argument "actual_units": are units numbers extracted from an address.
    * returns:
      - "match": if there are common units numbers between "actual_units" and "predicted_units".
      - "None" otherwise. 
    """
    # predicted_units 
    # actual_units .
    match = [value for value in predicted_units if value in actual_units] 
    if bool(match):
      return match
    else:
      return None
  
  def delivery_correctness(self):
    """ decides the correctness of deliveries based on distance and recognized units numbers from images taken by drivers 
        against the address given by the customer. refer to flowchart in the main page of the project to check out how 
        it functions. 
    * returns:
      - "'correct delivery'": if the delivery is correctly done by the driver.
      - "'wrong delivery'" otherwise. 
    """
    truth = {}
    
    for stop in self.dds:
      images = self.dds[stop]['images']
      address_units = self.units_from_address(stop)
      recognized_texts = []

      for image in images:
        img = self.img_read(image)
        text_crops = self.text_localization_crops(img)
        recognized_texts += self.crops_text_recognition(text_crops)

      recognized_units = self.match_units(recognized_texts, address_units)
      if recognized_units==None:
        recognized_units = []

      distance = self.distance_difference(stop)

      building_type = self.dds[stop]['type']
      if building_type == 'building':
        if (len(recognized_units)==2) & (distance <= self.distance_threshold):
          truth[stop] = 'correct delivery'
        else:
          truth[stop] = 'wrong delivery'
      if building_type == 'house':
        if (len(recognized_units)==1) & (distance <= self.distance_threshold):
          truth[stop] = 'correct delivery'
        else:
          truth[stop] = 'wrong delivery'
    return truth