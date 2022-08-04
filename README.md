# OCR-based MisDelivery Detection in Last Mile Application

![alt text](https://github.com/yahya-bader-khawam/OCR-based-Misdelivery-Detection-in-Last-Mile-Application/blob/main/del1.png?raw=true)


## What is The Project All About?

There are plenty of courier companies that are responsible for delivering packages purchased by customers online. As known, each package is designated to a certain address set by the customer. However, often drivers misdeliver packages to a wrong address. This creates a loss for the courier company as it shoud compensate the customer for losing the package. Normally, this is tracked using the distance between the coordinates of the customer and the coordinate captured by the driver when a derlivery is done. However, tracking distance between coordinates is not enough since a package might misderlivered to a wrong address within the error range of the GPS device of the driver which includes the neighbours of the customers. This project tackle this problem by tracking the distance between the actual and captured coordinates in addition to the units numbers extracted from the customers' addresses images captured by the drivers. The extracted unit numbers from images are compared against the addresses provided by the customers. 

## How The Algorithm Works?

* The following are passed to the LastMile() class: text localization model, text recognition model, delivery dataset and distance threshold.
  * text localization model: localizes texts in an image and crops the image to the detected texts.
  * text recognition model: recognizes texts in in the cropped images from the text localization model function.
  * delivery dataset is a dictionary that maps the stops numbers to their related info: actual long and lat coordinates, 
      long and lat coordinates captured by the driver responsible for delivery, images of unit number(s), address of the customer,
      type of address (building or house).
  * distance threshold (in meters) is the distance below which a delivery is considered to delivered in the right place. Meaning that the distance 
      between actual and captured coordinates must be less than or equal to the distance threshold. 
* for each stop in the database:
  * the unit numbers are extracted from the addresses given by the customers.
  * the haversine distance between the acutal coordinates and the captured coordinates by the driver is calculated. 
  * the unit numbers are extracted from the cropped text images.
  * the number of images captured by the driver depend on the type of the address. if the address is a building, then the building's unit number and the customer's apartment number must be captured by the driver. if a house, then a single image for unit number is needed. 
  * if (all unit numbers extracted from images captured by the driver match extracted units numbers extracted from customer's address) AND (the calculated distance is less than or equal to the threshold), then the delivery is considered correct, wrong otherwise.
    

## Data Formats:

* The database should be in this format. in this example, the database has two stops each of which with the corresponding information:


 ```
  database = {'stop_0':{'actual_lat_long':(43.569799, -79.504939),
                        'captured_lat_long':(43.568371, -79.507480),
                        'address': '412 - 4402 Street Name, Canada 1X1AB2',
                        'images':['4402_unit_image_directory','412_apartment_number_image_directory' ],
                        'type':'building'},
              'stop_1':{'actual_lat_long':(43.640656, -79.245704),
                        'captured_lat_long':(43.639965, -79.246326),
                        'address': '4435 Street Name, Canada 1X1AB2',
                        'images':['4435_unit_image_directory'],
                        'type':'house'}}
  ```


## Code Example:

```
import cv2 
from craft_text_detector import Craft
import numpy as np
import re
import easyocr
import haversine as hs


# creating a text localization object from CRAFT
craft = Craft(output_dir=None, crop_type="poly", cuda=True)
# creating a text recognition object from EasyOCR
reader = easyocr.Reader(['en'], gpu=True)


# sample dataset as a dictionary
database = {'stop_0':{'actual_lat_long':(43.569799, -79.504939),
                      'captured_lat_long':(43.568371, -79.507480),
                      'address': '412 - 4402 Street Name, Canada 1X1AB2',
                      'images':['4402_unit_image_directory','412_apartment_number_image_directory' ],
                      'type':'building'},
            'stop_1':{'actual_lat_long':(43.640656, -79.245704),
                      'captured_lat_long':(43.639965, -79.246326),
                      'address': '4435 Street Name, Canada 1X1AB2',
                      'images':['4435_unit_image_directory'],
                      'type':'house'}}


# creating an object from LastMile() class
lm = LastMile(text_localization_object = craft, 
              text_recognition_object = reader,
              delivery_dataset = database,
              distance_threshold=50)



# deciding on the correctness for each delivery in the database
results = lm.delivery_correctness()
print(results)
```




## Images References:
https://cdn.vectorstock.com/i/1000x1000/52/73/home-delivery-services-online-delivery-concept-vector-30605273.webp
