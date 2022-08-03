
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