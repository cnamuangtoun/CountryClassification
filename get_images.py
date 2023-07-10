import re
import random
import shapefile
import os
from tqdm import tqdm
from shapely.geometry import shape, Point
import requests
import json
from csv import writer



def random_point_in_country(shp_location, country_name):
    shapes = shapefile.Reader(shp_location) # reading shapefile with pyshp library
    country = [s for s in shapes.records() if country_name in s][0] # getting feature(s) that match the country name
    country_id = int(re.findall(r'\d+', str(country))[0]) # getting feature(s)'s id of that match

    shapeRecs = shapes.shapeRecords()
    feature = shapeRecs[country_id].shape.__geo_interface__

    shp_geom = shape(feature)

    minx, miny, maxx, maxy = shp_geom.bounds
    while True:
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if shp_geom.contains(p):
            return p.x, p.y
        

url = 'https://maps.googleapis.com/maps/api/streetview'
url2 = 'https://maps.googleapis.com/maps/api/streetview/metadata?'
path = './images/'

for name in os.listdir(path):
    if name.startswith('.'):
         continue
    coor = []
    i = 0
    while i < 500:
        x, y = random_point_in_country('World_Countries/World_Countries.shp', name)
        params = {
                'key': '', #search google static map key
                'size': '640x640',
                'location': f'{y:.3f},{x:.3f}',
                }
        response = requests.get(url2, params)
        if json.loads(response.content)['status'] == "OK":
                res = requests.get(url, params)
                coor.append([x,y,f'{name}_{i}'])
                with open(os.path.join(os.path.join(path, name),f'{name}_{i}.png'), "wb") as file:
                        file.write(res.content)
                i += 1
    with open(os.path.join(path, os.path.join("Dataset", name, f'{name}.csv')), 'w') as file:
            csv_writer = writer(file)
            for line in coor:
                csv_writer.writerow(line)