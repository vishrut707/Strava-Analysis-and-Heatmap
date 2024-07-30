import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import pandas as pd
import polyline
import math

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
import plotly.express as px



auth_url = "https://www.strava.com/oauth/token" #api url to get auth token using refresh token
activites_url = "https://www.strava.com/api/v3/athlete/activities" #api url to get list of activities

payload = {
    'client_id': "",
    'client_secret': '',
    'refresh_token': '',
    'grant_type': "refresh_token",
    'f': 'json'
}

print("Requesting Token...\n")
res = requests.post(auth_url, data=payload, verify=False)
access_token = res.json()['access_token']

print("Access Token = {}\n".format(access_token))
header = {'Authorization': 'Bearer ' + access_token}

# The first loop, request_page_number will be set to one, so it requests the first page. Increment this number after
# each request, so the next time we request the second page, then third, and so on...
request_page_num = 1
all_activities = []

while True:
    param = {'per_page': 200, 'page': request_page_num}
    # initial request, where we request the first page of activities
    my_dataset = requests.get(activites_url, headers=header, params=param).json()

    # check the response to make sure it is not empty. If it is empty, that means there is no more data left. So if you have
    # 1000 activities, on the 6th request, where we request page 6, there would be no more data left, so we will break out of the loop
    if len(my_dataset) == 0:
        print("breaking out of while loop because the response is zero, which means there must be no more activities\n")
        break

    # if the all_activities list is already populated, that means we want to add additional data to it via extend.
    if all_activities:
        print("all_activities is populated")
        all_activities.extend(my_dataset)

    # if the all_activities is empty, this is the first time adding data so we just set it equal to my_dataset
    else:
        print("all_activities is NOT populated")
        all_activities = my_dataset

    request_page_num += 1

print("There are {} activities in total\n".format(len(all_activities)))
for count, activity in enumerate(all_activities):
    print(activity["name"])
    print(count)

#make a dataframe with all the info
df = pd.DataFrame(all_activities)

print(df.head())
print(df.columns)
'''Code to put all info into one dataframe ends here'''
############################################

all_decoded_coordinates = []
for mapitem in df['map']:
    if mapitem['summary_polyline'] == '':
        # print('empty')
        continue
    else:
        encoded_polyline = mapitem['summary_polyline']
        all_decoded_coordinates = all_decoded_coordinates + polyline.decode(encoded_polyline)


leastdist = 5
for i in range (len(all_decoded_coordinates) - 1):
    if abs(math.dist(all_decoded_coordinates[i], all_decoded_coordinates[i+1])) < leastdist and math.dist(all_decoded_coordinates[i], all_decoded_coordinates[i+1]) != 0:
        leastdist = abs(math.dist(all_decoded_coordinates[i], all_decoded_coordinates[i+1]))


all_decoded_coordinates = []
for mapitem in df['map']:
    decoded_points = []
    if mapitem['summary_polyline'] == '':
        print('empty')
    else:
        encoded_polyline = mapitem['summary_polyline']
        decoded_points = polyline.decode(encoded_polyline)
        '''<code to add filler points to the decoded_points list>'''
        ######
        for i in range (len(decoded_points) - 1):
            distance = math.dist(decoded_points[i], decoded_points[i+1])
            if abs(distance) > leastdist and distance != 0:
                numofpoints = int(distance/leastdist)
                for j in range (numofpoints):
                    newpoint = [decoded_points[i][0] + (decoded_points[i+1][0] - decoded_points[i][0]) * (j/numofpoints), decoded_points[i][1] + (decoded_points[i+1][1] - decoded_points[i][1]) * (j/numofpoints)]
                    decoded_points.append(newpoint)
        ######
        all_decoded_coordinates = all_decoded_coordinates + decoded_points






import folium
from folium.plugins import HeatMap

# Create a map centered around your data
m = folium.Map(location=[17.5448901,78.572163], zoom_start=16)

# Create a heatmap layer using the decoded coordinates
HeatMap(all_decoded_coordinates, radius=1, blur=2).add_to(m)

# Save or display the heatmap
m.save('heatmap.html')


