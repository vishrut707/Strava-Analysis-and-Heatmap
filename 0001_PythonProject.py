import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import pandas as pd
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

'''Code to put all info into one dataframe ends here'''
############################################




type(all_activities)

#convert to kmph,km from m/s,m
df['average_speed'] = df['average_speed'] * 3.6
df['distance'] = df['distance'] / 1000

dfrun = df[df['type'] == 'Run']
dfride = df[df['type'] == 'Ride']

dfrunhr = dfrun.dropna(subset=["average_heartrate"])





df1 = dfrun[['distance','average_speed']]
df2 = dfrunhr[['distance','average_heartrate']]
df3 = dfrunhr[['average_heartrate','average_speed']]
df4 = dfrunhr[['distance','average_speed','average_heartrate']]

sns.jointplot(data=df1, x="distance", y="average_speed", kind='hex')
plt.show()

sns.jointplot(data=df2, x="distance", y="average_heartrate", kind='hex')
plt.show()

sns.jointplot(data=df3, x="average_speed", y="average_heartrate", kind='hex')
plt.show()

sns.kdeplot(dfrun['distance'])
sns.kdeplot(dfride['distance'])
plt.hist(dfride['distance'])

sns.kdeplot(dfrun['distance'], dfrun['average_speed'], shade=True)
sns.kdeplot(dfride['distance'], dfride['average_speed'], shade=True)

sns.lmplot(data=df3, x="average_speed", y="average_heartrate")
plt.show()

# df = df[df['distance'] > 10000]
# df = df[df['type'] == 'Ride']
# df
# df.reset_index

#df.columns

# df = df[['distance','average_speed']]
# df['average_speed'] = df['average_speed'] * 3.6

# sns.jointplot(data=df, x="distance", y="average_speed")
# plt.show()



# '''Create a 3D plot and set up the figure'''
# # Create a figure

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the data from the DataFrame
# ax.scatter(df4['distance'], df4['average_speed'], df4['average_heartrate'], c='b', marker='o')

# # Customize the plot as needed
# ax.set_xlabel('distance')
# ax.set_ylabel('average_speed')
# ax.set_zlabel('average_heartrate')
# ax.set_title('distance vs average_speed vs average_heartrate')

# # Display the plot
# plt.show()




'''Create a 3D plot and set up the figure'''
# Convert DataFrame columns to NumPy arrays
x = df4['distance'].values
y = df4['average_speed'].values
z = df4['average_heartrate'].values

# Create a grid of X and Y values
X, Y = np.meshgrid(x, y)

# Replace this with your actual Z values or a function to generate them
Z = np.sin(np.sqrt(X**2 + Y**2))

# Create a 3D plot and set up the figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a 3D surface plot
surface = ax.plot_surface(X, Y, Z, cmap='viridis')

# Customize the plot as needed
ax.set_xlabel('distance')
ax.set_ylabel('speed')
ax.set_zlabel('heartrate')
ax.set_title('3D Surface Plot')

# Add a color bar to the plot
fig.colorbar(surface, shrink=0.5, aspect=10)

# Display the plot
plt.show()


'''Create a 3D plot and set up the figure'''
# Create a 3D surface plot
surface = go.Surface(
    x=df4['distance'],
    y=df4['average_speed'],
    z=df4['average_heartrate']
)

layout = go.Layout(
    scene=dict(
        xaxis=dict(range=[min(df4['distance']), max(df4['distance'])], title='distance'),
        yaxis=dict(range=[min(df4['average_speed']), max(df4['average_speed'])], title='speed'),
        zaxis=dict(range=[min(df4['average_heartrate']), max(df4['average_heartrate'])], title='heartrate'),
    ),
    title='Interactive 3D Surface Plot'
)

fig = go.Figure(data=[surface], layout=layout)

# Display the interactive plot in a Jupyter Notebook or as a standalone HTML file
fig.show()



layout = go.Layout(
    scene=dict(
        xaxis=dict(title='distance'),
        yaxis=dict(title='speed'),
        zaxis=dict(title='heartrate'),
    ),
    title='Interactive 3D Surface Plot'
)

fig = go.Figure(data=[surface], layout=layout)

# Display the interactive plot in a Jupyter Notebook or as a standalone HTML file
fig.show()




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df4['average_speed'], df4['average_heartrate'], df4['distance'], c='b', marker='o')

# Customize the plot as needed
ax.set_xlabel('Average Speed (km/h)')
ax.set_ylabel('Average Heart Rate (bpm)')
ax.set_zlabel('Distance (km)')
ax.set_title('3D Scatter Plot of Running Data')

# Display the plot
plt.show()


fig = px.scatter_3d(df4, x='average_speed', y='average_heartrate', z='distance', 
                     title='Interactive 3D Scatter Plot of Running Data',
                     labels={'average_speed': 'Average Speed (km/h)', 'average_heartrate': 'Average Heart Rate (bpm)', 'distance': 'Distance (km)'})

# Display the interactive plot
fig.show()






#################################
# Define the bins for the three dimensions
x_bins = [3 ,4 ,5 ,6 , 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]  # Adjust the bin edges as needed
y_bins = [100, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210]  # Adjust the bin edges as needed
z_bins = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]  # Adjust the bin edges as needed

# Create a 3D histogram
histogram = go.Histogram2d(x=df4['average_speed'], y=df4['average_heartrate'], z=df4['distance'],
                            autobinx=False, xbins=dict(start=min(x_bins), end=max(x_bins)),
                            autobiny=False, ybins=dict(start=min(y_bins), end=max(y_bins)),
                            colorscale='Viridis', showscale=True)

layout = go.Layout(
    scene=dict(
        xaxis=dict(title='Average Speed (km/h)'),
        yaxis=dict(title='Average Heart Rate (bpm)'),
        zaxis=dict(title='Distance (km)'),
    ),
    title='3D Heatmap of Running Data'
)

fig = go.Figure(data=[histogram], layout=layout)

# Display the interactive plot in a Jupyter Notebook or as a standalone HTML file
fig.show()




# Create a 3D histogram with auto-binning
histogram = go.Histogram2d(x=df4['average_speed'], y=df4['average_heartrate'], z=df4['distance'],
                            colorscale='Viridis', showscale=True)

layout = go.Layout(
    scene=dict(
        xaxis=dict(title='Average Speed (km/h)'),
        yaxis=dict(title='Average Heart Rate (bpm)'),
        zaxis=dict(title='Distance (km)'),
    ),
    title='3D Heatmap of Running Data'
)

fig = go.Figure(data=[histogram], layout=layout)

# Display the interactive plot in a Jupyter Notebook or as a standalone HTML file
fig.show()