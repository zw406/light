#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
import re
import datetime
import requests
import warnings
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import plotly.express as px
import dash
import dash_bootstrap_components as dbc
from dash import html
from dash import dcc

pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', None)

restaurants_df = pd.read_csv('uk_data.csv',
                             encoding='utf8', low_memory=False)

# average price in euro
restaurants_df['minimum_range'] = pd.to_numeric(restaurants_df['price_range'].str.split('-').str[0].str.replace('€', '').str.replace(',', ''), errors='coerce')
restaurants_df['maximum_range'] = pd.to_numeric(restaurants_df['price_range'].str.split('-').str[1].str.replace('€', '').str.replace(',', ''), errors='coerce')
restaurants_df['avg_price'] = (restaurants_df['minimum_range'] + restaurants_df['maximum_range']) / 2

# drop the fields used for average_price calculation
restaurants_df.drop(['minimum_range', 'maximum_range'], axis=1, inplace=True)

restaurants_df.head(5)


# In[2]:


num_cities = restaurants_df['city'].nunique()
num_cities


# In[3]:


null_values_series = restaurants_df.drop(['avg_price'], axis=1).isnull().sum().where(lambda x : x > 0).dropna().astype('Int32')
print(null_values_series.to_string()) # to_string() removes the name and dtype from the output
msno.matrix(restaurants_df[null_values_series.index.tolist()], figsize=(15, 8));


# In[4]:


print(restaurants_df.shape)


# # Cities

# # city-ranking

# In[5]:


# Count of restaurant cities
top20_cities = restaurants_df['city'].value_counts()[:20]
top10_cities = restaurants_df['city'].value_counts()[:10].sort_values(ascending=True) 

city_ranking = go.Figure()
city_ranking.add_trace(go.Bar(
    x=top10_cities.values, 
    y=top10_cities.index,
    orientation='h',
    marker=dict(color=top10_cities.values, colorscale='portland', colorbar=dict(title='Count'))
))

city_ranking.update_layout(
    title='',
    template='plotly_white',
    width=500, height=397.4,margin=dict(l=0, r=0, t=50, b=0),
    xaxis=dict(
        title_standoff=30,
    ),
) 


# In[6]:


top20_cities_df = restaurants_df[restaurants_df['city'].isin(restaurants_df['city'].value_counts()[:20].index.to_list())]
# aggregating the data to find insights from the TripAdvisor dataset
agg_top20_cities_df = top20_cities_df.groupby(['country', 'city']).agg(
    total_restaurants=pd.NamedAgg(column='restaurant_link', aggfunc=np.size),
    mean_rating=pd.NamedAgg(column='avg_rating', aggfunc=np.mean),
    mean_food=pd.NamedAgg(column='food', aggfunc=np.mean),
    mean_service=pd.NamedAgg(column='service', aggfunc=np.mean),
    mean_values=pd.NamedAgg(column='value', aggfunc=np.mean),
    mean_atmosphere=pd.NamedAgg(column='atmosphere', aggfunc=np.mean),
    total_reviews=pd.NamedAgg(column='total_reviews_count', aggfunc=np.sum),
    mean_reviews_n=pd.NamedAgg(column='total_reviews_count', aggfunc=np.mean),
    median_reviews_n=pd.NamedAgg(column='total_reviews_count', aggfunc=np.median),
    mean_price=pd.NamedAgg(column='avg_price', aggfunc=np.mean),
    median_price=pd.NamedAgg(column='avg_price', aggfunc=np.median),
    open_days_per_week=pd.NamedAgg(column='open_days_per_week', aggfunc=np.mean),
    open_hours_per_week=pd.NamedAgg(column='open_hours_per_week', aggfunc=np.mean),
    working_shifts_per_week=pd.NamedAgg(column='working_shifts_per_week', aggfunc=np.mean),
    latitude=pd.NamedAgg(column='latitude', aggfunc=np.mean),
    longitude=pd.NamedAgg(column='longitude', aggfunc=np.mean)
).reset_index().sort_values(by='total_restaurants', ascending=False).head(20)

agg_top20_cities_df['median_reviews_n'] = agg_top20_cities_df['median_reviews_n'].astype('float')

for col in ['mean_rating', 'mean_reviews_n', 'mean_food', 'mean_service', 'mean_values', 'mean_atmosphere']:
    agg_top20_cities_df[col] = round(agg_top20_cities_df[col], 3)
# Bubble plot with the relationship between total_votes and avg_vote for the 20 most voted cities
fig_bubble_cities = go.Figure(data=go.Scatter(
    x=agg_top20_cities_df['total_restaurants'], 
    y=agg_top20_cities_df['mean_rating'],
    mode='markers', 
    marker=dict(
        size=agg_top20_cities_df['total_restaurants'].astype('float64') / 20,
        color=agg_top20_cities_df['mean_rating'],
        colorscale='portland',
        colorbar=dict(title='Mean Rating')
    ),
    text=agg_top20_cities_df['city'], 
    customdata=agg_top20_cities_df[['city', 'mean_food', 'mean_service', 'mean_values', 'mean_atmosphere', 'latitude', 'longitude']],
    hoverlabel=dict(namelength=0), # removes the trace number off to the side of the tooltip box
    hovertemplate='<b>%{text}</b>:<br>%{x:,} total restaurants<br>%{y:.2f} mean rating<br>Median Reviews: %{marker.color:.2f}<br>Mean Food: %{customdata[1]:.2f}<br>Mean Service: %{customdata[2]:.2f}<br>Mean Value: %{customdata[3]:.2f}<br>Mean Atmosphere: %{customdata[4]:.2f}<br>Latitude: %{customdata[5]:.2f}<br>Longitude: %{customdata[6]:.2f}'
))

# Add a trend line
z = np.polyfit(agg_top20_cities_df['total_restaurants'], agg_top20_cities_df['mean_rating'], 1)
p = np.poly1d(z)
fig_bubble_cities.add_trace(go.Scatter(
    x=agg_top20_cities_df['total_restaurants'], 
    y=p(agg_top20_cities_df['total_restaurants']),
    mode='lines',
    line=dict(color='black', width=1, dash='dash')
))

fig_bubble_cities.update_layout(title='', 
                                template='plotly_white',
                                legend=dict(yanchor='bottom', y=0, xanchor='left', x=0, font=dict(size=10), orientation='h'),
                                autosize=False, 
                                width=800, height=500)
fig_bubble_cities['layout']['xaxis']['title'] = 'Total Restaurants'
fig_bubble_cities['layout']['yaxis']['title'] = 'Mean Rating'
fig_bubble_cities.show()


# In[7]:


def round_decimals_up_or_down(direction:str, number:float, decimals:int=2):
    if not isinstance(decimals, int):
        raise TypeError('decimal places must be an integer')
    elif decimals < 0:
        raise ValueError('decimal places has to be 0 or more')
    elif decimals == 0:
        if direction == 'up':
            return math.ceil(number)
        elif direction == 'down':
            return math.floor(number)
        else:
            raise ValueError('direction needs to be up or down')
    factor = 10 ** decimals
    if direction == 'up':
        return math.ceil(number * factor) / factor
    elif direction == 'down':
        return math.floor(number * factor) / factor
    else:
        raise ValueError('direction needs to be up or down')


# In[8]:


top10_cities = agg_top20_cities_df['city'][:10]
city_agg_cols_dict = {'city': 'City', 'mean_rating': 'Rating', 'mean_food': 'Food', 'mean_service': 'Service', 'mean_values': 'Value',
                      'mean_atmosphere': 'Atmosphere'}
top10_cities_df = agg_top20_cities_df[agg_top20_cities_df['city'].isin(top10_cities)]
top10_cities_df = top10_cities_df[list(city_agg_cols_dict.keys())]
top10_cities_df.rename(columns=city_agg_cols_dict, inplace=True)

# melting the various categories, so that the line_polar graph can be easily called
top10_cities_df = top10_cities_df.melt(id_vars=['City'],
                                     value_vars=['Rating', 'Food', 'Service', 'Value', 'Atmosphere'],
                                     var_name='Category', value_name='AggValue')
top10_cities_df['AggValue'] = round(top10_cities_df['AggValue'], 3)

decimal_val = 1
max_cities_val = round_decimals_up_or_down(direction='up', number=top10_cities_df['AggValue'].max(), decimals=decimal_val)
min_cities_val = round_decimals_up_or_down(direction='down', number=top10_cities_df['AggValue'].min(), decimals=decimal_val)

# radar plot with plotly
radar_cities = px.line_polar(top10_cities_df, r='AggValue', range_r=[min_cities_val, max_cities_val],
                    theta='Category', color='City', line_close=True, width=480, height=600, color_discrete_sequence=px.colors.sequential.Magma, template='seaborn')

radar_cities.update_layout(title='', title_y=0.97,
                               width=667, height=397.44,margin=dict(l=20, r=0, t=50, b=0),
                               coloraxis_showscale=True)
radar_cities.show()


# # map

# In[9]:


fig_map = px.scatter_geo(data_frame=agg_top20_cities_df, lat='latitude', lon='longitude', color='mean_rating', hover_name='city',
                     size='total_restaurants', size_max=40, projection='natural earth', labels={'mean_rating': 'Mean Rating',},
                     color_continuous_scale='portland')

fig_map.update_geos(
    projection_type="natural earth",
    center=dict(lat=53.480759, lon=-2.242631), # Set the center to the UK (Manchester)
    lataxis=dict(range=[49, 60]), # Set the latitude range for the UK
    lonaxis=dict(range=[-10, 2]), # Set the longitude range for the UK
    showcountries=True,
    countrycolor="Black"
)

fig_map.update_layout(title='',
                   width=480, height=500, margin={'r':0, 'l':0, 'b':0, 'pad':0})

fig_map.show()


# In[ ]:


import dash
import dash_bootstrap_components as dbc
from dash import html
from dash import dcc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server=app.server

header_style = {"backgroundColor": "#9d292a", 
                "color": "#ffffff", 
                "fontWeight": "bold", 
                "fontSize": "120%",
                "text-align": "left"}

city_ranking_card = dbc.Card(
    [
        dbc.CardHeader("Number of Restaurants by City", style=header_style),
        dbc.CardBody(
            [
                dcc.Graph(figure=city_ranking)
            ]
        ),
    ],
)

# Define the new card
# Define the new card
new_card = dbc.Card([
    html.Div([
        html.H1("20 Cities", style={"color": "#ffffff", "fontWeight": "bold"}),
        
        html.Div([
            html.Hr(style={"borderTop": "1px solid white", "marginBottom": "20px"}), # add a horizontal line here
            html.H4("The top 20 cities by number of restaurants", 
                    style={"color": "#ffffff", "fontSize": "150%"}),
        ], style={"marginBottom": "15px"}), 
        
        html.Div([
            html.H1("171,000", style={"color": "#ffffff", "fontWeight": "bold"}),
            html.H3("Restaurants", style={"color": "#ffffff", "fontWeight": "bold","fontSize": "200%"}),
            html.Hr(style={"borderTop": "1px solid white", "marginBottom": "20px"}), # add a horizontal line here
            html.H4("Over 171,000 mean restaurant ratings on TripAdvisor to help you choose the best food desinations", 
                    style={"color": "#ffffff", "fontSize": "150%"}),
        ]),
    ], style={"backgroundColor": "#9d292a", "padding": "1rem", "height": "100%"}),
], style={"height": "480px"})





bubble_plot_card = dbc.Card(
    [
        dbc.CardHeader("Mean Rating and Total Restaurants of the 20 Cities with the Most Restaurants (Size by Number of Restaurants)", style=header_style),
        dbc.CardBody(
            [
                dcc.Graph(figure=fig_bubble_cities)
            ]
        ),
    ],
)

radar_plot_card = dbc.Card(
    [
        dbc.CardHeader("Radar Chart of Aggregate Ratings by City", style=header_style),
        dbc.CardBody(
            [
                dcc.Graph(figure=radar_cities)
            ]
        ),
    ],
)

map_card = dbc.Card(
    [
        dbc.CardHeader("Map of the 20 UK cities with the most restaurants (Size by Number of Restaurants)", style=header_style),
        dbc.CardBody(
            [
                dcc.Graph(figure=fig_map)
            ]
        ),
    ],
)

total_heading = dbc.Card(
    dbc.CardBody(
        [
            html.H1("Discovering the UK's Culinary Capitals", className="mb-0", style={"backgroundColor": "#9d292a", "color": "#ffffff", "fontWeight": "bold", "padding": "1.5rem 0", "padding-left": "1em", "text-align": "left"}),
            html.H2("A Foodie's Guide to the Best Dining Destinations Across British Cities", className="mb-0 text-muted", style={"text-align": "left", "padding-left": "1em","padding-top": "0.5em"}), # update style here
        ]
    ),
)

app.layout = dbc.Container(
    [
        total_heading,
        dbc.Row(
            [
                dbc.Col(new_card, md=2),  # add the new card here
                dbc.Col(city_ranking_card, md=5),
                dbc.Col(radar_plot_card, md=5),
            ],
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(bubble_plot_card, md=7),
                dbc.Col(map_card, md=5),
            ],
            className="mb-4",
        ),
    ],
    fluid=True,
)


if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False, port=8040)
