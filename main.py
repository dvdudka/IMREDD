#   Python script for forcasted weather data treatment
#   Source : weatherbit
#   API documentation link : https://www.weatherbit.io/api/weather-forecast-120-hour"
#   Content : Hourly forecast for 48 hours
#   Developed by : Dmytro DUDKA
#   Energy Engineer - Data Analyst
#   Location : IMREDD, Université Côte d'Azur, Nice, France
#   Date : 27/09/2020

# ========================================== importing the requests library ==========================================
from scipy import stats
import scipy
from windrose import WindroseAxes
import requests
import io
import json
from urllib.request import urlopen
from wordcloud import WordCloud, STOPWORDS
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.cm as cm
# matplotlib.use('Agg')
import matplotlib.colors as colors
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import streamlit as st
import warnings
warnings.filterwarnings(action='once')
# chose the style : dark_background, seaborn-white dark_background, fivethirtyeight, seaborn-white, seaborn, classic, bmh
style.use('fivethirtyeight')

# ========================================= function WeatherForecast ===========================================


def WeatherForecast():
    # set longitude and latitude of the desired city
    Nice = "&lat=43.7102&lon=7.2620&"
    #Nice = "&lat=43.7102&lon=7.2620"
    location = Nice  # set a location : name of the city
    #key = "5ea6a77dfbed4e439a05ac48203e207b&hours=48"
    # API key
    key = "c28cfb7263e6478fb3761f84569021ed&hours=48"
    # base url
    url_0 = "https://api.weatherbit.io/v2.0/forecast/hourly?"
    # url = "https://api.weatherbit.io/v2.0/forecast/hourly?&lat=43.7102&lon=7.2620&key=ec60e11c69454c4cbf4147806230f2a6&hours=48"
    url = url_0+location+key
    url = "https://api.weatherbit.io/v2.0/forecast/hourly?&lat=43.7102&lon=7.2620&key=bf9ec95319174855912aaeb08dc0638b&hours=48"

    print(url)
    json_obj = urlopen(url)
    # data draft dans le format JSON
    data = json.load(json_obj)
    print(data)
    # out_file = open('data.json', 'w')                                                                    # new JSON which stores changes in initial JSON
    with open('data.json', 'w') as f:
        # create a JSON file in the same directory
        json.dump(data, f)

    wind_cdir = []                      # abreviated wind direction
    rh = []                             # relative humidity [%]
    pod = []                            # part of day (d = day / n = night)
    timestamp_utc = []                  # timestamp at UTC time
    pres = []                           # pressure [mb]
    solar_rad = []                      # estimated solar radiation [W/m2]
    ozone = []                          # average ozone [Dobson units]
    weather = []                        # weather description in words
    wind_gust_spd = []                  # wind gust speed [m/s]
    timestamp_local = []                # timestamp at local time
    snow_depth = []                     # snow depth [mm]
    clouds = []                         # cloud coverage [%]
    ts = []                             # Unix timestamp at UTC time
    wind_spd = []                       # wind speed [m/s]
    pop = []                            # probability of precipitation
    wind_cdir_full = []                 # verbal wind direction
    slp = []                            # sea level pressure [mb]
    dni = []                            # direct normal solar irradiance [W/m2]  [Clear sky]
    dewpt = []                          # dew point [degree Celcius]
    snow = []                           # accumulated snowfall [mm]
    uv = []                             # UV index [0-11+]
    wind_dir = []                       # wind direction [degrees]
    precip = []                         # accumulated liquid equivalent precipitation [mm]
    vis = []                            # visibility [km]
    # diffuse horizontal solar irradiance [W/m2] [Clear sky]
    dhi = []
    # apparent / 'feels like' temperature [degree Celcius]
    app_temp = []
    # [DEPRECATED] forecast valid hour UTC (YYYY-MM-DD:HH)
    datetime = []
    temp = []                           # temperature [degree Celcius]
    ghi = []                            # global horizontal solar irradiance [W/m2]
    # high-level (> 5km AGL) cloud coverage [%]
    clouds_hi = []
    # Mid-level (3-5km AGL) cloud coverage [%]
    clouds_mid = []
    # Low-level (0-3km AGL) cloud coverage [%]
    clouds_low = []
    columns_hourly = []
    # create lists of the parameters
    for item in data['data']:
        wind_cdir.append(item['wind_cdir'])
        rh.append(item['rh'])
        pod.append(item['pod'])
        timestamp_utc.append(item['timestamp_utc'])
        pres.append(item['pres'])
        solar_rad.append(item['solar_rad'])
        ozone.append(item['ozone'])
        weather.append(item['weather']['description'])
        wind_gust_spd.append(item['wind_gust_spd'])
        timestamp_local.append(item['timestamp_local'])
        snow_depth.append(item['snow_depth'])
        clouds.append(item['clouds'])
        ts.append(item['ts'])
        wind_spd.append(item['wind_spd'])
        pop.append(item['pop'])
        wind_cdir_full.append(item['wind_cdir_full'])
        slp.append(item['slp'])
        dni.append(item['dni'])
        dewpt.append(item['dewpt'])
        snow.append(item['snow'])
        uv.append(item['uv'])
        wind_dir.append(item['wind_dir'])
        clouds_hi.append(item['clouds_hi'])
        precip.append(item['precip'])
        vis.append(item['vis'])
        dhi.append(item['dhi'])
        app_temp.append(item['app_temp'])
        datetime.append(item['datetime'])
        temp.append(item['temp'])
        ghi.append(item['ghi'])
        clouds_mid.append(item['clouds_mid'])
        clouds_low.append(item['clouds_low'])
    data_list = [timestamp_utc, timestamp_local, ts, datetime, pod, pres, slp, temp, app_temp, dewpt, solar_rad, dni, dhi, ghi, uv,
                 clouds, clouds_low, clouds_mid, clouds_hi, vis, wind_cdir, wind_spd, wind_gust_spd, wind_dir, wind_cdir_full,
                 rh, precip, pop, ozone, weather, snow_depth, snow]
    print(data_list)
    for j in range(1, 49):
        ch = 'Hour ' + str(j)
        if len(ch) < 7:
            ch = 'Hour ' + '0' + str(j)
            columns_hourly.append(ch)
        else:
            columns_hourly.append(ch)
    pdDataList = pd.DataFrame(data_list, columns=columns_hourly,
                              index=['timestamp_utc', 'timestamp_local', 'ts', 'datetime', 'pod', 'pres', 'slp', 'temp', 'app_temp', 'dewpt',
                                     'solar_rad', 'dni', 'dhi', 'ghi', 'uv', 'clouds', 'clouds_low', 'clouds_mid', 'clouds_hi', 'vis',
                                     'wind_cdir', 'wind_spd', 'wind_gust_spd', 'wind_dir', 'wind_cdir_full', 'rh', 'precip', 'pop', 'ozone',
                                     'weather', 'snow_depth', 'snow'])                                            # create a Data frame of the parameters in the list

    # create numpy array by means of build-in library NUMPY
    numpy_array = np.array(pdDataList)
    pdDataListT = numpy_array.T
    dfObj_HourlyT = pd.DataFrame(pdDataListT, index=columns_hourly,
                                 columns=['timestamp_utc', 'timestamp_local', 'ts', 'datetime', 'pod', 'pres', 'slp', 'temp', 'app_temp', 'dewpt',
                                          'solar_rad', 'dni', 'dhi', 'ghi', 'uv', 'clouds', 'clouds_low', 'clouds_mid', 'clouds_hi', 'vis',
                                          'wind_cdir', 'wind_spd', 'wind_gust_spd', 'wind_dir', 'wind_cdir_full', 'rh', 'precip', 'pop', 'ozone',
                                          'weather', 'snow_depth', 'snow'])                                            # create table where you can filter, select data in the columns

    # Read data with Streamlit
    st.title("Weather forecast @ Nice, France")
    st.info("Open Dat a Source : www.weatherbit.io")
    st.subheader(
        "API documentation link : https://www.weatherbit.io/api/weather-forecast-120-hour")
    st.info("Content: Hourly weather forecast API - 120 hours, JSON format, ")
    st.subheader("Measured parameters :")
    st.text("Air temperature, feels-like temperature, dew point temperature, athmospheric pressure, relative humidity, wind speed, wind gusts, wind direction, "
            "solar irradiation, UV index, clouds on different altitude, sky condition, visibility, precipitation, snow, ozone")
    st.subheader('Table : Hourly weather forecast for 2 days')
    # Create a table with parameters
    st.dataframe(dfObj_HourlyT)
    column = st.selectbox('Select the column which you want to display :',
                          dfObj_HourlyT.columns)              # chose column to plot a linechart
    # plot a line chart of selected parameter
    st.line_chart(dfObj_HourlyT[column])
    # plot a bar chart of selected parameter
    st.bar_chart(dfObj_HourlyT[column])
    # plot an area chart of selected parameter
    st.area_chart(dfObj_HourlyT[column])

    # ================================================== Visualisation ====================================================

    mycolors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown',
                'tab:grey', 'tab:pink', 'tab:olive']  # create a lsit of colors used in Python
    columns = ['feels like temperature',
               'real temperature', 'dew point temperature']

    # Create various figures and graphs by means of powerful Matplotlib library
    fig, ax = plt.subplots(1, 1, figsize=(16, 9), dpi=60)
    ax.fill_between(datetime, y1=temp, y2=0,
                    label=columns[1], alpha=0.5, color=mycolors[1], linewidth=2)
    ax.fill_between(datetime, y1=app_temp, y2=0,
                    label=columns[0], alpha=0.5, color=mycolors[0], linewidth=2)
    ax.fill_between(datetime, y1=dewpt, y2=0,
                    label=columns[2], alpha=0.5, color=mycolors[2], linewidth=2)
    ax.set_title('Forecasted hourly temperature records',
                 fontsize=18)                                       # Decorations
    ax.legend(loc='best', fontsize=12)
    plt.xticks(datetime[::6], fontsize=10, horizontalalignment='center')
    plt.xlabel('Date, Time')
    plt.ylabel('Temperature, CEL')
    # Draw Tick lines
    for y in np.arange(2.5, 30.0, 2.5):
        plt.hlines(y, xmin=0, xmax=len(datetime), colors='black',
                   alpha=0.3, linestyles="--", lw=0.5)
    # Lighten borders
    plt.gca().spines["top"].set_alpha(0)
    # Lighten borders
    plt.gca().spines["bottom"].set_alpha(.3)
    # Lighten borders
    plt.gca().spines["right"].set_alpha(0)
    # Lighten borders
    plt.gca().spines["left"].set_alpha(.3)
    plt.show()

    # Plot subplots of timeseries : air temperature, air pressure, ozone level, solar irradiation
    # Create subplots with a function "subplot2grid" where the size of a plot and subplots are predifined by an engineer
    ax1 = plt.subplot2grid((12, 2), (0, 0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((12, 2), (0, 1), rowspan=5, colspan=1)
    ax3 = plt.subplot2grid((12, 2), (7, 0), rowspan=5, colspan=1)
    ax4 = plt.subplot2grid((12, 2), (7, 1), rowspan=5, colspan=1)
    # Add grid to the subplots
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()
    # subplot 1 : Forecasted dew point temperature hourly records
    ax1.plot(datetime, temp, color=mycolors[0], label="real temperature")
    ax1.plot(datetime, app_temp,
             color=mycolors[1], label="feels like temperature")
    ax1.plot(datetime, dewpt, color=mycolors[2], label="dew point temperature")
    ax1.set_title('Forecasted temperature hourly records', fontsize=20)
    ax1.set_xlabel('Date, Time', fontsize=15)
    ax1.set_ylabel('Temperature, °C', fontsize=15)
    ax1.legend()
    # plt.locator_params(numticks=5)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(3))
    # subplot 2 : Forecasted hourly athmospheric pressure records
    ax2.plot(datetime, pres, color=mycolors[0], label="athmospheric pressure")
    ax2.set_title(
        'Forecasted hourly athmospheric pressure records', fontsize=20)
    # plt.locator_params(numticks=5)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax2.set_xlabel('Date, Time', fontsize=15)
    ax2.set_ylabel('Pressure, mbar', fontsize=15)
    # subplot 3 : Forecasted hourly ozone records
    ax3.plot(datetime, ozone, color=mycolors[1], label="ozone")
    ax3.set_title('Forecasted hourly ozone level', fontsize=20)
    # plt.locator_params(numticks=5)
    ax3.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax3.set_xlabel('Date, Time', fontsize=15)
    ax3.set_ylabel('Ozone, DU', fontsize=15)
    # subplot 4 : Forecasted solar irradiation hourly records
    ax4.set_title('Solar irradiation', fontsize=20)
    # plt.locator_params(numticks=5)
    ax4.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax4.set_xlabel('Date, Time', fontsize=15)
    ax4.set_ylabel('Solar irradiation, W/m2', fontsize=15)
    ax4.plot(datetime, solar_rad, color=mycolors[0], label="solar irradiation")
    ax4.plot(datetime, dni, color=mycolors[1],
             label="direct normal solar irradiance")
    ax4.plot(datetime, ghi, color=mycolors[2],
             label="global horizontal solar irradiance")
    ax4.set_title('Solar activity', fontsize=20)
    ax4.legend()
    ax44 = ax4.twinx()
    ax44.plot(datetime, uv, color=mycolors[3], label="UV index")
    ax44.set_ylabel('UV index', color=mycolors[3], fontsize=15)
    ax4.xaxis.set_major_locator(plt.MaxNLocator(3))
    plt.show()  # Show the entire figure with subplots

    # Plot subplots of timeseries : clouds, visibility, humidity, precipitation
    # Create subplots with a function "subplot2grid" where the size of a plot and subplots are predifined by an engineer
    ax1 = plt.subplot2grid((12, 2), (0, 0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((12, 2), (0, 1), rowspan=5, colspan=1)
    ax3 = plt.subplot2grid((12, 2), (7, 0), rowspan=5, colspan=1)
    ax4 = plt.subplot2grid((12, 2), (7, 1), rowspan=5, colspan=1)
    # Add grid to the subplots
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()
    # subplot 1 : Forecasted dew point temperature hourly records
    ax1.plot(datetime, clouds_low, color=mycolors[0], label="low clouds")
    ax1.plot(datetime, clouds_mid, color=mycolors[1], label="middle clouds")
    ax1.plot(datetime, clouds_hi, color=mycolors[2], label="high clouds")
    ax1.set_title('Clouds on different altitude', fontsize=20)
    ax1.set_xlabel('Date, Time', fontsize=15)
    ax1.set_ylabel('Cloudness, %', fontsize=15)
    ax1.legend()
    # plt.locator_params(numticks=5)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(3))
    # subplot 2 : Forecasted hourly athmospheric pressure records
    ax2.plot(datetime, vis, color=mycolors[1], label="visibility")
    ax2.set_title('Forecasted visibility', fontsize=20)
    # plt.locator_params(numticks=5)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax2.set_xlabel('Date, Time', fontsize=15)
    ax2.set_ylabel('Visibility, km', fontsize=15)
    # subplot 3 : Forecasted hourly humidity records
    ax3.bar(datetime, rh, color=mycolors[1], label="humidity")
    ax3.set_title('Forecasted hourly humidity records', fontsize=20)
    # plt.locator_params(numticks=5)
    ax3.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax3.set_xlabel('Date, Time', fontsize=15)
    ax3.set_ylabel('Humidity, %', fontsize=15)
    # subplot 4 : Forecasted wind speed and wind direction hourly records
    norm_cdf = scipy.stats.norm.cdf(precip)
    t_dist = scipy.stats.t(20)
    # .plot .bar .scatter .fill_between
    ax4.fill_between(datetime, precip,
                     color=mycolors[0], label="precipitation")
    ax4.set_title('Forecasted hourly precipitation', fontsize=20)
    # plt.locator_params(numticks=5)
    ax4.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax4.set_xlabel('Date, Time', fontsize=15)
    ax4.set_ylabel('Precipitation, mm', fontsize=15)
    ax44 = ax4.twinx()
    ax44.plot(datetime, norm_cdf, color=mycolors[2], label="Cumulative rain")
    ax44.set_ylabel('Cumulative distribution of precipitation',
                    color=mycolors[2], fontsize=15)
    ax4.xaxis.set_major_locator(plt.MaxNLocator(3))
    plt.show()  # Show the entire figure with subplots
    # Show the entire figure with subplots

    s = pd.Series(precip)
    cumulativeProduct = s.cumprod()
    print(cumulativeProduct)
    # Create various figures and graphs by means of powerful Matplotlib library
    fig, ax = plt.subplots(1, 1, figsize=(16, 9), dpi=60)
    ax.fill_between(datetime, y1=cumulativeProduct, y2=0,
                    label=columns[1], alpha=0.5, color=mycolors[1], linewidth=2)
    ax.set_title('Cumulative precipitation : 48 hours forecast', fontsize=20)
    #ax.set_xlabel('Date, Time', fontsize=15)
    #ax.set_ylabel('Cumulative rain level, mm', fontsize=15)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    plt.xticks(datetime[::6], fontsize=10, horizontalalignment='center')
    plt.xlabel('Date, Time')
    plt.ylabel('Cumulative rain level, mm')
    plt.show()

    # Plot Wind Rose by using WindroseAxes library
    df = pd.DataFrame({"speed": wind_spd, "direction": wind_dir})
    ax1 = WindroseAxes.from_ax()
    ax1.bar(df.direction, df.speed, normed=True, opening=0.8,
            edgecolor='white', capstyle='butt')       # 'butt', 'round', 'projecting'
    ax1.set_legend()
    plt.title("Wind Rose - 2 days hourly forecast")
    # Plot Streamlit
    st.pyplot(plt)
    ax2 = WindroseAxes.from_ax()
    # 'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink', 'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper']
    ax2.contourf(df.direction, df.speed, bins=np.arange(0, 8, 1), cmap=cm.cool)
    ax2.contour(df.direction, df.speed,
                bins=np.arange(0, 8, 1), colors='black')
    ax2.set_legend()
    plt.title("Wind Rose - 2 days hourly forecast")
    plt.show()
    st.pyplot(plt)

    # Create a text cloud : description of the sky condition [forecast for 1 week]
    st.subheader("Text cloud")
    st.text("Hourly forecast | Desription of the sky condition")
    comment_words = ''
    stopwords = set(STOPWORDS)
    # iterate through the csv file
    for val in weather:
        # typecaste each val to string
        val = str(val)
        # split the value
        tokens = val.split()
        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        comment_words += " ".join(tokens) + " "
    wordcloud = WordCloud(width=500, height=500, background_color='azure',
                          stopwords=stopwords, min_font_size=8).generate(comment_words)
    # plot the WordCloud image
    plt.figure(figsize=(6, 6), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
    # Plot within Streamlit
    st.pyplot(plt)

# ================================================== Call the function ====================================================


# Call the function WeatherForecast()
WeatherForecast()
