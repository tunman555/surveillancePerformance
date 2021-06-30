from pandas.core.base import SpecificationError
import numpy as np
import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
import json
from haversine import haversine,Unit
import random 
from PIL import Image
import plotly.graph_objs as go
from numpy import pi,sin,cos

l = haversine([99,100],[100,100],unit=Unit.NAUTICAL_MILES)
t = np.linspace(0, 2*pi, 100)

def dms2dd(Lat,Lon):
    Lat_dd = int(Lat[:2]) + float(int(Lat[2:4]))/60 + float(int(Lat[4:]))/3600
    Lon_dd = int(Lon[:3]) + float(int(Lon[3:5]))/60 + float(int(Lon[5:]))/3600
    return Lat_dd,Lon_dd

@st.cache
def create_data():
    df = pd.DataFrame()

    f = open ('./site_coor.json', "r")
    js = json.load(f)

    site_dms={}
    for i in js :
        site_dms[i] = dms2dd(js[i]['lat'],js[i]['lon'])
    cov = [[0.75, 0], [0, 0.75]]

    j=0
    for i in site_dms :
        mean = site_dms[i]
        x,y =  np.random.multivariate_normal(mean, cov, 100000).T
        exec(str(i) +"_lat = j ")
        exec(str(i) +"_lon = j+1")
        df[str(i)+"_lat"] = pd.Series(x)
        df[str(i)+"_lon"] = pd.Series(y)
        j=j+2
        
    a_file = open("rep_dms.json", "r")
    rep_dms = json.load(a_file)
    a_file.close()

    return df,site_dms,rep_dms
    
df,site_dms,rep_dms = create_data()


def create_pipeline(list_site):
    pipeline = []
    for index, transformation in enumerate(list_site):
        if transformation:
            pipeline.append(index)
    return pipeline

def random_height(pipeline):
    rand_alt = []
    n =  len(pipeline)
    for i in range(n):
        x = round(random.uniform(300,320))
        rand_alt.append(x)
    return rand_alt

def random_farth():
    rand_farth = {}
    k  = list(site_dms.keys())
    for i in range(len(k)):
        rand_farth[k[i]] = round(random.uniform(230,250))
    return rand_farth
all_site_farth = random_farth()

@st.cache
def load_status():
    df = pd.read_csv('./src/site_status.txt',sep=';')
    df = df.loc[df['site_identification'].str.contains('SSR')]
    return df

def create_point(site,range_site):
    R = range_site
    center_lon = site[1]
    center_lat = site[0]

    circle_lon =center_lon + R*cos(t)
    circle_lat =center_lat +  R*sin(t)
    circle_lon = list(circle_lon)
    circle_lat = list(circle_lat)
    circle_lon.append(None)
    circle_lat.append(None)
    return circle_lat,circle_lon

def reporting_plot(rep_point):
    l = haversine([99,100],[100,100],unit=Unit.NAUTICAL_MILES)
    t = np.linspace(0, 2*pi, 100)

    report_p = rep_dms[rep_point]
    receiver = []
    k = list(site_dms.keys())
    for i in k :
        if haversine(report_p,site_dms[i], unit=Unit.NAUTICAL_MILES) < all_site_farth[i] :
            receiver.append(i)

    cover_lat,cover_lon = [],[]
    for i in receiver:
        tmp_site = site_dms[i]
        tmp_range_site = all_site_farth[i]
        rl = tmp_range_site / l
        cover_lat = cover_lat + create_point(tmp_site,rl)[0]
        cover_lon = cover_lon + create_point(tmp_site,rl)[1]
    del cover_lat[-1]
    del cover_lon[-1]

    fig = go.Figure(go.Scattermapbox(
        mode = "lines", fill = "toself",
        lon = cover_lon,
        lat = cover_lat,
        hoverinfo='none'))

    fig.add_trace(go.Scattermapbox(
        lat=[report_p[0]],
        lon=[report_p[1]],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=5
        ),
        text=[report_p],
    ))
    fig.update_layout(
        mapbox = {
                    'accesstoken':"pk.eyJ1IjoidHVubWFuNTU1IiwiYSI6ImNrcHA5MGhzajBsMXIydm1ubzE1YmZhbHIifQ.8rPLrf8FYUq0ZkDrAF6_vA",
                    'style': "light",
                    'center': {'lon': report_p[1], 'lat': report_p[0]}, 'zoom': 4.5},
        showlegend = False,
        margin = {'l':0, 'r':0, 'b':0, 't':0})
    return fig,receiver

def visual(list_site,rpoint):
    image = Image.open('./src/pic.jpg')
    st.image(image)
    st.title('Surveillance performance datavisualization')
    st.markdown("""
    * The glowing is refered to the location of Secondary RADAR site.
    * Your plots will appear below.
    """)

    pipeline = create_pipeline(list_site)

    lat_site,lon_site = [],[]
    farthest_p ={}
    v = list(site_dms.values())
    k = list(site_dms.keys())
    for i in pipeline:
        mean = v[i]
        tmp_max = 0
        tmp_lat = df[df.columns[2*i]].to_list()
        tmp_lon = df[df.columns[2*i + 1]].to_list()
        
        for j in range(10000):
            point=(tmp_lat[j],tmp_lon[j])
            if haversine(mean,point, unit=Unit.NAUTICAL_MILES) > 250 :
                tmp_lat[j] =  mean[0]
                tmp_lon[j] =  mean[1]
            elif haversine(mean,point, unit=Unit.NAUTICAL_MILES) > tmp_max :
                tmp_max = haversine(mean, point, unit=Unit.NAUTICAL_MILES)
        farthest_p[k[i]] = tmp_max
        mean_x = [mean[0]]*20
        mean_y = [mean[1]]*20
        lat_site = lat_site + tmp_lat + mean_x
        lon_site = lon_site + tmp_lon + mean_y
    
    px.set_mapbox_access_token("pk.eyJ1IjoidHVubWFuNTU1IiwiYSI6ImNrcHA5MGhzajBsMXIydm1ubzE1YmZhbHIifQ.8rPLrf8FYUq0ZkDrAF6_vA")
    fig = ff.create_hexbin_mapbox(
        #data_frame=df, lat="DMASSR_lat", lon="DMASSR_lon",
        lat =lat_site,lon =lon_site,mapbox_style ='light',
        nx_hexagon=75, opacity=0.4,min_count=1,width=480,height=600
    )

    fig.update_layout(margin=dict(b=0, t=0, l=0, r=0),hovermode=False,coloraxis_showscale=False)
    st.plotly_chart(fig,use_container_width=True)
    
    rand_alt = random_height(pipeline)
    farth_list = list(farthest_p.values())
    area_cov = [ 3.14*1.852*(i*0.9)**2 for i in farth_list]

    df_stat = pd.DataFrame.from_dict(farthest_p,orient = 'index',columns = ['Farthest Range (NM)'])
    df_stat['Area (km^2)'] = area_cov
    df_stat['Altitude at farthest (ft)'] = rand_alt

    st.markdown("# The summary of SSR surveillance.")
    st.table(df_stat)
    st.markdown("The area was calculated from 90% from the farthest Range")

    #Reporting Point part
    rpoint_key = list(rep_dms.keys())
    if rpoint != "":
        if rpoint in rpoint_key :
            st.markdown(" # Reporting Point VS SSR site !")
            st.markdown(" This part is showing How many SSR covering your selected Reporting Point")
            rep_plot,receiver  = reporting_plot(rpoint) 
            st.markdown(f"{', '.join(receiver)} is covering your reporting point")
            st.plotly_chart(rep_plot)
        else :
            st.warning(" Your selected point doesn't exist !")
    else :
        st.warning("'You haven't selected any point :P")
    st.beta_expander("See Chart")
    
    # Statistical part 
    df_down = load_status()
    st.markdown("# Site status changed.")
    st.markdown(" The site status change has occured due to many reasons such as Preventive Maintainance (PM), Missing Pulse repetition frequency(PRF)"
                " and etc.")
    st.markdown("The table below this is the example of site_change_status table in the TMCS database")
    st.dataframe(df_down.head(5))

    st.markdown("## The downtime analytic is coming very soon !")

    with st.beta_expander("See notes"):
        st.markdown("To see more information of the concept of multivariate normal distribution on https://en.wikipedia.org/wiki/Multivariate_normal_distribution")
        st.markdown("To see more about the Radiation pattern on https://en.wikipedia.org/wiki/Radiation_pattern ")
        st.markdown("To see more about the PRF https://en.wikipedia.org/wiki/Radar_signal_characteristics#Pulse_repetition_frequency_(PRF)")

def main():
    p_holder1 = st.empty()
    p_holder2 = st.empty()
    p_holder3 = st.empty()
    p_holder1.markdown("# SSR Performance Data Visualization.\n"
                         "### Select ssr sites to see coverage plot on the sidebar.\n"
                         "Once you have chosen SSR site,Then click \"Apply\" to start!")
    p_holder2.markdown("### Additional feature \n"
                        "You can select a list of reporting point to see which SSR is covered the point you've selected.")
    p_holder3.markdown("Once you have chosen all the option,Then click \"Apply\" to start!\n"
                        "Disclaimer : For now the data this project is the mock-up data.")

    st.sidebar.markdown("# Background")
#    st.sidebar.markdown("   This project was create to cross checking the performance of SSR status. \n")
    st.sidebar.markdown("The idea of the plot is using multivariate normal distribution to locate site \n")
    st.sidebar.markdown("Try it by checking the box and press the Apply button below :)")

    st.sidebar.markdown("# Choose the SSR here:")
    DMASSR = st.sidebar.checkbox("DMASSR")
    SVBSSR = st.sidebar.checkbox("SVBSSR")
    CMASSR = st.sidebar.checkbox("CMASSR")
    SRTSSR = st.sidebar.checkbox("SRTSSR")
    PSLSSR = st.sidebar.checkbox("PSLSSR")
    PUTSSR = st.sidebar.checkbox("PUTSSR")
    UBNSSR = st.sidebar.checkbox("UBNSSR")
    HTYSSR = st.sidebar.checkbox("HTYSSR")
    CMPSSR = st.sidebar.checkbox("CMPSSR")
    CTRSSR = st.sidebar.checkbox("CTRSSR")
    ROTSSR = st.sidebar.checkbox("ROTSSR")
    UDNSSR = st.sidebar.checkbox("UDNSSR")
    INTSSR = st.sidebar.checkbox("INTSSR")
    PHKSSR = st.sidebar.checkbox("PHKSSR")
    st.sidebar.markdown("---")
    st.sidebar.markdown("# Reporting Point VS SSR site !")
    st.sidebar.markdown(" Waana know your reporting point was coveraged by what radar ?\n"
                        " Select it below :)")
    rpoint = st.sidebar.selectbox("Select Reporting point here",options=list(rep_dms.keys()))
    apply_button = st.sidebar.button("Apply")

    if apply_button :
        p_holder1.empty()
        p_holder2.empty()
        p_holder3.empty()
        list_site = [DMASSR,SVBSSR,CMASSR,SRTSSR,PSLSSR,PUTSSR,UBNSSR,HTYSSR,CMPSSR,CTRSSR,ROTSSR,UDNSSR,INTSSR,PHKSSR] 
        visual(list_site,rpoint)

    st.sidebar.markdown("Thanks to Dopple finance community for giving me a big inspiration !!")
    st.sidebar.markdown("Keep calm and hodl dop")

main()