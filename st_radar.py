import numpy as np
import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
import json
from haversine import haversine,Unit

def dms2dd(Lat,Lon):
    Lat_dd = int(Lat[:2]) + float(int(Lat[2:4]))/60 + float(int(Lat[4:]))/3600
    Lon_dd = int(Lon[:3]) + float(int(Lon[3:5]))/60 + float(int(Lon[5:]))/3600
    return Lat_dd,Lon_dd
"""
@st.cache
def create_data():
    df = pd.read_csv("./data")
    df = df[df.columns[1:]]

    f = open ('./site_coor.json', "r")
    js = json.load(f)

    site_dms={}
    for i in js :
        site_dms[i] = dms2dd(js[i]['lat'],js[i]['lon'])

    return df,site_dms"""
@st.cache    
def create_data():
    df = pd.DataFrame()
    cov = [[1, 0], [0, 1]]
    j=0
    for i in site_dms :
        mean = site_dms[i]
        x,y =  np.random.multivariate_normal(mean, cov, 100000).T
        exec(str(i) +"_lat = j ")
        exec(str(i) +"_lon = j+1")
        df[str(i)+"_lat"] = pd.Series(x)
        df[str(i)+"_lon"] = pd.Series(y)
        j=j+2

    f = open ('./site_coor.json', "r")
    js = json.load(f)

    site_dms={}
    for i in js :
        site_dms[i] = dms2dd(js[i]['lat'],js[i]['lon'])
        
    return df,site_dms
df,site_dms = create_data()

def create_pipeline(list_site):
    pipeline = []
    for index, transformation in enumerate(list_site):
        if transformation:
            pipeline.append(index)
    return pipeline

def plot_map(list_site):
    st.title('Surveillance performance datavisualization')
    st.markdown("""
    * The glowing hexagon is refer to Secondary RADAR site
    * Your plots will appear below
    """)

    pipeline = create_pipeline(list_site)

    l = np.arange(14)
    lat_site,lon_site = [],[]
    farthest_p ={}
    v = list(site_dms.values())
    k = list(site_dms.keys())
    for i in l:
        mean = v[i]
        tmp_max = 0
        tmp_lat = df[df.columns[2*i]].to_list()
        tmp_lon = df[df.columns[2*i + 1]].to_list()
        
        for j in range(10000):
            point=(tmp_lat[j],tmp_lon[j])
            if haversine(mean,point, unit=Unit.NAUTICAL_MILES) > tmp_max :
                tmp_max = haversine(mean, point, unit=Unit.NAUTICAL_MILES)
        farthest_p[k[i]] = tmp_max
        
        lat_site = lat_site + tmp_lat
        lon_site = lon_site + tmp_lon
    
    px.set_mapbox_access_token("pk.eyJ1IjoidHVubWFuNTU1IiwiYSI6ImNrcHA5MGhzajBsMXIydm1ubzE1YmZhbHIifQ.8rPLrf8FYUq0ZkDrAF6_vA")
    fig = ff.create_hexbin_mapbox(
        #data_frame=df, lat="DMASSR_lat", lon="DMASSR_lon",
        lat =lat_site,lon =lon_site,mapbox_style ='light',
        nx_hexagon=50, opacity=0.2,min_count=1,labels={"color": "Surveillance site"}
    )

    fig.update_layout(margin=dict(b=0, t=0, l=0, r=0),hovermode=False,coloraxis_showscale=False)
    st.plotly_chart(fig)
    
    df_stat = pd.DataFrame.from_dict(farthest_p,orient = 'index',columns = ['Farthest Range (NM)'])
    df_stat = df_stat.T
    st.table(df_stat)

    with st.beta_expander("See notes"):
        st.markdown("To see more information of the concept of multivariate normal distribution on https://en.wikipedia.org/wiki/Multivariate_normal_distribution")
        st.markdown("And the information of the SSR on https://en.wikipedia.org/wiki/Secondary_surveillance_radar ")

def main():

    p_holder1 = st.empty()
    p_holder2 = st.empty()
    p_holder1.markdown("# Visualize an radar coverage pipeline\n"
                         "### Select the site of the pipeline in the sidebar.\n"
                         "Once you have chosen SSR site,Then click \"Apply\" to start!")
    p_holder2.markdown("After clicking start, the individual steps of the pipeline are visualized. The ouput of the previous step is the input to the next step.")
    
    st.sidebar.markdown("# Background")
    st.sidebar.markdown("   This project was create to cross checking the performance of SSR status \n" 
                        "The idea of the plot is using multivariate normal distribution to locate site \n"
                        "Try it by checking the box below :)")

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
    
    st.sidebar.markdown("Thanks to Dopple finance community for giving me a big inspiration !!\n"
                        "Dop to da moon")
    if st.sidebar.button("Apply"):
        p_holder1.empty()
        p_holder2.empty()
        list_site = [DMASSR,SVBSSR,CMASSR,SRTSSR,PSLSSR,PUTSSR,UBNSSR,HTYSSR,CMPSSR,CTRSSR,ROTSSR,UDNSSR,INTSSR,PHKSSR] 
        plot_map(list_site)
    
if __name__ == '__main__':
    #st.set_page_config(layout="wide",page_title="Radar coverage visualization")
    main()



