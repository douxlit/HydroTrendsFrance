"""
Copyright (c) [2024] [Quentin DASSIBAT]

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""


####################
# Import libraries #
####################

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import hydrofunc
import os
import geopandas as gpd
import datetime
import numpy as np
import shapely
import functools
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
pd.options.mode.chained_assignment = None

##################
# Set parameters # 
##################

#Area of Interest
AoI_filePath = "empriseSudLoire.gpkg" #path to file with extension
AoI_fileIsRaster = False #False if AoI is a vector file (.shp, .gpkg...); True if AoI is a raster file (.tif...)
AoI_EPSG = 4326 #ESPG code in which AoI_filePath is projected [int] e.g. 4326 #WARNING: !!has been developped with EPSG:4326 only, hence may not work with other EPSGs at this stage!!

#Period of Interest
timeRange = [2000,2020] #e.g. [2000,2020] for Water Balance Model (Wisser et al, 2010) as used in Rockstrom et al, 2023 

#Modules to run
runModule0 = False #True if Module 0 needs to be ran ; False if not
runModule1 = False 
runModule2 = False
runModule3 = False
runModule4 = False
runModule5 = False
runModule6 = True

#Miscellaneous
flushAllDirectories = False # Usefull to study a new AoI
operationStatus = False #True will keep only those stations that are still operating now a days; False will keep them all 
plottingMMF = False #True will generate the scatter plot with deviation bars of MMF for each station; False will not generate plot 
plottingMKtest = True #true will generate the scatter plot of the series and the trend of Mann-Kendall regression ; False will not generate plot
percentChange = 0.2 #Value from 0 to 1 that defines the threshold considered unsustainable for flow variability anomaly (executed in Module 6 only) 
flushDisk = False #True will delete folder "./tmp" and its files; False will keep it
accThreshold = 1e4 #Defines which pixel is considered part of an actively flowing river, with the value of 1000 as a presumptive standard

#####################
# Flush directories #
#####################

# Usefull to study a new AoI or time range

wd = os.getcwd() #returns the same wd as the wd where Python is launched in terminal
dataDirectory = f"{wd}/data"
analysisDirectory = f"{wd}/analysis"
tmpDirectory = f"{wd}/tmp"

if flushAllDirectories is True:

    if os.path.isdir(dataDirectory) is True:
        entries = os.listdir(dataDirectory)
        for entry in entries:
            os.remove(f"{dataDirectory}/{entry}")
        os.rmdir(dataDirectory)
    else:
        pass
    if os.path.isdir(analysisDirectory) is True:
        entries = os.listdir(analysisDirectory)
        for entry in entries:
            os.remove(f"{analysisDirectory}/{entry}")
        os.rmdir(analysisDirectory)
    else:
        pass
    if os.path.isdir(tmpDirectory) is True:
        entries = os.listdir(tmpDirectory)
        for entry in entries:
            os.remove(f"{tmpDirectory}/{entry}")
        os.rmdir(tmpDirectory)
    else:
        pass

else:
    pass


###################
# Set directories #
###################

wd = os.getcwd() #returns the same wd as the wd where Python is launched in terminal
dataDirectory = f"{wd}/data"
analysisDirectory = f"{wd}/analysis"
tmpDirectory = f"{wd}/tmp"
if os.path.isdir(dataDirectory) is False:
    os.mkdir(f"{wd}/data")
else:
    pass
if os.path.isdir(analysisDirectory) is False:
    os.mkdir(f"{wd}/analysis")
else:
    pass
if os.path.isdir(tmpDirectory) is False:
    os.mkdir(f"{wd}/tmp")
else:
    pass


####################################
# Module 0: download required data #
####################################

if runModule0 is True:

    globstart = datetime.datetime.now()

    print('Run MODULE 0')

    print("Get the spatial extent of the AoI")
    
    if AoI_fileIsRaster is True:
        AoI_bbox = hydrofunc.extract_Rasterbbox(AoI_filePath)
    else:
        AoI_bbox = hydrofunc.extract_Vectorbbox(AoI_filePath)
    
    print("Retrieve stations from hubeau.eaufrance.fr")
    
    dst = f"{dataDirectory}/stations_locations.gpkg"
    hydrofunc.request_locations_hubeau(AoI_bbox,dst,operating=operationStatus,tRange=timeRange)
    del dst
    
    print("Retrieve Mean Daily Flows from hubeau.eaufrance.fr")
    
    src = f"{dataDirectory}/stations_locations_{str(timeRange[0])}{str(timeRange[1])}.gpkg"
    dst = f"{dataDirectory}/stations_observations_mdf_{str(timeRange[0])}{str(timeRange[1])}.gpkg"
    period = [f"{timeRange[0]}-01-01",f"{timeRange[1]}-01-01"] #["yyyy-mm-dd","yyyy-mm-dd"]
    hydrofunc.requestFrontend_observations_hubeau(src,dst,tRange=period)
    del src, dst

    print("Download SRTM-90 DEM from opentopography API")
    
    dst = f"{dataDirectory}/DEM.tif"
    hydrofunc.request_DEM(AoI_bbox,AoI_EPSG,dst)

    print("Download OSM spatial features")

    #See: https://wiki.openstreetmap.org/wiki/Map_features
    #Wastewater plants
    dst = f"{dataDirectory}/wastewater_plants.gpkg"
    feat = {'man_made':'wastewater_plant'}
    hydrofunc.request_osm_feature(AoI_bbox,AoI_EPSG,feat,dst)
    del dst, feat
    #Dams
    dst = f"{dataDirectory}/dams.gpkg"
    feat = {'waterway':'dam'}
    hydrofunc.request_osm_feature(AoI_bbox,AoI_EPSG,feat,dst)
    del dst, feat
    #Weirs
    dst = f"{dataDirectory}/weirs.gpkg"
    feat = {'waterway':'weir'}
    hydrofunc.request_osm_feature(AoI_bbox,AoI_EPSG,feat,dst)
    del dst, feat
    #lock
    dst = f"{dataDirectory}/locks.gpkg"
    feat = {'water':'lock'}
    hydrofunc.request_osm_feature(AoI_bbox,AoI_EPSG,feat,dst)
    del dst, feat
    #reservoirs
    dst = f"{dataDirectory}/reservoirs.gpkg"
    feat = {'water':'reservoir'}
    hydrofunc.request_osm_feature(AoI_bbox,AoI_EPSG,feat,dst)
    del dst, feat

    print("Force CLC land occupation raster to the same projection as the current project") 
    
    src = "U2018_CLC2018_V2020_20u1.tif"
    dst = f"{dataDirectory}/U2018_CLC2018_V2020_20u1_EPSG{str(AoI_EPSG)}.tif"
    hydrofunc.reproject_raster(src,dst,AoI_EPSG)
    del src, dst


    print("Total Elapsed Time: ", datetime.datetime.now()-globstart)
    
    with open(f"{wd}/log.txt", 'a') as file:
        file.write(f"MODULE0.py Elapsed Time: {str(datetime.datetime.now()-globstart)}\n")

else:
    pass



#########################################
# Module 1: create topographic material #
#########################################

if runModule1 is True:

    globstart = datetime.datetime.now()

    print('Run MODULE 1')

    entries = os.listdir(tmpDirectory)
    for entry in entries:
        os.remove(f"{tmpDirectory}/{entry}")

    print("Generate Local Drain Direction") 

    src = f"{dataDirectory}/DEM.tif"
    dst = f"{tmpDirectory}/DEM.map"
    msk = f"{dataDirectory}/DEM.tif"
    hydrofunc.convert_to_pcraster(src,dst,AoI_EPSG,msk)
    del src, dst, msk

    src = f"{tmpDirectory}/DEM.map"
    dst = f"{tmpDirectory}/LDD.map"
    cln = f"{tmpDirectory}/DEM.map"
    hydrofunc.create_flowdirection(src,dst,cln,1e31,1e31,1e31,1e31)
    del src, dst 

    print("Generate Flow Accumulation with a user-specified threshold")
    #Create material
    dst = f"{tmpDirectory}/material.map"
    msk = f"{dataDirectory}/DEM.tif"
    hydrofunc.create_material(msk,dst,AoI_EPSG,1)
    del dst, msk
    #Create accuflux
    ldd = f"{tmpDirectory}/LDD.map"
    mat = f"{tmpDirectory}/material.map"
    dst = f"{tmpDirectory}/accuflux.map"
    hydrofunc.create_accuflux(ldd,mat,dst,cln)
    del ldd, mat, dst
    #Convert to .tif file
    src = f"{tmpDirectory}/accuflux.map"
    dst = f"{tmpDirectory}/accuflux.tif"
    hydrofunc.convert_to_geotiff(src,dst,AoI_EPSG)
    del src, dst
    #Filter accuflux with a cutoff
    src = f"{tmpDirectory}/accuflux.tif"
    dst = f"{tmpDirectory}/accuflux_geq{str(accThreshold)}.tif"
    calc = f"numpy.where(A>{np.float32(accThreshold)},A,numpy.nan)" #not zero for False case to further apply log10 function for vizualisation purpose
    cmd = 'gdal_calc.py --overwrite --calc "{calc}" --format GTiff --type Float32 --extent=intersect --NoDataValue={nd} -A {src} --A_band 1 --outfile {dst}'.format(nd="'none'",src=src,dst=dst,calc=calc) #new scalar raster intersecting the AoI
    os.system(cmd)
    del src, dst

    print("Generate subcatchments of all stations retrieved")

    #Join analyzed stations to acculfux
    gdf = gpd.read_file(f"{dataDirectory}/stations_locations_NoTimeRangeConstraint.gpkg")
    ids = np.arange(1,len(gdf)+1,1)
    gdf.loc[:,'id_station'] = list(ids)
    gdf.to_file(f"{dataDirectory}/stations_locations_NoTimeRangeConstraint.gpkg")
    points = f"{dataDirectory}/stations_locations_NoTimeRangeConstraint.gpkg"
    layer = 'id_station'
    acc = f"{tmpDirectory}/accuflux_geq{str(accThreshold)}.tif"
    dst = f"{tmpDirectory}/stations2accuflux_all.gpkg"
    hydrofunc.join_points_to_pixels(points,layer,acc,AoI_EPSG,dst)
    del points, gdf, layer, acc, dst
    #Create subcatchments
    cln = f"{tmpDirectory}/DEM.map"
    points = f"{tmpDirectory}/stations2accuflux_all.gpkg"
    ldd = f"{tmpDirectory}/LDD.map"
    dst = f"{tmpDirectory}/subcatchments_all.map"
    hydrofunc.create_subcatchments(points,ldd,dst,cln)
    del points, ldd, dst
    #Convert to vector file
    src = f"{tmpDirectory}/subcatchments_all.map"
    dst = f"{dataDirectory}/subcatchments_all.gpkg"
    hydrofunc.raster_to_polygons(src,dst,AoI_EPSG,'catchment_id',zRestriction=None)
    del src, dst
    #Add station codes as catchments ids
    stations = gpd.read_file(f"{dataDirectory}/stations_locations_NoTimeRangeConstraint.gpkg")
    catchments = gpd.read_file(f"{dataDirectory}/subcatchments_all.gpkg")
    stations = stations[['id_station','code_station']]
    m = catchments.merge(stations, left_on='catch_id', right_on='id_station', how='left')
    m.drop(['catch_id','id_station'],axis=1,inplace=True)
    m.to_file(f"{dataDirectory}/subcatchments_all.gpkg")
    del m

    print("Flush temporary files from disk if required")
    
    if flushDisk is True:
        entries = os.listdir(tmpDirectory)
        for entry in entries:
            os.remove(f"{tmpDirectory}/{entry}")
        os.rmdir(tmpDirectory)
    else:
        pass


    print("Total Elapsed Time: ", datetime.datetime.now()-globstart)
    
    with open(f"{wd}/log.txt", 'a') as file:
        file.write(f"MODULE1.py Elapsed Time: {str(datetime.datetime.now()-globstart)}\n")

else:
    pass




#############################################
# Module 2: compute hydrographic statistics #
#############################################

if runModule2 is True:

    globstart = datetime.datetime.now()

    print('Run MODULE 2')
    
    print("Compute MMF average and deviation for all stations")
    
    #Read observations file
    obs = gpd.read_file(f"{dataDirectory}/stations_observations_mdf_{str(timeRange[0])}{str(timeRange[1])}.gpkg")
    stationsList = list(set(obs['code_station']))

    #Compute average MMF and deviation
    l = []
    
    for station in stationsList:
    
        print(f"Station {len(l)+1}/{len(stationsList)}")
        
        dst = f"{tmpDirectory}/mmf_average_station_{str(station)}.gpkg"
        d = hydrofunc.compute_MeanMonthlyFlow_average(str(station),obs,dst,timeRange,generate_plot=plottingMMF)
        l.append(d)
        del dst

    #Concatenate and save to disk
    #Make a .gpkg file
    mmf = pd.concat(l)
    final_mmf = gpd.GeoDataFrame(mmf, crs="EPSG:4326")
    final_mmf.set_geometry('geometry')
    final_mmf.to_file(f"{analysisDirectory}/stations_observations_mmf_average_{str(timeRange[0])}{str(timeRange[1])}_points.gpkg")
    del final_mmf
    #Make a .csv file
    gdf = gpd.read_file(f"{analysisDirectory}/stations_observations_mmf_average_{str(timeRange[0])}{str(timeRange[1])}_points.gpkg")
    gdf.drop('geometry', axis=1, inplace=True)
    gdf.to_csv(f"{analysisDirectory}/stations_observations_mmf_average_{str(timeRange[0])}{str(timeRange[1])}.csv")
    del gdf

    print("Compute MMF for all month for each station and MK regression")
    
    #Read observations file
    obs = gpd.read_file(f"{dataDirectory}/stations_observations_mdf_{str(timeRange[0])}{str(timeRange[1])}.gpkg")
    stationsList = list(set(obs['code_station']))

    #Compute MMF for each month and each station and the compute Mann-Kendall trend for average and stdev

    list_mmf = [] #to save the df of mean monthly flows and further merge them
    list_mk = [] #to save the df of MK regression and further merge them
    
    for station in stationsList:
    
        print(f"Station {station}")

        frames = {'code_station': [],
           'z_value_mean': [],
          'slope_mean' : [],
          'intercept_mean': [],
          'globalTrend_mean': [],
          'p_value_mean' : [],
          'z_value_std': [],
          'slope_std' : [],
          'intercept_std': [],
          'globalTrend_std': [],
          'p_value_std' : []}

        frames['code_station'].append(station)

        print("Compute MMF for all months")
        
        dst = f"{dataDirectory}/mmf_all_station_{str(station)}_{str(timeRange[0])}{str(timeRange[1])}_points.gpkg"
        mmf = hydrofunc.compute_MeanMonthlyFlow_all(str(station),obs,dst,timeRange,AoI_EPSG)
        #Export as .csv
        if os.path.exists(dst) is True:
            mmf.drop('geometry',axis=1,inplace=True)
            mmf.to_csv(f"{dataDirectory}/mmf_all_station_{str(station)}_{str(timeRange[0])}{str(timeRange[1])}.csv",index=False)
            list_mmf.append(mmf)
        else:
            pass
        del dst, mmf
        
        print("Compute Mann-Kendall trend")
        
        src = f"{dataDirectory}/mmf_all_station_{str(station)}_{str(timeRange[0])}{str(timeRange[1])}.csv"
        if os.path.exists(src) is True:
            #Compute MK test for average series
            stat = "mean"
            datesLayer = "date"
            valuesLayer = "MMF_mean"
            res = hydrofunc.MannKendallStat(station,src,datesLayer,valuesLayer,stat,generate_plot=plottingMKtest) 
            z, s, i, t, c = res
            frames['z_value_mean'].append(z)
            frames['slope_mean'].append(s)
            frames['intercept_mean'].append(i)
            frames['globalTrend_mean'].append(t)
            frames['p_value_mean'].append(c)
            #Compute MK test for stdev series
            stat = "std"
            datesLayer = "date"
            valuesLayer = "MMF_std"
            res = hydrofunc.MannKendallStat(station,src,datesLayer,valuesLayer,stat,generate_plot=plottingMKtest)
            z, s, i, t, c = res
            frames['z_value_std'].append(z)
            frames['slope_std'].append(s)
            frames['intercept_std'].append(i)
            frames['globalTrend_std'].append(t)
            frames['p_value_std'].append(c)
            #Store data 
            list_mk.append(pd.DataFrame(frames))
        else:
            print(f"File does not exist: {src}")
        
    #Generate and export MK dataframe     
    df = pd.concat(list_mk)     
    #As a .csv     
    dst = f"{analysisDirectory}/MannKendallRegression_{str(timeRange[0])}{str(timeRange[1])}.csv"
    df.to_csv(dst,index=False)
    del dst, df
    #Generate and export MMF dataframe
    df = pd.concat(list_mmf)
    dst = f"{dataDirectory}/mmf_all_{str(timeRange[0])}{str(timeRange[1])}.csv"
    df.to_csv(dst,index=False)
    del dst, df
    
    #Export MK df as a point-geometry .gpkg
    dst = f"{analysisDirectory}/MannKendallRegression_{str(timeRange[0])}{str(timeRange[1])}_points.gpkg"
    points = gpd.read_file(f"{dataDirectory}/stations_locations_{str(timeRange[0])}{str(timeRange[1])}.gpkg")
    points_cut = points[['code_station','geometry']]
    df = pd.read_csv(f"{analysisDirectory}/MannKendallRegression_{str(timeRange[0])}{str(timeRange[1])}.csv")
    join = df.merge(points_cut,left_on='code_station',right_on='code_station',how = 'left')
    gdf = gpd.GeoDataFrame(join,geometry='geometry',crs=f"EPSG:{str(AoI_EPSG)}")
    gdf.to_file(dst)
    del df, join, gdf, dst

    #Export MMF df as a point-geometry .gpkg
    dst = f"{dataDirectory}/mmf_all_{str(timeRange[0])}{str(timeRange[1])}_points.gpkg"
    points = gpd.read_file(f"{dataDirectory}/stations_locations_{str(timeRange[0])}{str(timeRange[1])}.gpkg")
    points_cut = points[['code_station','geometry']]
    df = pd.read_csv(f"{dataDirectory}/mmf_all_{str(timeRange[0])}{str(timeRange[1])}.csv")
    join = df.merge(points_cut,left_on='code_station',right_on='code_station',how = 'left')
    gdf = gpd.GeoDataFrame(join,geometry='geometry',crs=f"EPSG:{str(AoI_EPSG)}")
    gdf.to_file(dst)
    del df, join, gdf, dst
    
    print("Total Elapsed Time: ", datetime.datetime.now()-globstart)
    
    with open(f"{wd}/log.txt", 'a') as file:
        file.write(f"MODULE2.py Elapsed Time: {str(datetime.datetime.now()-globstart)}\n")


else:
    pass


####################################
# Module 3: Generate subcatchments #
####################################

if runModule3 is True:

    globstart = datetime.datetime.now()

    print('Run MODULE 3')

    print("Generate subcatchments based on the number of analyzed stations")

    #Join analyzed stations to acculfux
    gdf = gpd.read_file(f"{analysisDirectory}/stations_observations_mmf_average_{str(timeRange[0])}{str(timeRange[1])}_points.gpkg")
    ids = np.arange(1,len(gdf)+1,1)
    gdf.loc[:,'id_station'] = list(ids)
    gdf.to_file(f"{analysisDirectory}/stations_observations_mmf_average_{str(timeRange[0])}{str(timeRange[1])}_points.gpkg")
    points = f"{analysisDirectory}/stations_observations_mmf_average_{str(timeRange[0])}{str(timeRange[1])}_points.gpkg"
    layer = 'id_station'
    acc = f"{tmpDirectory}/accuflux_geq{str(accThreshold)}.tif"
    dst = f"{tmpDirectory}/stations2accuflux_analyzed.gpkg"
    hydrofunc.join_points_to_pixels(points,layer,acc,AoI_EPSG,dst)
    del points, gdf, layer, acc, dst
    #Create subcatchments
    cln = f"{tmpDirectory}/DEM.map"
    points = f"{tmpDirectory}/stations2accuflux_analyzed.gpkg"
    ldd = f"{tmpDirectory}/LDD.map"
    dst = f"{tmpDirectory}/subcatchments_analyzed.map"
    hydrofunc.create_subcatchments(points,ldd,dst,cln)
    del points, ldd, dst
    #Convert to vector file
    src = f"{tmpDirectory}/subcatchments_analyzed.map"
    dst = f"{dataDirectory}/subcatchments_analyzed.gpkg"
    hydrofunc.raster_to_polygons(src,dst,AoI_EPSG,'catchment_id',zRestriction=None)
    del src, dst
    #Add station codes as catchments ids
    stations = gpd.read_file(f"{analysisDirectory}/stations_observations_mmf_average_{str(timeRange[0])}{str(timeRange[1])}_points.gpkg")
    catchments = gpd.read_file(f"{dataDirectory}/subcatchments_analyzed.gpkg")
    stations = stations[['id_station','code_station']]
    m = catchments.merge(stations, left_on='catch_id', right_on='id_station', how='left')
    m.drop(['catch_id','id_station'],axis=1,inplace=True)
    m.to_file(f"{dataDirectory}/subcatchments_analyzed.gpkg")
    del m
    
    print("Flush temporary files from disk if required")
    
    if flushDisk is True:
        entries = os.listdir(tmpDirectory)
        for entry in entries:
            os.remove(f"{tmpDirectory}/{entry}")
        os.rmdir(tmpDirectory)
    else:
        pass

    
    with open(f"{wd}/log.txt", 'a') as file:
        file.write(f"MODULE3.py Elapsed Time: {str(datetime.datetime.now()-globstart)}\n")


else:
    pass


#######################
# Module 4: Rendering #
#######################

if runModule4 is True:

    globstart = datetime.datetime.now()

    print("Module 4")
    
    print("Rendering MMF_average catchment-wise")

    #Join with subcatchment vector file
    mmf_average = gpd.read_file(f"{analysisDirectory}/stations_observations_mmf_average_{str(timeRange[0])}{str(timeRange[1])}_points.gpkg")
    mmf_average.drop('geometry',axis=1,inplace=True)
    catch = gpd.read_file(f"{dataDirectory}/subcatchments_analyzed.gpkg")
    m = catch.merge(mmf_average,left_on='code_station',right_on='code_station',how='left')
    m.to_file(f"{analysisDirectory}/stations_observations_mmf_average_{str(timeRange[0])}{str(timeRange[1])}_subcatchments.gpkg")
    del mmf_average, m, catch

    print("Rendering MMF_all catchment-wise")

    #Join with subcatchment vector file
    mmf_all = gpd.read_file(f"{dataDirectory}/mmf_all_{str(timeRange[0])}{str(timeRange[1])}_points.gpkg")
    mmf_all.drop('geometry',axis=1,inplace=True)
    catch = gpd.read_file(f"{dataDirectory}/subcatchments_analyzed.gpkg")
    m = catch.merge(mmf_all,left_on='code_station',right_on='code_station',how='left')
    m.to_file(f"{dataDirectory}/mmf_all_{str(timeRange[0])}{str(timeRange[1])}_subcatchments.gpkg")
    del mmf_all, m, catch

    print("Rendering MK analysis catchment-wise")

    #Read dataframe of MK regression
    src = f"{analysisDirectory}/MannKendallRegression_{str(timeRange[0])}{str(timeRange[1])}.csv"
    df = pd.read_csv(src)

    #Export as a subcatchment-geometry .gpkg
    dst = f"{analysisDirectory}/MannKendallRegression_{str(timeRange[0])}{str(timeRange[1])}_subcatchments.gpkg"
    catch = gpd.read_file(f"{dataDirectory}/subcatchments_analyzed.gpkg")
    catch_cut = catch[['code_station','geometry']]
    #only keep catchments that have valid geometry
    #catch_cut2 = catch_cut[~catch_cut.geometry.isnull()]
    #catch_cut2['geometry'] = catch_cut2['geometry'].astype("string")
    join = df.merge(catch_cut,left_on='code_station',right_on='code_station',how='left')
    #join['geometry'] = join['geometry'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(join,geometry='geometry',crs=f"EPSG:{str(AoI_EPSG)}")
    gdf.to_file(dst)
    del df, join, gdf, dst

    print("Generate fancy maps")

    #For the trend on the flow mean
    srcPolygData = f"{analysisDirectory}/MannKendallRegression_{str(timeRange[0])}{str(timeRange[1])}_subcatchments.gpkg"
    ##Prepare pointsLayer to display: retrieve position of stations in accuflux map
    points2accuflux = gpd.read_file(f"{tmpDirectory}/stations2accuflux_analyzed.gpkg")
    points = gpd.read_file(f"{analysisDirectory}/MannKendallRegression_{str(timeRange[0])}{str(timeRange[1])}_points.gpkg")
    points2accuflux_cut = points2accuflux[['code_station','geometry']]
    points_cut = points.drop('geometry',axis=1)
    m = points2accuflux_cut.merge(points_cut,left_on='code_station',right_on='code_station',how='left')
    m.to_file(f"{analysisDirectory}/stations2accuflux_MannKendallRegression_{str(timeRange[0])}{str(timeRange[1])}_points.gpkg")
    ##Plot
    srcPoints = f"{analysisDirectory}/stations2accuflux_MannKendallRegression_{str(timeRange[0])}{str(timeRange[1])}_points.gpkg"
    srcPolygNodata = f"{dataDirectory}/subcatchments_all.gpkg"
    dst = f"{analysisDirectory}/MannKendallRegression_{str(timeRange[0])}{str(timeRange[1])}_GlobalTrendOfMean_map.png"
    polyg = 'globalTrend_mean'
    label = 'p_value_mean'
    title = f"Global trend of Mean Monthly Flows (colors) and their p-values (labels) over the period {str(timeRange[0])}-{str(timeRange[1])}"
    hydrofunc.make_map_LabelsOnPoints(srcPolygData,polyg,label,srcPoints,srcPolygNodata,title,dst)
    del srcPolygData, polyg, label, title, dst, srcPoints, srcPolygNodata

    #For the trend on the flow deviation
    srcPolygData = f"{analysisDirectory}/MannKendallRegression_{str(timeRange[0])}{str(timeRange[1])}_subcatchments.gpkg"
    ##Prepare pointsLayer to display: retrieve position of stations in accuflux map
    points2accuflux = gpd.read_file(f"{tmpDirectory}/stations2accuflux_analyzed.gpkg")
    points = gpd.read_file(f"{analysisDirectory}/MannKendallRegression_{str(timeRange[0])}{str(timeRange[1])}_points.gpkg")
    points2accuflux_cut = points2accuflux[['code_station','geometry']]
    points_cut = points.drop('geometry',axis=1)
    m = points2accuflux_cut.merge(points_cut,left_on='code_station',right_on='code_station',how='left')
    m.to_file(f"{analysisDirectory}/stations2accuflux_MannKendallRegression_{str(timeRange[0])}{str(timeRange[1])}_points.gpkg")
    ##Plot
    srcPoints = f"{analysisDirectory}/stations2accuflux_MannKendallRegression_{str(timeRange[0])}{str(timeRange[1])}_points.gpkg"
    srcPolygNodata = f"{dataDirectory}/subcatchments_all.gpkg"
    dst = f"{analysisDirectory}/MannKendallRegression_{str(timeRange[0])}{str(timeRange[1])}_GlobalTrendOfDeviation_map.png"
    polyg = 'globalTrend_std'
    label = 'p_value_std'
    title = f"Global trend of Monthly Flow Deviations (colors) and their p-values (labels) over the period {str(timeRange[0])}-{str(timeRange[1])}"
    hydrofunc.make_map_LabelsOnPoints(srcPolygData,polyg,label,srcPoints,srcPolygNodata,title,dst)
    del srcPolygData, polyg, label, title, dst, srcPoints, srcPolygNodata
    
    #For the trend on the flow mean
    #srcPolygData = f"{analysisDirectory}/MannKendallRegression_{str(timeRange[0])}{str(timeRange[1])}_subcatchments.gpkg"
    #srcPoints = f"{dataDirectory}/stations_locations_NoTimeRangeConstraint.gpkg"
    #srcPolygNodata = f"{dataDirectory}/subcatchments_all.gpkg"
    #dst = f"{analysisDirectory}/MannKendallRegression_{str(timeRange[0])}{str(timeRange[1])}_GlobalTrendOfMean_map.png"
    #polyg = 'globalTrend_mean'
    #label = 'p_value_mean'
    #title = f"Gobal trend of Mean Monthly Flows (colors) and their p-values (labels) over the period {str(timeRange[0])}-{str(timeRange[1])}"
    #hydrofunc.make_map_LabelsOnPolygons(srcPolygData,polyg,label,srcPoints,srcPolygNodata,title,dst,plotPoints=False)
    #del srcPolygData, polyg, label, title, dst, srcPoints, srcPolygNodata
    
    #For the trend on the flow deviation
    #srcPolygData = f"{analysisDirectory}/MannKendallRegression_{str(timeRange[0])}{str(timeRange[1])}_subcatchments.gpkg"
    #srcPoints = f"{dataDirectory}/stations_locations_NoTimeRangeConstraint.gpkg"
    #srcPolygNodata = f"{dataDirectory}/subcatchments_all.gpkg"
    #dst = f"{analysisDirectory}/MannKendallRegression_{str(timeRange[0])}{str(timeRange[1])}_GlobalTrendOfDeviation_map.png"
    #polyg = 'globalTrend_std'
    #label = 'p_value_std'
    #title = f"Global trend of Monthly Flow Deviations (colors) and their p-values (labels) over the period {str(timeRange[0])}-{str(timeRange[1])}"
    #hydrofunc.make_map_LabelsOnPolygons(srcPolygData,polyg,label,srcPoints,srcPolygNodata,title,dst,plotPoints=False)
    #del srcPolygData, polyg, label, title, dst, srcPoints, srcPolygNodata
    
    #For the average annual flow deviation
    srcPolygData = f"{analysisDirectory}/stations_observations_mmf_average_{str(timeRange[0])}{str(timeRange[1])}_subcatchments.gpkg"
    ##Prepare pointsLayer to display: retrieve position of stations in accuflux map
    points2accuflux = gpd.read_file(f"{tmpDirectory}/stations2accuflux_analyzed.gpkg")
    points = gpd.read_file(f"{analysisDirectory}/stations_observations_mmf_average_{str(timeRange[0])}{str(timeRange[1])}_points.gpkg")
    points2accuflux_cut = points2accuflux[['code_station','geometry']]
    points_cut = points.drop('geometry',axis=1)
    m = points2accuflux_cut.merge(points_cut,left_on='code_station',right_on='code_station',how='left')
    m.to_file(f"{analysisDirectory}/stations2accuflux_observations_mmf_average_{str(timeRange[0])}{str(timeRange[1])}_points.gpkg")
    ##Plot
    srcPoints = f"{analysisDirectory}/stations2accuflux_observations_mmf_average_{str(timeRange[0])}{str(timeRange[1])}_points.gpkg"
    srcPolygNodata = f"{dataDirectory}/subcatchments_all.gpkg"
    dst = f"{analysisDirectory}/MeanAnnualCoefficientofVariation_{str(timeRange[0])}{str(timeRange[1])}_map.png"
    polyg = f"MeanAnnualCoV_{timeRange[0]}{timeRange[1]}"
    label = f"MeanAnnualCoV_{timeRange[0]}{timeRange[1]}"
    title = f"Annual Coefficient of Variation over the period {str(timeRange[0])}-{str(timeRange[1])}"
    hydrofunc.make_map_LabelsOnPoints(srcPolygData,polyg,label,srcPoints,srcPolygNodata,title,dst)
    del srcPolygData, polyg, label, title, dst, srcPoints, srcPolygNodata
    
    print("Flush temporary files from disk if required")
    
    if flushDisk is True:
        entries = os.listdir(tmpDirectory)
        for entry in entries:
            os.remove(f"{tmpDirectory}/{entry}")
        os.rmdir(tmpDirectory)
    else:
        pass

    
    with open(f"{wd}/log.txt", 'a') as file:
        file.write(f"MODULE4.py Elapsed Time: {str(datetime.datetime.now()-globstart)}\n")


else:
    pass



##############################
# Module 5: Spatial Analysis #
##############################

if runModule5 is True:

    globstart = datetime.datetime.now()

    print("MODULE 5")
    
    print("Build dataset for correlation matrix")
    
    #Drop potential duplicates
    src = f"{analysisDirectory}/stations_observations_mmf_average_{str(timeRange[0])}{str(timeRange[1])}_subcatchments.gpkg"
    srcData = gpd.read_file(src)
    srcData.drop_duplicates(subset='code_station',inplace=True)
    #Build a single geometry catchment dataframe
    tmp = srcData[['geometry']]
    union_polygon = shapely.ops.unary_union(tmp.geometry)
    unitCatchment = gpd.GeoDataFrame({'geometry': [union_polygon]})
    unitCatchment.to_file(f"{tmpDirectory}/unitCatchment.gpkg")
    del src, tmp, union_polygon
    #Clip land occupation raster to the shape of all catchments within the AoI
    src = f"{dataDirectory}/U2018_CLC2018_V2020_20u1_EPSG{str(AoI_EPSG)}.tif"
    dst = f"{dataDirectory}/CLC_AoI.tif"
    m = f"{tmpDirectory}/unitCatchment.gpkg"
    hydrofunc.clip_to_shapefile(src,dst,AoI_EPSG,m)
    del src, dst
    #Count all tied pixel values 
    srcAll = f"{dataDirectory}/CLC_AoI.tif"
    f = hydrofunc.extract_cellsValues(srcAll)
    df = pd.DataFrame(f)
    occupationClassesAll = list(set(df['values']))
    del f, df
    
    listStations = list(set(srcData['code_station']))
    listFeatures = ['wastewater_plants',
                    'dams',
                    'weirs',
                    'locks',
                    'reservoirs']
    listDataframes = []
    
    
    for station in listStations:

        print("Station",station)

        mask = srcData.loc[srcData['code_station']==station]

        print("Count features intersecting each catchment")

        for feat in listFeatures:
            
            #Import features and check wheter there is any such feature throughout the whole AoI

            src = f"{dataDirectory}/{str(feat)}.gpkg"
            
            if os.path.exists(src):
            
                features = gpd.read_file(src) #Has been saved in EPSG = to AoI_EPSG
    
                if len(features) != 0:
                    
                    feature = list(features['feature'])[0]
                
                    #Clip features
                    features_clip = gpd.clip(features, mask)
                    
                    #Check whether there is any such feature thoughout the subcatchment
                    
                    if len(features_clip) != 0:
                        
                        #Get the km2 area of the subcatchment
                        mask.to_crs("EPSG:3857",inplace=True) #Cartesian metric CRS
                        mask.loc[:,"area_km2"] = mask['geometry'].area/(10**6)
                        mask.to_crs(f"EPSG:{str(AoI_EPSG)}",inplace=True)
                        #Count the number of entities as a fraction of area_km2
                        features_clip.drop_duplicates(subset='feature',inplace=True)
                        km2 = list(mask['area_km2'])[0]
                        entities_per_km2 = len(features_clip)/float(km2)
                        mask.loc[:,'entities_per_km2'] = entities_per_km2
                        #Write into dataframe the feature collected
                        mask.loc[:,'feature'] = feature
                        #Keep in mask only 'feature', 'entities_per_km2', 'geometry', 'MeanAnnualCoV_timeRange', 'code_station'
                        tmp = mask[['code_station',
                                    f"MeanAnnualCoV_{str(timeRange[0])}{str(timeRange[1])}",
                                    'feature',
                                    'entities_per_km2',
                                    'geometry']]
        
                    else:
                        
                        mask.loc[:,'feature'] = feature
                        mask.loc[:,'entities_per_km2'] = float(0)
                        tmp = mask[['code_station',
                                    f"MeanAnnualCoV_{str(timeRange[0])}{str(timeRange[1])}",
                                    'feature',
                                    'entities_per_km2',
                                    'geometry']]
                    
                    #Save to list to further concatenate
                    listDataframes.append(tmp)
                
                else:
                    pass
            else:
                pass


        print("Count land occupation intersecting each catchment")

        #Clip land occupation raster to the current catchment
        maskFile = f"{tmpDirectory}/mmf_average_station_{str(station)}_catchment.gpkg"
        mask.to_file(maskFile)
        src = f"{dataDirectory}/U2018_CLC2018_V2020_20u1_EPSG{str(AoI_EPSG)}.tif"
        dst = f"{tmpDirectory}/CLC_station_{str(station)}.tif"
        hydrofunc.clip_to_shapefile(src,dst,AoI_EPSG,maskFile)
        del maskFile, src, dst

        srcCatch = f"{tmpDirectory}/CLC_station_{str(station)}.tif"
        
        if os.path.exists(srcCatch) is True: #This 2 case condition is made to prevent computations for catchments with a total area smaller than the land occupation raster's resolution
            
            f = hydrofunc.extract_cellsValues(srcCatch)
            d = pd.DataFrame(f)
            if float(-128) in list(set(d['values'])): #Delete -128 pixel value from list as it corresponds to no data value
                df = d.loc[d['values'] != float(-128)]
            else:
                df = d.copy()
            tot_pixels = len(df)
            occupationClassesCatch = list(set(df['values']))
            del f 
            #f = hydrofunc.count_tiedCells(srcCatch,occupationClassesCatch)
            #df = pd.DataFrame(f)
            #Export as a dataframe
            geom = list(mask["geometry"])[0]
            MeanAnnualCoV = list(mask[f"MeanAnnualCoV_{str(timeRange[0])}{str(timeRange[1])}"])[0]
            frames = {'code_station':[],'feature':[],'entities_per_km2':[],f"MeanAnnualCoV_{str(timeRange[0])}{str(timeRange[1])}":[],'geometry':[]}
            for occ in occupationClassesCatch:
                frames['feature'].append(f"occupation_class{str(occ)}")
                dfc = df.loc[df['values'] == occ]
                count = len(dfc)/tot_pixels
                frames['entities_per_km2'].append(count)
                frames[f"MeanAnnualCoV_{str(timeRange[0])}{str(timeRange[1])}"].append(MeanAnnualCoV)
                frames['geometry'].append(geom)
                frames['code_station'].append(station)
            occupationClassesDiff = [item for item in occupationClassesAll if item not in occupationClassesCatch]
            for occ in occupationClassesDiff:
                frames['feature'].append(f"occupation_class{str(occ)}")
                frames['entities_per_km2'].append(float(0))
                frames[f"MeanAnnualCoV_{str(timeRange[0])}{str(timeRange[1])}"].append(MeanAnnualCoV)
                frames['geometry'].append(geom)
                frames['code_station'].append(station)
            tmp = pd.DataFrame(frames)
        
        else:
            
            frames = {'code_station':[],'feature':[],'entities_per_km2':[],f"MeanAnnualCoV_{str(timeRange[0])}{str(timeRange[1])}":[],'geometry':[]}
            MeanAnnualCoV = list(mask[f"MeanAnnualCoV_{str(timeRange[0])}{str(timeRange[1])}"])[0]
            geom = list(mask["geometry"])[0]
            for occ in occupationClassesAll:
                frames['feature'].append(f"occupation_class{str(occ)}")
                frames['entities_per_km2'].append(float(0))
                frames[f"MeanAnnualCoV_{str(timeRange[0])}{str(timeRange[1])}"].append(MeanAnnualCoV)
                frames['geometry'].append(geom)
                frames['code_station'].append(station)
            tmp = pd.DataFrame(frames)
            
        listDataframes.append(tmp)

    #Concatenate all dataframes and save to disk    
    df = pd.concat(listDataframes)
    gdf = gpd.GeoDataFrame(df)
    gdf.set_geometry('geometry',inplace=True)
    gdf.set_crs(crs=f"EPSG:{str(AoI_EPSG)}",inplace=True)
    dst = f"{dataDirectory}/anthropogenic_features_count_by_subcatchment_raw.gpkg"
    gdf.to_file(dst) #Polygon dataframe with columns 'feature', 'entities_per_km2', 'geometry', 'MeanAnnualCoV_timeRange', 'code_station'
    del gdf


    print("Restructure dataset for correlation matrix")

    #Prepare dataframe (each column must correspond to one single anthropogenic feature and first column to flow deviation)

    src = f"{dataDirectory}/anthropogenic_features_count_by_subcatchment_raw.gpkg"
    gdf = gpd.read_file(src)
    listStations = list(set(gdf['code_station']))
    listFeatures = list(set(gdf['feature']))
    listAllStation = []
    
    for station in listStations:

        gdc = gdf.loc[gdf['code_station'] == station]
        listOneStation = []
        
        #Build temporary dataframe for saving columns
        frames = {'code_station':[],f"MeanAnnualCoV_{str(timeRange[0])}{str(timeRange[1])}":[],'geometry':[]}
        frames['code_station'].append(station)
        frames[f"MeanAnnualCoV_{str(timeRange[0])}{str(timeRange[1])}"].append(list(gdc[f"MeanAnnualCoV_{str(timeRange[0])}{str(timeRange[1])}"])[0])
        frames['geometry'].append(list(gdc['geometry'])[0])
        tmp1 = pd.DataFrame(frames)

        for feat in listFeatures:

            gdcc = gdc.loc[gdc['feature'] == feat]
            #Build the called feature as a column
            gdcc.loc[:,f"{str(feat)}"] = gdcc['entities_per_km2']
            tmp2 = gdcc[['code_station',f"{str(feat)}"]]
            #Append to saving list
            listOneStation.append(tmp2)
            del gdcc

        tmp3 = functools.reduce(lambda left, right: pd.merge(left, right, on='code_station'), listOneStation)
        tmp4 = pd.merge(tmp1,tmp3,on='code_station')
        listAllStation.append(tmp4)
        del tmp1, tmp2, tmp3, tmp4, frames
        del gdc
    
    #Concatenate all dataframes and save to disk    
    df = pd.concat(listAllStation)
    if 'occupation_class-128.0' in df.columns:
        df.drop('occupation_class-128.0',axis=1,inplace=True)
    else:
        pass
    gdf = gpd.GeoDataFrame(df)
    gdf.set_geometry('geometry',inplace=True)
    gdf.set_crs(crs=f"EPSG:{str(AoI_EPSG)}",inplace=True)
    dst = f"{dataDirectory}/anthropogenic_features_count_by_subcatchment_structured.gpkg"
    gdf.to_file(dst) #Polygon dataframe with columns 'feature', 'entities_per_km2', 'geometry', 'MeanAnnualCoV_timeRange', 'code_station'
    del gdf


    print("Compute Pearson correlation matrix")

    #Compute matrix
    src = f"{dataDirectory}/anthropogenic_features_count_by_subcatchment_structured.gpkg"
    gdf = gpd.read_file(src)
    gdf.drop(labels=['geometry','code_station'],axis=1,inplace=True)
    
    if len(gdf) >= 2:
    
        #Join correlation labels to CLC nomenclature
        src = "CLC_pixel2class_nomenclature.csv"
        nomm = pd.read_csv(src)
        nomm['pixelValue'] = nomm['pixelValue'].astype(float)
        occupationClassesAll = [float(x) for x in occupationClassesAll]
        if float(-128) in occupationClassesAll:
            occupationClassesAll.remove(float(-128))
        else:
            pass
        for occ in occupationClassesAll:
            pixValue = float(occ)
            tmp = nomm.loc[nomm['pixelValue'] == pixValue]
            pixLabel = list(tmp['pixelLabel'])[0]
            values = [np.round(x,decimals=2) for x in list(gdf[f"occupation_class{str(occ)}"])]
            gdf.loc[:,f"{str(pixLabel)}"] = values
            #Clean dataframe
            gdf.drop(f"occupation_class{str(occ)}",axis=1,inplace=True)
            gdf.dropna(axis=1,how='all',inplace=True)
        #Compute correlation and significance
        corr_frames = {}
        pval_frames = {}
        idx = []
        for x in list(gdf.columns):
            idx.append(f"{str(x)}")
            for y in list(gdf.columns):
                c, p = scipy.stats.pearsonr(list(gdf[f"{str(x)}"]), list(gdf[f"{str(y)}"]), alternative='two-sided')
                if f"{str(y)}" not in corr_frames:
                    corr_frames[f"{str(y)}"] = []
                    corr_frames[f"{str(y)}"].append(c)
                    pval_frames[f"{str(y)}"] = []
                    pval_frames[f"{str(y)}"].append(p)
                else:
                    corr_frames[f"{str(y)}"].append(c)
                    pval_frames[f"{str(y)}"].append(p)
        corr = pd.DataFrame(corr_frames,index=idx)
        pval = pd.DataFrame(pval_frames,index=idx)
        #corr = gdf.corr(method='pearson')
        #pval = corr.map(lambda x: np.round(scipy.stats.norm.cdf(x),decimals=2))
        #Clean dataframe from (for some reason) empty rows
        corr.dropna(axis=1,how='all',inplace=True)
        corr.dropna(axis=0,how='all',inplace=True)
        pval.dropna(axis=1,how='all',inplace=True)
        pval.dropna(axis=0,how='all',inplace=True)
        #Write to disk
        dst = f"{analysisDirectory}/corrMatrix_flowDeviations_vs_anthropogenicFeatures.csv"
        corr.to_csv(dst)
        dst = f"{analysisDirectory}/pvalMatrix_flowDeviations_vs_anthropogenicFeatures.csv"
        pval.to_csv(dst)
        #plot corr as a heatmap
        plt.figure(figsize=(10, 8))  # Adjust the size to your needs
        mask = corr >= float(0.99)
        sns.heatmap(corr, annot=True, fmt='.1e', cmap='vlag', center=0, annot_kws={"fontsize": 4}, mask=mask)
        plt.title('Pearson Correlation Matrix between CoV and Land Occupation')
        plt.savefig(f"{analysisDirectory}/corrMatrix_flowDeviations_vs_anthropogenicFeatures.png", dpi=300, bbox_inches='tight')
        plt.close()
        #plot pval as a heatmap
        plt.figure(figsize=(10, 8))  # Adjust the size to your needs
        mask = pval <= float(1e-300)
        sns.heatmap(pval, annot=True, fmt='.1e', cmap='Purples_r', center=0, annot_kws={"fontsize": 4}, mask=mask)
        plt.title('p-values of Pearson Correlation Matrix between CoV and Land Occupation')
        plt.savefig(f"{analysisDirectory}/pvalMatrix_flowDeviations_vs_anthropogenicFeatures.png", dpi=300, bbox_inches='tight')
        plt.close()
        del src, gdf, dst
    
    
        print("Extract 1st column of correlation and pvalue matrices as a separate file")
        
        corr_col = list(corr[f"MeanAnnualCoV_{timeRange[0]}{timeRange[1]}"])
        pval_col = list(pval[f"MeanAnnualCoV_{timeRange[0]}{timeRange[1]}"])
        idx = list(corr.columns)
        frames = {'land_occ': idx, 'corr_coeff': corr_col, 'p_value':pval_col}
        df = pd.DataFrame(frames)
        df.to_csv(f"{analysisDirectory}/corrMatrix_flowDeviations_only.csv",index=False)

    else:
        print("WARNING : Correlation matrix is not possible with only one catchment, adjust the time range to increase the number of catchments") 

    with open(f"{wd}/log.txt", 'a') as file:
        file.write(f"MODULE5.py Elapsed Time: {str(datetime.datetime.now()-globstart)}\n")

else:
    pass


##########################
# Module 6: Extra graphs #
##########################

if runModule6 is True:

    globstart = datetime.datetime.now()

    print("MODULE 6")

    print("Plot Cumulative Distribution Function of monthly CoV forall month and station")
    
    gdf = gpd.read_file(f"{analysisDirectory}/stations_observations_mmf_average_{timeRange[0]}{timeRange[1]}_points.gpkg")
    listStations = list(gdf['code_station'])
    gdf.set_index('code_station',inplace=True)
    col_list = [f"CoV_month{x}_{timeRange[0]}{timeRange[1]}" for x in range(1,13)]
    gdfc = gdf[col_list]
    gdfcT = gdfc.T #station_codes go now column-wise and CoVs go row-wise

    covList = []
    plt.figure(figsize=(10, 6))
    
    for stas in listStations:
        gdfcTc = gdfcT[[stas]]
        gdfcTc.dropna(axis=0,inplace=True)
        cov = list(gdfcTc[stas])
        covList.append(cov)
        # Create the plot
        #plt.plot([i for i in range(1,13)], cov, label=f"{stas}")
    values = []
    for x in range(len(covList)):
        values += covList[x]
    values_NoNaN = [x for x in values if np.float64(x) != np.nan]
    #Calculate the cumulative distribution values
    counts, bin_edges = np.histogram(values_NoNaN, bins=100, density=True)
    cdf = np.cumsum(counts)
    cdf = cdf / cdf[-1]

    # Create the plot
    plt.plot(bin_edges[1:], cdf, label='CDF')

    # Add title and labels
    plt.title(f"Cumulative Distribution Function of the Coefficients of Variation over the period {timeRange[0]}-{timeRange[1]}")
    plt.xlabel('Monthly coefficient of variation')
    plt.ylabel('Cumulative Distribution Function')
    #plt.legend()
    plt.savefig(f"{analysisDirectory}/monthlyCoV_CDF_{timeRange[0]}{timeRange[1]}.png", dpi=300, bbox_inches='tight')
    plt.close()


    print("Compute the ratio of all monthly CoVs to average monthly CoVs")

    gdf_all = gpd.read_file(f"{dataDirectory}/mmf_all_{timeRange[0]}{timeRange[1]}_subcatchments.gpkg")
    gdf_average = gpd.read_file(f"{analysisDirectory}/stations_observations_mmf_average_{timeRange[0]}{timeRange[1]}_subcatchments.gpkg")
    
    gdf_all['date'] = gdf_all['date'].astype("string") 
    listStations = list(set(gdf_all['code_station']))
    
    list_gdfAll = []
    list_gdfAverage = []
    
    for s in listStations:
        
        print(f"Station {str(s)} out of {str(len(listStations))}")
        
        gdf_all_cut = gdf_all.loc[gdf_all['code_station'] == s]
        gdf_average_cut = gdf_average.loc[gdf_average['code_station'] == s]

        list2plot = []
        
        for i in range(1,13):
            
            if i <= 9:
                pattern = rf'^\d{{4}}-0{str(i)}-\d{{2}}$'
            else:
                pattern = rf'^\d{{4}}-{str(i)}-\d{{2}}$'
                
            gdf_all_cut2 = gdf_all_cut.loc[gdf_all_cut['date'].str.contains(pattern, regex=True)]
            gdf_all_cut2.loc[:,"currCoV"] = gdf_all_cut2['MMF_std']/gdf_all_cut2['MMF_mean']
            avgCoV = gdf_all_cut2['currCoV'].mean(skipna=True)
            gdf_all_cut2.loc[:,"avgCoV"] = avgCoV
            gdf_all_cut2.loc[:,"deltaCoV"] = (gdf_all_cut2['currCoV']-avgCoV)/avgCoV
            lambda_function = lambda x: 1 if (x >= percentChange) | (x <= -percentChange) else (np.nan if x is np.nan else 0)
            gdf_all_cut2.loc[:,"CoV_overshoot"] = gdf_all_cut2['deltaCoV'].apply(lambda_function)
            if len(gdf_all_cut2.dropna(axis=0,inplace=False)) != 0:
                overshot_fq = gdf_all_cut2.loc[:,'CoV_overshoot'].sum() / len(gdf_all_cut2.dropna(axis=0,inplace=False))
            else:
                overshot_fq = np.nan
            gdf_average_cut.loc[:,f"overshoot_fq_month{str(i)}"] = overshot_fq
            list_gdfAll.append(gdf_all_cut2)
            list2plot.append(gdf_all_cut2)

        #Plot graph of monthly deltaCoV
        #Intermediate concatenation for all months and for one station 
        tmp = pd.concat(list2plot)
        dates = list(tmp['date'])
        values = list(tmp['deltaCoV'])
        tmp['date'] = pd.to_datetime(tmp['date'])
        tmp = tmp.sort_values(by='date')
        # Create the plot
        plt.figure(figsize=(10, 5))
        # Plot the main line
        plt.plot(tmp['date'], tmp['deltaCoV'], label='CoV anomaly', color = 'black', marker='o', linestyle='-', markersize=2, linewidth=0.5)
        # Add the corridor as a filled rectangle
        corridor_min = percentChange
        corridor_max = -percentChange
        plt.fill_between(dates, corridor_min, corridor_max, color='lightblue', alpha=0.3, label='maximum anomaly corridor')
        # Add labels and title
        #datesDatetime = np.array(dates, dtype='datetime64')
        #datesOrdered = np.sort(datesDatetime)
        #datesStringed = [str(x) for x in list(datesOrdered)]
        #i = int(np.floor(len(datesStringed)/20))
        #ticks = [datesStringed[x*i] for x in range(20)]

        nb_ticks = min(20,len(tmp['date']))
        i = len(tmp['date']) // nb_ticks
        ticks_position = tmp['date'][::i]
        if nb_ticks == 20:
            ticks_strings = [str(x) for x in ticks_position]
            ticks_labels = [x[:-15] for x in ticks_strings]
        else:
            ticks_labels = ticks_position
        plt.xticks(ticks=ticks_position,labels=ticks_labels,rotation=90)
        plt.xlabel('Date')
        plt.ylabel('CoV anomaly (monthly_CoV/average_CoV)')
        plt.title(f"Monthly Coefficient of Variation anomaly for station {str(s)} over the period {timeRange[0]}-{timeRange[1]}")
        plt.legend()
        # Show the plot
        plt.text(0.03, 0.03, '(c) Q. DASSIBAT -- CC BY 4.0', fontsize=5, color='gray', transform=plt.gcf().transFigure)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{analysisDirectory}/monthly_CoV_anomaly_station_{str(s)}.png", dpi=300, bbox_inches='tight')
        plt.close()
        del tmp
        
        #Compute mean annual frequency of overshoot for each station
        col_list = [f"overshoot_fq_month{str(x)}" for x in range(1,13)]
        tmp = gdf_average_cut[col_list]
        tmpT = tmp.T #gives a df with one column 'code_station' and former columns turned to rows
        tmpT.dropna(inplace=True)
        col_name = list(tmpT.columns)[0]
        mean = tmpT[col_name].mean()
        gdf_average_cut.loc[:,'overshoot_fq_meanAnnual'] = mean
        
        list_gdfAverage.append(gdf_average_cut)
        
    #Concat dfs and export
    gdf_all_export = pd.concat(list_gdfAll)
    gdf_average_export = pd.concat(list_gdfAverage)
    #Export as polygons
    gdf_average_export.to_file(f"{analysisDirectory}/stations_observations_mmf_average_{timeRange[0]}{timeRange[1]}_subcatchments.gpkg")
    gdf_all_export.to_file(f"{dataDirectory}/mmf_all_{timeRange[0]}{timeRange[1]}_subcatchments.gpkg")
    #Export as points 
    src = f"{analysisDirectory}/stations_observations_mmf_average_{timeRange[0]}{timeRange[1]}_points.gpkg"
    points = gpd.read_file(src)
    points_cut = points[['code_station','geometry']]
    gdf_average_export_cut = gdf_average_export.drop(['geometry'],axis=1,inplace=False)
    points_merged = points_cut.merge(gdf_average_export_cut,left_on='code_station',right_on='code_station',how ='left')
    gdf = gpd.GeoDataFrame(points_merged,geometry='geometry',crs=f"EPSG:{str(AoI_EPSG)}")
    dst = f"{analysisDirectory}/stations_observations_mmf_average_{timeRange[0]}{timeRange[1]}_points.gpkg"
    gdf.to_file(dst)

    #Plot fancy map of mean annual frequency of overshoot
    #Prepare pointsLayer to display: retrieve position of stations in accuflux map
    points2accuflux = gpd.read_file(f"{tmpDirectory}/stations2accuflux_analyzed.gpkg")
    points = gpd.read_file(f"{analysisDirectory}/stations_observations_mmf_average_{str(timeRange[0])}{str(timeRange[1])}_points.gpkg")
    points2accuflux_cut = points2accuflux[['code_station','geometry']]
    points_cut = points.drop('geometry',axis=1)
    m = points2accuflux_cut.merge(points_cut,left_on='code_station',right_on='code_station',how='left')
    m.to_file(f"{analysisDirectory}/stations2accuflux_observations_mmf_average_{str(timeRange[0])}{str(timeRange[1])}_points.gpkg")
    #Plot
    srcPolygData = f"{analysisDirectory}/stations_observations_mmf_average_{str(timeRange[0])}{str(timeRange[1])}_subcatchments.gpkg"
    srcPoints = f"{analysisDirectory}/stations2accuflux_observations_mmf_average_{str(timeRange[0])}{str(timeRange[1])}_points.gpkg"
    srcPolygNodata = f"{dataDirectory}/subcatchments_all.gpkg"
    dst = f"{analysisDirectory}/VariabilityAnomaly_{str(timeRange[0])}{str(timeRange[1])}_map.png"
    polyg = f"overshoot_fq_meanAnnual"
    label = f"overshoot_fq_meanAnnual"
    title = f"Mean annual anomaly of flow variability over the period {str(timeRange[0])}-{str(timeRange[1])}"
    hydrofunc.make_map_LabelsOnPoints(srcPolygData,polyg,label,srcPoints,srcPolygNodata,title,dst)
    del srcPolygData, polyg, label, title, dst, srcPoints, srcPolygNodata

    #Normalization of overshoot annual frequency to [0,12] as for ESTRESS indicator in Rockstrom 2023
    #src = "/home/q.dassibat/python_proj/env3/modules/HydroTrendsFrance/HydroTrendsSudLoireFinal20002020/analysis/stations_observations_mmf_average_20002020_subcatchments.gpkg"
    #gdf = gpd.read_file(src)
    #values = list(gdf['overshoot_fq_meanAnnual'])
    #values_changed = [int(np.trunc(x*12)) for x in values]
    #gdf.loc[:,'overshoot_fq_meanAnnual_normalizedTo12'] = values_changed
    #gdf.to_file()


    print("Plot correlation matrix of CoV against land occupation as a histogram")

    df = pd.read_csv(f"{analysisDirectory}/corrMatrix_flowDeviations_only.csv")
    df.sort_values(by=['corr_coeff'],ascending=True, inplace=True)
    #Drop CoV row 
    dfc = df.loc[df['land_occ'] != f"MeanAnnualCoV_{str(timeRange[0])}{str(timeRange[1])}"]
    # Set the positions and width for the bars
    bar_width = 0.35
    x = np.arange(len(dfc['land_occ']))
    # Plot the bars
    fig, ax = plt.subplots(figsize=(12, 10))
    bar1 = ax.bar(x - bar_width/2, dfc['corr_coeff'], bar_width, label='corr_coeff')
    bar2 = ax.bar(x + bar_width/2, dfc['p_value'], bar_width, label='p_value')
    # Add labels, title, and legend
    ax.set_ylim(-1.1,1.1) 
    ax.set_yticks(np.arange(-1, 1, step=0.05))
    ax.set_xlabel('Land occupation classes or features')
    ax.set_ylabel('Pearson correlation coefficients and p-values')
    ax.set_title(f"Correlation between flow variability and land occupation over the period {str(timeRange[0])}-{str(timeRange[1])}")
    ax.set_xticks(x)
    ax.set_xticklabels(dfc['land_occ'],rotation=90)
    ax.legend()
    #Annotate
    for bar in bar1:
        height = bar.get_height()
        if height >= 0:
            offset = 3
            va = 'bottom'
        else:
            offset = -height + 3  # Position just above zero
            va = 'top'
        ax.annotate(f"{height:.1e}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, offset),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va=va, fontsize=5,rotation=90)
    for bar in bar2:
        height = bar.get_height()
        if height >= 0:
            offset = 3
            va = 'bottom'
        else:
            offset = -height + 3  # Position just above zero
            va = 'top'
        ax.annotate(f"{height:.1e}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, offset),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va=va, fontsize=5,rotation=90)
    # Display the plot
    fig.tight_layout()
    ax.grid(axis='x')
    plt.savefig(f"{analysisDirectory}/corrMatrix_flowDeviations_only.png", dpi=300, bbox_inches='tight')
    plt.close()



    with open(f"{wd}/log.txt", 'a') as file:
        file.write(f"MODULE6.py Elapsed Time: {str(datetime.datetime.now()-globstart)}\n")

else:
    pass













