####################
# Import libraries #
####################

import hydrofunc
import os
import geopandas as gpd
import pandas as pd
import datetime
import numpy as np
from shapely import wkt

##################
# Set parameters # 
##################

#Area of Interest
AoI_filePath = "envelope_scot_sl.gpkg" #path to file with extension
AoI_fileIsRaster = False #False if AoI is a vector file (.shp, .gpkg...); True if AoI is a raster file (.tif...)
AoI_EPSG = 4326 #ESPG code in which AoI_filePath is projected [int] e.g. 4326

#Period of Interest
timeRange = [1960,2020] #e.g. [2000,2020] for Water Balance Model (Wisser et al, 2010) as used in Rockstrom et al, 2023 

#Modules to run
runModule0 = False #True if Module 0 needs to be ran ; False if not
runModule1 = False 
runModule2 = True

#Miscellaneous
flushAllDirectories = False # Usefull to study a new AoI
operationStatus = False #True will keep only those stations that are still operating now a days; False will keep them all 
plottingMMF = False #True will generate the scatter plot with deviation bars of MMF for each station; False will not generate plot 
plottingMKtest = True #true will generate the scatter plot of the series and the trend of Mann-Kendall regression ; False will not generate plot
flushDisk = False #True will delete folder "./tmp" and its files; False will keep it
accThreshold = 1e3 #Defines which pixel is considered part of an actively flowing river, with the value of 1000 as a presumptive standard

################
# User warning #
################

# Module 0 and Module 1 need to be ran before Module 2, otherwise Module 2 is lacking material

#####################
# Flush directories #
#####################

# Usefull to study a new AoI

wd = os.getcwd() #returns the same wd as the wd where Python is launched in terminal
dataDirectory = f"{wd}/data"
analysisDirectory = f"{wd}/analysis"
tmpDirectory = f"{wd}/tmp"

if flushAllDirectories is True:

    if os.path.isdir(dataDirectory) is True:
        entries = os.listdir(dataDirectory)
        for entry in entries:
            os.remove(f"{dataDirectory}/{entry}")
        #os.rmdir(dataDirectory)
    else:
        pass
    if os.path.isdir(analysisDirectory) is True:
        entries = os.listdir(analysisDirectory)
        for entry in entries:
            os.remove(f"{analysisDirectory}/{entry}")
        #os.rmdir(analysisDirectory)
    else:
        pass
    if os.path.isdir(tmpDirectory) is True:
        entries = os.listdir(tmpDirectory)
        for entry in entries:
            os.remove(f"{tmpDirectory}/{entry}")
        #os.rmdir(tmpDirectory)
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
    
    src = f"{dataDirectory}/stations_locations.gpkg"
    dst = f"{dataDirectory}/stations_observations_mdf_{str(timeRange[0])}{str(timeRange[1])}.gpkg"
    period = [f"{timeRange[0]}-01-01",f"{timeRange[1]}-01-01"] #["yyyy-mm-dd","yyyy-mm-dd"]
    hydrofunc.requestFrontend_observations_hubeau(src,dst,tRange=period)
    del src, dst

    print("Download SRTM-30 DEM from opentopography API")
    
    dst = f"{dataDirectory}/DEM.tif"
    hydrofunc.request_DEM(AoI_bbox,AoI_EPSG,dst)


    print("Total Elapsed Time: ", datetime.datetime.now()-globstart)
    
    with open(f"{wd}/log.txt", 'a') as file:
        file.write(f"MODULE0.py Elapsed Time: {str(datetime.datetime.now()-globstart)}\n")

else:
    pass



#####################################################
# Module 1: create the subcatchment of each station #
#####################################################

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

    print("Generate subcatchments")
    #Join stations to acculfux
    gdf = gpd.read_file(f"{dataDirectory}/stations_locations.gpkg")
    ids = np.arange(1,len(gdf)+1,1)
    gdf.loc[:,'id'] = list(ids)
    gdf.to_file(f"{dataDirectory}/stations_locations.gpkg")
    points = f"{dataDirectory}/stations_locations.gpkg"
    layer = 'id'
    acc = f"{tmpDirectory}/accuflux_geq{str(accThreshold)}.tif"
    dst = f"{tmpDirectory}/stations2accuflux.gpkg"
    hydrofunc.join_points_to_pixels(points,layer,acc,AoI_EPSG,dst)
    del points, gdf, layer, acc, dst
    #Create subcatchments
    points = f"{tmpDirectory}/stations2accuflux.gpkg"
    ldd = f"{tmpDirectory}/LDD.map"
    dst = f"{tmpDirectory}/subcatchments.map"
    hydrofunc.create_subcatchments(points,ldd,dst,cln)
    del points, ldd, dst
    #Convert to vector file
    src = f"{tmpDirectory}/subcatchments.map"
    dst = f"{dataDirectory}/subcatchments.gpkg"
    hydrofunc.raster_to_polygons(src,dst,AoI_EPSG,'catchment_id',zRestriction=None)
    del src, dst
    #Add station codes as catchments ids
    stations = gpd.read_file(f"{dataDirectory}/stations_locations.gpkg")
    catchments = gpd.read_file(f"{dataDirectory}/subcatchments.gpkg")
    stations = stations[['id','code_station']]
    m = catchments.merge(stations, left_on='catch_id', right_on='id', how='left')
    m.drop(['catch_id','id'],axis=1,inplace=True)
    m.to_file(f"{dataDirectory}/subcatchments.gpkg")

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
    #Make a .csv file
    gdf = gpd.read_file(f"{analysisDirectory}/stations_observations_mmf_average_{str(timeRange[0])}{str(timeRange[1])}_points.gpkg")
    gdf.drop('geometry', axis=1, inplace=True)
    gdf.to_csv(f"{analysisDirectory}/stations_observations_mmf_average_{str(timeRange[0])}{str(timeRange[1])}.csv")
    #Join with subcatchment vector file
    catch = gpd.read_file(f"{dataDirectory}/subcatchments.gpkg")
    m = catch.merge(gdf,left_on='code_station',right_on='code_station',how='left')
    m.to_file(f"{analysisDirectory}/stations_observations_mmf_average_{str(timeRange[0])}{str(timeRange[1])}_subcatchments.gpkg")


    
    print("Compute MMF for all month for each station and MK regression")
    
    #Read observations file
    obs = gpd.read_file(f"{dataDirectory}/stations_observations_mdf_{str(timeRange[0])}{str(timeRange[1])}.gpkg")
    stationsList = list(set(obs['code_station']))

    #Compute MMF for each month and each station and the compute Mann-Kendall trend for average and stdev

    list_df = [] #to save the df of each station and further merge them
    
    for station in stationsList:
    
        print(f"Station {station}")

        frames = {'code_station': [],
           'z_value_mean': [],
          'slope_mean' : [],
          'intercept_mean': [],
          'globalTrend_mean': [],
          'confidence_mean' : [],
          'z_value_std': [],
          'slope_std' : [],
          'intercept_std': [],
          'globalTrend_std': [],
          'confidence_std' : []}

        frames['code_station'].append(station)

        print("Compute MMF for all months")
        dst = f"{dataDirectory}/mmf_all_station_{str(station)}_{str(timeRange[0])}{str(timeRange[1])}_points.gpkg"
        mmf = hydrofunc.compute_MeanMonthlyFlow_all(str(station),obs,dst,timeRange,AoI_EPSG)
        #Export as .csv
        if os.path.exists(dst) is True:
            mmf.drop('geometry',axis=1,inplace=True)
            mmf.to_csv(f"{dataDirectory}/mmf_all_station_{str(station)}_{str(timeRange[0])}{str(timeRange[1])}.csv")
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
            frames['confidence_mean'].append(c)
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
            frames['confidence_std'].append(c)
            #Store data 
            list_df.append(pd.DataFrame(frames))
        else:
            print(f"File does not exist: {src}")
        
    #Generate and export dataframe     
    df = pd.concat(list_df)     
    #As a .csv     
    dst = f"{analysisDirectory}/MannKendallRegression_{str(timeRange[0])}{str(timeRange[1])}.csv"
    df.to_csv(dst)
    del dst
    #As a point-geometry .gpkg
    dst = f"{analysisDirectory}/MannKendallRegression_{str(timeRange[0])}{str(timeRange[1])}_points.gpkg"
    points = gpd.read_file(f"{dataDirectory}/stations_locations.gpkg")
    points_cut = points[['code_station','geometry']]
    #only keep points that have valid geometry
    #points_cut2 = points_cut[~points_cut.geometry.isnull()]
    #points_cut2['geometry'] = points_cut2['geometry'].astype("string")
    join = df.merge(points_cut,left_on='code_station',right_on='code_station',how = 'left')
    #join['geometry'] = join['geometry'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(join,geometry='geometry',crs=f"EPSG:{str(AoI_EPSG)}")
    gdf.to_file(dst)
    del join, gdf, dst
    #As a subcatchment-geometry .gpkg
    dst = f"{analysisDirectory}/MannKendallRegression_{str(timeRange[0])}{str(timeRange[1])}_subcatchments.gpkg"
    catch = gpd.read_file(f"{dataDirectory}/subcatchments.gpkg")
    catch_cut = catch[['code_station','geometry']]
    #only keep catchments that have valid geometry
    #catch_cut2 = catch_cut[~catch_cut.geometry.isnull()]
    #catch_cut2['geometry'] = catch_cut2['geometry'].astype("string")
    join = df.merge(catch_cut,left_on='code_station',right_on='code_station',how='left')
    #join['geometry'] = join['geometry'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(join,geometry='geometry',crs=f"EPSG:{str(AoI_EPSG)}")
    gdf.to_file(dst)
    del join, gdf, dst

    print("Generate fancy map")
    
    #For the trend on the flow mean
    src = f"{analysisDirectory}/MannKendallRegression_{str(timeRange[0])}{str(timeRange[1])}_subcatchments.gpkg"
    dst = f"{analysisDirectory}/MannKendallRegression_{str(timeRange[0])}{str(timeRange[1])}_GlobalTrendOfMean_map.png"
    polyg = 'globalTrend_mean'
    label = 'confidence_mean'
    title = f"Gobal trend of Mean Monthly Flows (colors) and their confidence level (labels) over the period {str(timeRange[0])}-{str(timeRange[1])}"
    hydrofunc.make_map(src,polyg,label,title,dst)
    del src, polyg, label, title, dst
    
    #For the trend on the flow deviation
    src = f"{analysisDirectory}/MannKendallRegression_{str(timeRange[0])}{str(timeRange[1])}_subcatchments.gpkg"
    dst = f"{analysisDirectory}/MannKendallRegression_{str(timeRange[0])}{str(timeRange[1])}_GlobalTrendOfDeviation_map.png"
    polyg = 'globalTrend_std'
    label = 'confidence_std'
    title = f"Global trend of Monthly Flow Deviations (colors) and their confidence level (labels) over the period {str(timeRange[0])}-{str(timeRange[1])}"
    hydrofunc.make_map(src,polyg,label,title,dst)
    del src, polyg, label, title, dst
    
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
        file.write(f"MODULE2.py Elapsed Time: {str(datetime.datetime.now()-globstart)}\n")


else:
    pass





