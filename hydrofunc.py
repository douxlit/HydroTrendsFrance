def extract_Rasterbbox(srcFile):

    """
    srcFile: path/to/source/raster/file.tif [string]

    output : bbox of srcFile as a string "xmin ymin xmax ymax" [string]
    """

    from osgeo import gdal
    gdal.UseExceptions()

    src = gdal.Open(srcFile)

    upx, xres, xskew, upy, yskew, yres = src.GetGeoTransform()
    cols = src.RasterXSize
    rows = src.RasterYSize
     
    ulx = upx + 0*xres + 0*xskew
    uly = upy + 0*yskew + 0*yres
     
    llx = upx + 0*xres + rows*xskew
    lly = upy + 0*yskew + rows*yres
     
    lrx = upx + cols*xres + rows*xskew
    lry = upy + cols*yskew + rows*yres
     
    urx = upx + cols*xres + 0*xskew
    ury = upy + cols*yskew + 0*yres

    bbox = f"{llx} {lly} {urx} {ury}" #xmin (llx) ymin (lly) xmax (urx) ymax (ury)

    return bbox


def extract_Vectorbbox(srcFile):

    """
    srcFile: path/to/source/vector/file.gpkg [string] or any fiona-driver compatible format

    output : bbox of srcFile as a string "xmin ymin xmax ymax" [string]
    """

    import geopandas as gpd

    gdf = gpd.read_file(srcFile)
    bbox = f"{gdf.total_bounds[0]} {gdf.total_bounds[1]} {gdf.total_bounds[2]} {gdf.total_bounds[3]}"

    del gdf

    return bbox
    


def request_locations_hubeau(bbox,dstFile,operating):

    """
    Makes a request to https://hubeau.eaufrance.fr/api/v1/hydrometrie/referentiel/stations? and returns a geoDataFrame with station codes
    with colomuns ['code_station','date_ouverture_station','date_fermeture_station','geometry']

    bbox: coordinates as a string "xmin ymin xmax ymax" to define the request's spatial extent [string]
    dstFile: /path/to/destination/layer.gpkg containing the geoDataFrame projected in EPSG:4326 [string]
    operating: geoDataFrame will be filtered to only still operating stations if True, or will keep all returned stations if False [python object]
    """

    #Import libraries
    import geopandas as gpd
    import pandas as pd
    import requests
    import json
    import shapely

    #Split coordinates string 
    l = bbox.split()
    xmin = l[0]
    ymin = l[1]
    xmax = l[2]
    ymax = l[3]
    
    print("Request to hubeau.eaufrance.fr")
    size = 10000 #maximum depth of the json response from hubeau.eaufrance.fr 
    url = f"https://hubeau.eaufrance.fr/api/v1/hydrometrie/referentiel/stations?size={size}&pretty&bbox={xmin},{ymin},{xmax},{ymax}"
    r = requests.get(url, allow_redirects=True)
    
    #Convert data from json to dataframe
    
    data = json.loads(r.content)
    df_stations = pd.json_normalize(data['data'])
    df_stations['longitude_station'] = df_stations['longitude_station'].astype("string")
    df_stations['latitude_station'] = df_stations['latitude_station'].astype("string")
    df_stations['coordonnees']="POINT ("+df_stations['longitude_station']+" "+df_stations['latitude_station']+")"
    geometry = df_stations['coordonnees'].map(shapely.wkt.loads)
    df_stations.drop('coordonnees', axis=1, inplace=True)
    gdf_stations = gpd.GeoDataFrame(df_stations, crs="EPSG:4326", geometry=geometry)
    
    #Filter dataframe to only specific columns and set integer index
    champs = ['code_station','date_ouverture_station','date_fermeture_station','geometry']
    gdf_stations1 = gdf_stations[champs]

    #Filter dataframe according to the value of operating parameter and set integer index
    if operating is True:
        m = gdf_stations1['date_fermeture_station'].isnull()
        gdf_stations2 = gdf_stations1.loc[m]
        m = gdf_stations2['date_ouverture_station'].notna()
        gdf_stations3 = gdf_stations2.loc[m]
        gdf_stations3.set_index(pd.Index([i for i in range(len(gdf_stations3))]),inplace=True)
    else:
        gdf_stations3 = gdf_stations1
        gdf_stations3.set_index(pd.Index([i for i in range(len(gdf_stations3))]),inplace=True)
    
    #Write dataframe to disk as .gpkg
    gdf_stations3.to_file(dstFile)

    return


def requestBackend_observations_hubeau(stationCode,start,end):

    """
    stations: code_station [string]
    start: YYYY-MM-DD [string]
    end: YYYY-MM-DD [string]
    """

    #Import libraries 
    import datetime
    import geopandas as gpd
    import pandas as pd
    import requests
    import json
    import numpy as np

    #Request observations
    
    grandeur = "QmJ"
    size = 10000
    
    url = f"https://hubeau.eaufrance.fr/api/v1/hydrometrie/obs_elab?\
size={str(size)}&pretty&\
code_entite={str(stationCode)}&\
grandeur_hydro_elab={grandeur}&\
date_debut_obs_elab={str(start)}&\
date_fin_obs_elab={str(end)}"
    
    r = requests.get(url, allow_redirects=True)
        
    if  r.status_code != 503:
    
        data = json.loads(r.content)
        df_response = pd.json_normalize(data['data'])

        #Clear response dataframe and store it if it is not empty

        if df_response.empty is not True:

            #Select specific fields
            fields = ['code_station','date_obs_elab','resultat_obs_elab','grandeur_hydro_elab','libelle_statut']
            dfc = df_response[fields]
            del df_response
        
        else:
            print(f"null response from url {url}")
            
            frames = {'code_station':[str(stationCode)],
                      'date_obs_elab':[np.float32(np.nan)],
                      'resultat_obs_elab':[np.float32(np.nan)],
                      'grandeur_hydro_elab':[np.float32(np.nan)],
                      'libelle_statut':[np.float32(np.nan)]}
            dfc = pd.DataFrame(frames)
    
    else:
        print(f"error 503 from url {url}")
        
        frames = {'code_station':[str(stationCode)],
                      'date_obs_elab':[np.float32(np.nan)],
                      'resultat_obs_elab':[np.float32(np.nan)],
                      'grandeur_hydro_elab':[np.float32(np.nan)],
                      'libelle_statut':[np.float32(np.nan)]}
        dfc = pd.DataFrame(frames)

    
    return dfc



def requestFrontend_observations_hubeau(srcFile,dstFile,tRange=None):

    """
    Makes a request to https://hubeau.eaufrance.fr/api/v1/hydrometrie/obs_elab? and returns a geoDataFrame with station observations (Mean Daily Flows)
    with colomuns ['code_station','date_obs_elab','resultat_obs_elab','geometry']
    
    tRange [optionnal] [list]: default is None, meaning their is no time range specified (sation observations are extracted from each station's openning date to present date or to the station's closing date). If tRange has both values specified, e.g. ["yyyy-mm-dd","yyyy-mm-dd"], values are retrieved for this time range.

    srcFile: /path/to/source/layer.gpkg containing the geoDataFrame of station codes #index must be integers [string]
    dstFile: /path/to/destination/layer.gpkg [string]
    """

    #Import libraries 
    import numpy as np
    import datetime
    import geopandas as gpd
    import pandas as pd
    import requests
    import json

    #Read station location file
    gdf = gpd.read_file(srcFile)
    gdf.drop_duplicates(subset='code_station',inplace=True)
    gdf.reset_index(drop=True,inplace=True)
    
    #Itération sur les stations

    stations_observ = []
    
    for s in range(len(gdf)):
        
        stas = gdf.loc[s,'code_station']
        geom = gdf.loc[s,'geometry']
        print("Station",stas)
        
        #Calcul du nombre de requêtages nécessaires 

        if tRange is None:
        
            ##Selon la date d'ouverture de la station (par défaut) : depuis la date d'ouverture jusqu'à aujourd'hui ou jusqu'à la date de fermeture
            
            Nmax = 10000 #profondeur max de la requête
            
            mask = gdf['code_station'].str.match(str(stas))
            ouv = list(gdf.loc[mask,"date_ouverture_station"])
            ouv_string = ouv[0][:10]
            ouv_datetime = datetime.datetime.strptime(ouv_string, '%Y-%m-%d')
            
            ajd = str(datetime.datetime.now())
            ajd_string = ajd[:10]
            ajd_datetime = datetime.datetime.strptime(ajd_string, '%Y-%m-%d')
            arret = list(gdf.loc[mask,"date_ouverture_station"])
            if len(arret) != 0:
                arret_string = ouv[0][:10]
                arret_datetime = datetime.datetime.strptime(arret_string, '%Y-%m-%d')
                if arret_datetime < ajd_datetime:
                    ferm_datetime = arret_datetime
                else:
                    ferm_datetime = ajd_datetime
            else:
                ferm_datetime = ajd_datetime 
        
            delta = ferm_datetime - ouv_datetime
            
            nb_iter = np.trunc(delta.days/Nmax)+1 #partie entière du nombre d'itérations

        else:
            
            ##Selon la plage temporelle spécifiée
            
            Nmax = 10000 #profondeur max de la requête
            
            ouv_string = tRange[0]
            ouv_datetime = datetime.datetime.strptime(ouv_string, '%Y-%m-%d')
            
            ferm_string = tRange[1]
            ferm_datetime = datetime.datetime.strptime(ferm_string, '%Y-%m-%d')
        
            delta = ferm_datetime - ouv_datetime
            
            nb_iter = np.trunc(delta.days/Nmax)+1 #partie entière du nombre d'itérations
        
        #Requêtages et concaténations successives
        
        print("closing_datetime",ferm_datetime)
        print("openning_datetime",ouv_datetime)
        
        for i in range(int(nb_iter)):
            
            save_deb = 0
            fin = 0
            
            incr = datetime.timedelta(days=(i+1)*Nmax)
            
            if ferm_datetime-incr <= ouv_datetime:
                
                tmp = str(ouv_datetime)
                deb = tmp[:10]
                print("deb_obs",deb)
                
                tmp = str(ferm_datetime - datetime.timedelta(days=i*Nmax))
                fin = tmp[:10]
                print("fin_obs",fin)
                
                df_observ = requestBackend_observations_hubeau(stas,deb,fin)
                #Add geometry column
                df_observ.loc[:,'geometry'] = geom
                stations_observ.append(df_observ)
                del df_observ
            
            else:
                
                tmp = str(ferm_datetime - datetime.timedelta(days=(i+1)*Nmax-1))
                deb = tmp[:10]
                print("deb_obs",deb)
                
                tmp = str(ferm_datetime - datetime.timedelta(days=i*Nmax))
                fin = tmp[:10]
                print("fin_obs",fin)
                
                df_observ = requestBackend_observations_hubeau(stas,deb,fin)
                #Add geometry column
                df_observ.loc[:,'geometry'] = geom
                stations_observ.append(df_observ)
                del df_observ
        

    #Concatenate every dataframes of all stations and convert to geodataframe
    df_stations = pd.concat(stations_observ)
    gdf_stations = gpd.GeoDataFrame(df_stations, crs="EPSG:4326")
    gdf_stations.set_geometry('geometry')

    #Write to disk 
    gdf_stations.to_file(dstFile)

    return



def compute_MeanMonthlyFlow(stationCode,stationsLayer,dstLayer,period,generate_plot):
    
    """
    stationCode: code of the station to be analyzed [string]
    stationsLayer: source layer that is the geodataframe with stations observations #must be a gdf already loaded [geopandas object]
    where columns must be labelled as follow ['code_station','date_obs_elab','resultat_obs_elab','grandeur_hydro_elab','libelle_statut','geometry']
    period: time window to perform the analysis upon given as a list [start,end] with date format YYYY [list]
    generate_plot: if set to True, the function also creates a plot of MMF along the period [python object]
    
    dstLayer: /path/to/destination/layer.gpkg that is a geodataframe with columns = ['code_station','MMFmu_#month_#period','MMFsigma_#month_#period','geometry']
    """
    
    import re
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import datetime
    
    dfc_hydro = stationsLayer.copy()
    
    #Extraction des relevés hydro pour la station
    dfc_hydro.code_station = dfc_hydro.code_station.astype("string")
    dfc_cut = dfc_hydro[dfc_hydro.code_station.str.match(str(stationCode))]
    dfc_cut.dropna(subset='resultat_obs_elab',axis=0,inplace=True)

    if dfc_cut.empty is not True:
    
        #Grouper par mois et calculer la moyenne
        
        #Convertir la colonne date en objet datetime
        serie = dfc_cut.date_obs_elab.astype('datetime64[ns]')
        df = pd.DataFrame(serie,index=dfc_cut.index)
        dfc_cut = dfc_cut[['code_station','resultat_obs_elab','grandeur_hydro_elab','libelle_statut','geometry']]
        dfm = pd.merge(df, dfc_cut, left_index=True, right_index=True)
        dfm.reset_index(drop=True,inplace=True)
        
        #Vérification de la plage temporelle appelée
        deb = datetime.datetime.strptime(str(period[0]), '%Y')
        fin = datetime.datetime.strptime(str(period[1]), '%Y')  
        
        
        if dfm.date_obs_elab.min() <= deb:
            
            #Restriction du df aux années appelées
            dfm_cut = dfm.loc[(dfm.date_obs_elab >= deb) & (dfm.date_obs_elab <= fin)]
    
            #Réindexation par date
            dfm_cut.set_index('date_obs_elab',inplace=True)
    
            #Grouper par mois avec la moyenne comme agrégateur
            dfm_grouped = dfm_cut.groupby(pd.Grouper(freq="M"))['resultat_obs_elab'].mean()
    
            #Génération des (Yi), du vecteur sigma_Y et du vecteur mu_Y
    
            #Préparation du df pour le regex sur les dates
            df2 = pd.DataFrame(dfm_grouped)
            df2.reset_index(inplace = True)
            df2['date_obs_elab'] = df2['date_obs_elab'].astype("string")
    
            mu_Y = []
            sigma_Y = []
            
            for i in range(12):
    
                #Grouper par mois en créant un df par mois : chaque df correspond à un Yi de la suite (Yi)1<i<12
                if (i+1) <= 9:
                    regex_pattern = r"(\d+)-0{m}-(\d+)".format(m=i+1)
                    p = re.compile(regex_pattern)
                    f = np.vectorize(lambda x: pd.notna(x) and bool(p.search(x)))
                    df_month = df2[f(df2['date_obs_elab'])]
                else:
                    regex_pattern = r"(\d+)-{m}-(\d+)".format(m=i+1)
                    p = re.compile(regex_pattern)
                    f = np.vectorize(lambda x: pd.notna(x) and bool(p.search(x)))
                    df_month = df2[f(df2['date_obs_elab'])]
                
                #Générer les vecteurs de barres d'erreur et de moyennes 
                mu = df_month.resultat_obs_elab.mean()
                sigma = df_month.resultat_obs_elab.std(skipna=True, ddof=1)
                mu_Y.append(mu)
                sigma_Y.append(sigma)
                
                df_month = df_month.iloc[0:0] #del df_month content
          
            #Gather monthly data in a dataframe and save it to disk as .gpkg
           
            dic = {'code_station':[stationCode],
            	f"MMFmu_month1_{period[0]}{period[1]}":[mu_Y[0]],
            	f"MMFmu_month2_{period[0]}{period[1]}":[mu_Y[1]],
            	f"MMFmu_month3_{period[0]}{period[1]}":[mu_Y[2]],
            	f"MMFmu_month4_{period[0]}{period[1]}":[mu_Y[3]],
            	f"MMFmu_month5_{period[0]}{period[1]}":[mu_Y[4]],
            	f"MMFmu_month6_{period[0]}{period[1]}":[mu_Y[5]],
            	f"MMFmu_month7_{period[0]}{period[1]}":[mu_Y[6]],
            	f"MMFmu_month8_{period[0]}{period[1]}":[mu_Y[7]],
            	f"MMFmu_month9_{period[0]}{period[1]}":[mu_Y[8]],
            	f"MMFmu_month10_{period[0]}{period[1]}":[mu_Y[9]],
            	f"MMFmu_month11_{period[0]}{period[1]}":[mu_Y[10]],
            	f"MMFmu_month12_{period[0]}{period[1]}":[mu_Y[11]],
            	f"MMFsigma_month1_{period[0]}{period[1]}":[sigma_Y[0]],
            	f"MMFsigma_month2_{period[0]}{period[1]}":[sigma_Y[1]],
            	f"MMFsigma_month3_{period[0]}{period[1]}":[sigma_Y[2]],
            	f"MMFsigma_month4_{period[0]}{period[1]}":[sigma_Y[3]],
            	f"MMFsigma_month5_{period[0]}{period[1]}":[sigma_Y[4]],
            	f"MMFsigma_month6_{period[0]}{period[1]}":[sigma_Y[5]],
            	f"MMFsigma_month7_{period[0]}{period[1]}":[sigma_Y[6]],
            	f"MMFsigma_month8_{period[0]}{period[1]}":[sigma_Y[7]],
            	f"MMFsigma_month9_{period[0]}{period[1]}":[sigma_Y[8]],
            	f"MMFsigma_month10_{period[0]}{period[1]}":[sigma_Y[9]],
            	f"MMFsigma_month11_{period[0]}{period[1]}":[sigma_Y[10]],
            	f"MMFsigma_month12_{period[0]}{period[1]}":[sigma_Y[11]],
            	'geometry':[dfm.loc[0,'geometry']]
            	}
            	
            dfg = pd.DataFrame(dic)
            final_gdf = gpd.GeoDataFrame(dfg, crs="EPSG:4326")
            final_gdf.set_geometry('geometry',inplace=True)
            final_gdf.to_file(dstLayer)
            
            
            #Draw hydrogram if specified by generate_plot=True
            if generate_plot is True:
                x_axis = [x+1 for x in range(12)]
                #plt.plot(x_axis, mu_Y, 'k') #plot(x,y,format,x,y2,format2,...,x,yN,formatN)
                plt.errorbar(x_axis, mu_Y, yerr = sigma_Y, fmt ='o')
                #plt.axis([1, 12, np.min(mu_Y), 20]) #graph bbox xmin, xmax, ymin, ymax
                plt.xticks(x_axis)
                plt.ylabel('QmM moyen sur la période [l.s-1]',fontsize=9)
                plt.xlabel('mois',fontsize=9)
                plt.title('Hydrogramme mensuel moyen de la station {stas} sur la période {span}'.
                          format(stas=stationCode,span=period),fontsize=9)
                plt.savefig('./debits_mensuels_moyens/dmm_{stas}.png'.format(stas=stationCode),bbox_inches='tight')
                plt.close() #sinon à chaque appel de la fonction la figure est dessinée sur le même graphe
            else:
                pass
            
        else:
            
            print(f"Station {str(stationCode)} does not cover the range {str(period)}")

    else:
        pass

    try:
        final_gdf
    except:
        pass
    else:
        return final_gdf



def request_DEM(bbox,srcEPSG,dstFile):

    """
    Makes a request at https://portal.opentopography.org/apidocs/#/Public/getGlobalDem and returns a SRTM 30m DEM for the specified bounding box
    bbox: bounding box formated as "xmin ymin xmax ymax" [string]. bbox can be extracted from vector with extract_Vectorbbox() or from raster with extract_Rasterbbox() functions 
    srcEPSG: EPSG code of srcFile [int] The function then converts projection to WGS84 to make the request and then reproject the output back to srcEPSG
    dstFile: /path/to/destination/file.tif [string] Path where the request's output is being written to disk
    
    output: DEM downloaded .tif file reprojected to srcESPG
    """

    import pyproj
    import os
    import requests

    #Read coordinates and reproject them to WGS84 (EPSG 4326)
    bbox_list = bbox.split() #xmin ymin xmax ymax
    transformer = pyproj.Transformer.from_crs(f"EPSG:{str(srcEPSG)}", "EPSG:4326")
    xmin, ymin = transformer.transform(bbox_list[0], bbox_list[1])
    xmax, ymax = transformer.transform(bbox_list[2], bbox_list[3])
    

    #Make the request to opentopography API
    url = f"https://portal.opentopography.org/API/globaldem?demtype=SRTMGL1&south={str(ymin)}6&north={str(ymax)}&west={str(xmin)}&east={str(xmax)}&outputFormat=GTiff&API_Key=8aea7de3cd2f16fbdcfcc819216f5eb4"
    print(url)
    r = requests.get(url, allow_redirects=True)
    with open(f"{dstFile[:-4]}_wgs84.tif", "wb") as f:
        f.write(r.content)

    #Reproject tif file to srcEPSG
    src = f"{dstFile[:-4]}_wgs84.tif"
    dst = dstFile
    cmd = f"gdalwarp -overwrite -s_srs EPSG:4326 -t_srs EPSG:{str(srcEPSG)} -r near -of GTiff {src} {dst}"
    os.system(cmd)
    os.remove(f"{dstFile[:-4]}_wgs84.tif")

    return


def split_singleband(srcFile,epsgCode,zRestriction=None):

    """
    Creates a raster .tif scalar file for each pixel value in the single band raster srcFile. Pixel values must be id-like. 
    srcFile: /path/to/source/raster/file.tif
    zRestriction [optional]: list of pixel values to be processed in the srcFile zField [list]
    dstFile follows the geoTransform propreties of srcFile and has path f"{srcFile[:-4]}_PixelValueIs{#}.tif"
    """

    #Import libraries
    
    import numpy as np
    from osgeo import gdal, osr
    gdal.UseExceptions()

    gdalDataTypes = {
                      "uint8": 1,
                      "int8": 1,
                      "uint16": 2,
                      "int16": 3,
                      "uint32": 4,
                      "int32": 5,
                      "float32": 6,
                      "float64": 7,
                      "complex64": 10,
                      "complex128": 11
                    } #see: https://borealperspectives.org/2014/01/16/data-type-mapping-when-using-pythongdal-to-write-numpy-arrays-to-geotiff/

    #Read srcFile and extract info
    
    r = gdal.Open(srcFile)
    band = r.GetRasterBand(1)
    src = band.ReadAsArray().astype(float)
    
    ulx, xres, xskew, uly, yskew, yres = r.GetGeoTransform()
    geoT = r.GetGeoTransform()
    
    h,w = src.shape

    max = band.GetMaximum()

    r = None


    #Iterate over each each pixel value in srcFile and create a separate dstFile for each value

    if zRestriction is None:

        for val in range(int(max)+1):
    
            print(f"Pixel value {str(val)} out of {str(max)}")
        
            #Create an empty array populated with NaN values to further store results
            
            array = np.empty([h,w])
            array[:] = np.float32(np.nan)
        
            #Range over cells and retrieve cells on condition
        
            for y in range(h): 
        
                if y == 0: #yres is negative and y axis goes x-wise
                    ymax = uly 
                    ymin = uly + yres
                else:
                    ymax = uly + y * yres #for last iteration, h = h-1 because of range() behavior
                    ymin = uly + y * yres + yres 
                
                for x in range(w):
        
                    if x == 0:
                        xmin = ulx 
                        xmax = ulx + xres
                    else:
                        xmin = ulx + x * xres 
                        xmax = ulx + x * xres + xres
        
                    if (xmin >= xmax) | (ymin >= ymax):
                        print("Error (x,y)",(x,y))
                    else:
                        pass
        
                    #Set float values
                    current_pixel = np.float32(src[y,x])
                    target_pixel = np.float32(val)
                
        
                    #Retieve pixel on condition
                    if current_pixel == target_pixel:
                        array[y,x] = current_pixel 
                    else:
                        array[y,x] = np.float32(np.nan)
                
            #Export array as dstFile .tif
    
            dstFile = f"{srcFile[:-4]}_PixelValueIs{str(val)}.tif"
            gdalType = gdalDataTypes[array.dtype.name]
            outDs = gdal.GetDriverByName('GTiff').Create(dstFile, w, h, 1, gdalType)
            outBand = outDs.GetRasterBand(1)
            outBand.WriteArray(array)
            outDs.SetGeoTransform(geoT)
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(epsgCode)
            outDs.SetProjection(srs.ExportToWkt())
            outDs = None

    else:

        for val in list(zRestriction):
    
            print(f"Pixel value {str(val)} in {str(zRestriction)}")
        
            #Create an empty array populated with NaN values to further store results
            
            array = np.empty([h,w])
            array[:] = np.float32(np.nan)
        
            #Range over cells and retrieve cells on condition
        
            for y in range(h): 
        
                if y == 0: #yres is negative and y axis goes x-wise
                    ymax = uly 
                    ymin = uly + yres
                else:
                    ymax = uly + y * yres #for last iteration, h = h-1 because of range() behavior
                    ymin = uly + y * yres + yres 
                
                for x in range(w):
        
                    if x == 0:
                        xmin = ulx 
                        xmax = ulx + xres
                    else:
                        xmin = ulx + x * xres 
                        xmax = ulx + x * xres + xres
        
                    if (xmin >= xmax) | (ymin >= ymax):
                        print("Error (x,y)",(x,y))
                    else:
                        pass
        
                    #Set float values
                    current_pixel = np.float32(src[y,x])
                    target_pixel = np.float32(val)
                
        
                    #Retieve pixel on condition
                    if current_pixel == target_pixel:
                        array[y,x] = current_pixel 
                    else:
                        array[y,x] = np.float32(np.nan)
                
            #Export array as dstFile .tif
    
            dstFile = f"{srcFile[:-4]}_PixelValueIs{str(val)}.tif"
            gdalType = gdalDataTypes[array.dtype.name]
            outDs = gdal.GetDriverByName('GTiff').Create(dstFile, w, h, 1, gdalType)
            outBand = outDs.GetRasterBand(1)
            outBand.WriteArray(array)
            outDs.SetGeoTransform(geoT)
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(epsgCode)
            outDs.SetProjection(srs.ExportToWkt())
            outDs = None

    return max



def raster_to_polygons(srcFile,dstFile,epsgCode,zName,zRestriction=None):

    """
    Converts raster to polygons with one polygon for a connected region of same pixel value
    This function makes a shadow call to relio.split_singleband() to first create one unique raster for each pixel value in srcFile
    Then it builds temporary polygon vector files and then it puts them together in a geodataframe written to disk as dstFile
    srcFile: /path/to/source/raster/file.tif which must contain a unique pixel value and other pixels = NoData: see function relio.split_singleband() [string]
    dstFile: /path/to/destination/vector/file.gpkg which contains the polygons associated with srcFile [string]
    zName: name of the z value field to be given as a layer in dstFile [string]
    zRestriction [optional]: pass a list of values to lighten domain and to process only these values in zName [list]
    """

    #Import libraries
    
    from osgeo import gdal, osr
    gdal.UseExceptions()
    import geopandas as gpd
    import pandas as pd
    import os

    #Call function relio.split_singleband() to split srcFile raster into multiple rasters with unique pixel value
    max = split_singleband(srcFile,epsgCode,zRestriction) #dstFile = f"{srcFile[:-4]}_PixelValueIs{#}.tif"

    #For each above created raster call gdal_polygonize.py

    vectorList =[]

    if zRestriction is None:
    
        for val in range(1,int(max)+1): #catchment=0 is a "fake" catchment generated by pcraster.subcatchment(), so forget it
    
            rasterFile = f"{srcFile[:-4]}_PixelValueIs{str(val)}.tif"
            tmpVectorFile = f"{srcFile[:-4]}_PixelValueIs{str(val)}.gpkg"
            cmd = 'gdal_polygonize.py {r} -overwrite -b 1 -f "GPKG" {v} OUTPUT {z}'.format(r=rasterFile,v=tmpVectorFile,z=zName)
            os.system(cmd)
            #The above gdal command returns a vector file with multiple entities: one polygon for the envelop, and multiple polygons where region is broken
            #So we remove the envelop (catch_id=0) polygon mnd merge the others 
            tmpVectorLoad = gpd.read_file(tmpVectorFile)
            tmpVectorLoad[f"{zName}"] = tmpVectorLoad[f"{zName}"].astype("string")
            m = tmpVectorLoad[f"{zName}"] == str(val)
            tmpVectorLoadCleared = tmpVectorLoad.loc[m] #returns a gdf without catch_id=0
            union = tmpVectorLoadCleared.unary_union #returns a shapely.geometry
            frames = {'catch_id':[val],
              'geometry' : [union]}
            uniongdf = gpd.GeoDataFrame(frames, crs=f"EPSG:{str(epsgCode)}")
            uniongdf = uniongdf.set_geometry('geometry')
            if os.path.exists(dstFile):
                os.remove(tmpVectorFile)
                uniongdf.to_file(tmpVectorFile)
            else:
                uniongdf.to_file(tmpVectorFile)
            vectorList.append(uniongdf)
            del rasterFile, tmpVectorFile, tmpVectorLoad, tmpVectorLoadCleared, union, uniongdf
    
        #Concatenate each above creater vector layers into a single geodatframe
        df = pd.concat(vectorList)
        gdf = gpd.GeoDataFrame(df, crs=f"EPSG:{str(epsgCode)}")
        gdf = gdf.set_geometry('geometry')
    
        #Write geodataframe to disk
        if os.path.exists(dstFile):
            os.remove(dstFile)
            gdf.to_file(dstFile)
        else:
            gdf.to_file(dstFile)
    

    else:

        for val in list(zRestriction): #catchment=0 is a "fake" catchment generated by pcraster.subcatchment(), so forget it
    
            rasterFile = f"{srcFile[:-4]}_PixelValueIs{str(val)}.tif"
            tmpVectorFile = f"{srcFile[:-4]}_PixelValueIs{str(val)}.gpkg"
            cmd = 'gdal_polygonize.py {r} -overwrite -b 1 -f "GPKG" {v} OUTPUT {z}'.format(r=rasterFile,v=tmpVectorFile,z=zName)
            os.system(cmd)
            #The above gdal command returns a vector file with multiple entities: one polygon for the envelop, and multiple polygons where region is broken
            #So we remove the envelop (catch_id=0) polygon mnd merge the others 
            tmpVectorLoad = gpd.read_file(tmpVectorFile)
            tmpVectorLoad[f"{zName}"] = tmpVectorLoad[f"{zName}"].astype("string")
            m = tmpVectorLoad[f"{zName}"] == str(val)
            tmpVectorLoadCleared = tmpVectorLoad.loc[m] #returns a gdf without catch_id=0
            union = tmpVectorLoadCleared.unary_union #returns a shapely.geometry
            frames = {'catch_id':[val],
              'geometry' : [union]}
            uniongdf = gpd.GeoDataFrame(frames, crs=f"EPSG:{str(epsgCode)}")
            uniongdf = uniongdf.set_geometry('geometry')
            if os.path.exists(dstFile):
                os.remove(tmpVectorFile)
                uniongdf.to_file(tmpVectorFile)
            else:
                uniongdf.to_file(tmpVectorFile)
            vectorList.append(uniongdf)
            del rasterFile, tmpVectorFile, tmpVectorLoad, tmpVectorLoadCleared, union, uniongdf
    
        #Concatenate each above creater vector layers into a single geodatframe
        df = pd.concat(vectorList)
        gdf = gpd.GeoDataFrame(df, crs=f"EPSG:{str(epsgCode)}")
        gdf = gdf.set_geometry('geometry')
    
        #Write geodataframe to disk
        if os.path.exists(dstFile):
            os.remove(dstFile)
            gdf.to_file(dstFile)
        else:
            gdf.to_file(dstFile)
    
   
    return


def convert_to_pcraster(srcFile,dstFile,epsgCode,maskFile):

    """
    Convert continuous .tif raster to a scalar .map raster

    maskFile : path/to/mask/raster/file.tif used to retrieve a geoTransform and set proper bbox to the PCRatser file
    """

    import os
    from osgeo import gdal, gdalconst
    gdal.UseExceptions()

    #Read maskfile to get geotransfrom info
    mask = gdal.Open(maskFile)
    upx, xres, xskew, upy, yskew, yres = mask.GetGeoTransform()
    cols = mask.RasterXSize
    rows = mask.RasterYSize
     
    ulx = upx + 0*xres + 0*xskew
    uly = upy + 0*yskew + 0*yres
     
    llx = upx + 0*xres + rows*xskew
    lly = upy + 0*yskew + rows*yres
     
    lrx = upx + cols*xres + rows*xskew
    lry = upy + cols*yskew + rows*yres

    #Set gdal options
    ot = gdalconst.GDT_Float32
    mtdOptions = "VS_SCALAR"
    
    #GDAL Translate
    cmd = f"gdal_translate -a_srs EPSG:{str(epsgCode)} -a_ullr {str(ulx)} {str(uly)} {str(lrx)} {str(lry)} -ot Float32 -of PCRaster {str(srcFile)} {str(dstFile)}"
    os.system(cmd)
    #Note: still with a proper geoTransform options setting, .tif and .map rasters do not exactly overlap...
    
    #Properly close the datasets to flush to disk
    del mask

    return




def create_flowdirection(srcFile, dstFile, cloneFile, outflowdepth, corevolume, corearea, catchmentprecipitation):

    """
    Remove pits from a DEM raster (prior conversion to PCRaster format is required) through generating a FlowDirection map
    Parameters to remove pits (when the cell under consideration is higher than the 4 values, it is considered a pit and not removed) 
    So to remove all pits set very high value e.g. 1e31
    
    cloneFile: Only the PCRaster file format is supported as input argument
    """ 

    from pcraster import readmap, setclone, lddcreate, report
    import os
    #import importlib
    #importlib.invalidate_caches()

    #Create clone map 
    setclone(cloneFile)

    #Create flow direction map
    dem = readmap(srcFile)
    flowDirection = lddcreate(dem,outflowdepth,corevolume,corearea,catchmentprecipitation)
    report(flowDirection,dstFile)

    #Close variables
    dem = None
    flowDirection = None

    return
    

def create_subcatchments(outletsLayer,flowdirectionRaster,dstRaster,cloneMap):
    
    """
    outletsLayer: /path/to/point/vector/layer.gpkg or any compatible format with geopandas drivers, must be point geometry [string]
    flowdirectionRaster: /path/to/flowdirection/raster.map [string]
    cloneMap: /path/to/pcraster/clone.map [string]
    
    dstRaster: /path/to/destination/raster.map where subcatchments appear on the same raster with one pixel value for each [string]
    """
    
    import geopandas as gpd
    import os
    from pcraster import readmap, subcatchment, report
    wd = os.getcwd()
    #import gc
    #gc.collect()
    
    #Convert .gpkg outletsLayer to .txt file with structure: x_coord y_coord id
    
    gdf = gpd.read_file(outletsLayer)
    coords_wkt = list(gdf['geometry'])
    coords_string = [str(x) for x in coords_wkt]
    coords_export = []
    idx = 0
    for elem in coords_string:
        tmp = elem[6:]
        coords_split = tmp.split(" ")
        xcoord = coords_split[0][1:]
        ycoord = coords_split[1][:-1]
        if idx != len(coords_string)-1:
            coords_export.append(f"{str(xcoord)}"+" "+f"{str(ycoord)}"+" "+f"{str(idx+1)}"+"\n")
        else:
            coords_export.append(f"{str(xcoord)}"+" "+f"{str(ycoord)}"+" "+f"{str(idx+1)}")
        idx += 1
    
    txtFile = f"{wd}/tmp_outlets.txt"
    if os.path.isfile(txtFile) is True:
        os.remove(txtFile)
    else:
        pass
    f = open(txtFile,"w+")
    f.writelines(coords_export)
    f.close()
    
    
    #Convert .txt file to .map PCRaster file
    
    dst = f"{wd}/tmp_outlets_pcraster.map"
    if os.path.isfile(dst) is True:
        os.remove(dst)
    else:
        pass
    cmd = f"col2map --clone {cloneMap} -N {txtFile} {dst}"
    print(cmd)
    os.system(cmd)
    os.remove(txtFile)
    
    #Generate catchments map with FlowDirection map and Outlet map 

    #FlowDirection = readmap(flowdirectionRaster)
    #OutletMap = readmap(dst)
    Catch = subcatchment(flowdirectionRaster,dst)
    report(Catch,f"{dstRaster}")
    
    #Flush memory 
    #os.remove(txtFile)
    #os.remove(dst)
    FlowDirection = None
    OutletMap = None
    Catch = None
    cloneMap = None
    
    
    return


def create_material(maskFile,dstFile,epsgCode,materialValue,initialConditionsArray=None):
    
    """
    Creates a raster .tif scalar file with every cell equals to materialValue and with geo properties equals to maskFile
    Usefull to generate a material map, which together with a flow direction map, are used to derive an accuflux map 

    initialConditionsArray [optional] : numpy.array of shape (nb_rows,2) with each element [initial_accuflux,index].
    It must be indexed on a same-shape raster as mskFile
    """

    import numpy as np
    from osgeo import gdal, osr
    gdal.UseExceptions()
    gdalDataTypes = {
              "uint8": 1,
              "int8": 1,
              "uint16": 2,
              "int16": 3,
              "uint32": 4,
              "int32": 5,
              "float32": 6,
              "float64": 7,
              "complex64": 10,
              "complex128": 11
            } #see: https://borealperspectives.org/2014/01/16/data-type-mapping-when-using-pythongdal-to-write-numpy-arrays-to-geotiff/

    #Read maskfile to get geotransfrom info
    
    r = gdal.Open(maskFile)
    band = r.GetRasterBand(1)
    a = band.ReadAsArray().astype(float)
    ulx, xres, xskew, uly, yskew, yres = r.GetGeoTransform()
    nrows, ncols = a.shape
    geoT = r.GetGeoTransform()

    #Generate numpy array with value = materialValue and properties = maskFile
    
    materialArray = np.full(a.shape,np.float32(materialValue))

    #Apply optional initialConitionsArray

    if initialConditionsArray is None:

        pass

    else:

        conditionsArray = np.array(initialConditionsArray)
        
        #Generate the correspondance matrix between pixel index in raster maskFile and array
        
        index_matrix = np.empty([nrows,ncols])
        index_matrix[:] = np.float32(np.nan)
        index = 0
    
        for y in range(nrows): 
            if y == 0: #yres is negative and y axis goes x-wise
                ymax = uly 
                ymin = uly + yres
            else:
                ymax = uly + y * yres #for last iteration, h = h-1 because of range() behavior
                ymin = uly + y * yres + yres 
            for x in range(ncols):
                if x == 0:
                    xmin = ulx 
                    xmax = ulx + xres
                else:
                    xmin = ulx + x * xres 
                    xmax = ulx + x * xres + xres
                if (xmin >= xmax) | (ymin >= ymax):
                    print("Error (x,y)",(x,y))
                else:
                    pass
                
                index_matrix[y,x] = np.int32(index)
                index += 1

        #Modify materialArray with initialConditionsArray based on index_matrix

        index = 0
        h,w = conditionsArray.shape

        #Range over elements in materialArray
        for y in range(nrows):
            
            for x in range(ncols):

                #Find matching index in initialConditionsArray if any and change corresponding materialValue accordingly
                for row in range(h):
                    
                    i = conditionsArray[row,1]
                    
                    if np.int32(i) == np.int32(index):

                        initialValue = np.float32(conditionsArray[row,0])
                        materialArray[y,x] = initialValue

                    else:
                        pass
                        
                index += 1

    #Export array as dstFile .tif

    gdalType = gdalDataTypes[materialArray.dtype.name]
    outDs = gdal.GetDriverByName('GTiff').Create(dstFile, ncols, nrows, 1, gdalType)
    outBand = outDs.GetRasterBand(1)
    outBand.WriteArray(materialArray)
    outDs.SetGeoTransform(geoT)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsgCode)
    outDs.SetProjection(srs.ExportToWkt())
    outDs = None

    return



def create_accuflux(flowDirection, material, dstFile, cloneFile):

    """
    Creates accuflux .map based on flowDirection .map and material .map

    cloneFile: Only the PCRaster file format is supported as input argument
    """ 

    from pcraster import setclone, readmap, accuflux, report
    #import gc
    #gc.collect()
    #import importlib
    #importlib.invalidate_caches()

    #Create clone map 
    setclone(cloneFile)

    #Create accuflux map
    flowdir = readmap(flowDirection)
    mat = readmap(material)
    acc = accuflux(flowdir,mat)
    report(acc,dstFile)

    #Close variables
    del flowdir
    del mat
    del acc

    return


def extract_cellsValues(srcFile,*dropna):
    
    """
    srcFile: path/to/source/raster/file.tif [string]
    dropna [optional]: if set to True, the returned dict will not include cells with NaN values [Python object]
    output : python dictionary with keys 'coordinates', 'values', 'id' i.e. 'coordinates':["xmin ymin xmax ymax", ...], 'value':[...], 'id':[(y,x),...] for each cell in srcFile [dict]
    """
    
    from osgeo import gdal
    gdal.UseExceptions()
    import numpy as np
    
    r = gdal.Open(srcFile)
    band = r.GetRasterBand(1)
    a = band.ReadAsArray().astype(float)
    
    ulx, xres, xskew, uly, yskew, yres = r.GetGeoTransform()
    
    h,w = a.shape
    
    coords = []

    #Create empty dictionary to further store results
    
    frames = {'coordinates':[],
              'values':[],
             'id':[]}
    
    for y in range(h): 

        if y == 0: #yres is negative and y axis goes x-wise
            ymax = uly 
            ymin = uly + yres
        else:
            ymax = uly + y * yres #for last iteration, h = h-1 because of range() behavior
            ymin = uly + y * yres + yres 
        
        for x in range(w):

            if x == 0:
                xmin = ulx 
                xmax = ulx + xres
            else:
                xmin = ulx + x * xres 
                xmax = ulx + x * xres + xres

            if (xmin >= xmax) | (ymin >= ymax):
                print("Error (x,y)",(x,y))
            else:
                pass

            if dropna is True:
            
                if np.float32(a[y,x]) == np.float32(np.nan):
                    pass
                else:
                    frames['coordinates'].append(f"{xmin} {ymin} {xmax} {ymax}")
                    frames['values'].append(np.float32(a[y,x])) #numpy default float format is float64 which is not compatible with pandas.df float32 format (for further uses)
                    frames['id'].append((y,x))
            
            else:
                frames['coordinates'].append(f"{xmin} {ymin} {xmax} {ymax}")
                frames['values'].append(np.float32(a[y,x]))
                frames['id'].append((y,x))
    
    return frames


def cells_to_points(dictCells,epsgCode,*dstFile):

    """
    dictCells: python dictionary containing keys 'coordinates' and 'values' i.e. 'coordinates':["xmin ymin xmax ymax", ...], 'value':[...] [dict]
    dstFile [optional]: /path/to/vector/layer.gpkg [string]

    output: geodataframe with cells turned to points through centroid [geopandas.DataFrame]
    """

    #Import libraries
    
    import pandas as pd
    from shapely.geometry import Point
    import geopandas as gpd

    #Convert dictCells to dataframe

    df = pd.DataFrame(dictCells)
    df['coordinates'] = df['coordinates'].astype('string')
    listCoordinates = list(df['coordinates'])
    listxmin = []
    listxmax = []
    listymin = []
    listymax = []
    for elem in listCoordinates:
        tmp = elem.split(" ")
        listxmin.append(tmp[0])
        listymin.append(tmp[1])
        listxmax.append(tmp[2])
        listymax.append(tmp[3])
    df['xmin'] = listxmin
    df['ymin'] = listymin
    df['xmax'] = listxmax
    df['ymax'] = listymax
    
    #Create columns for center x coord and center y coord"
    
    df['xmin'] = df['xmin'].astype('float')
    df['xmax'] = df['xmax'].astype('float')
    df['ymin'] = df['ymin'].astype('float')
    df['ymax'] = df['ymax'].astype('float')
    df['xcenter'] = (df['xmin'] + df['xmax']) / 2
    df['ycenter'] = (df['ymin'] + df['ymax']) / 2
    
    #Drop useless colmumns
    
    fields = ['values','xcenter','ycenter']
    dfc = df[fields]

    df = None

    #Reset index
    
    dfc = dfc.reset_index()

    #Convert dataframe to geodataframe
    
    geometry = [Point(xy) for xy in zip(dfc['xcenter'], dfc['ycenter'])]
    tmp = dfc.drop(['xcenter', 'ycenter','index'], axis=1)
    gdf = gpd.GeoDataFrame(tmp, crs=f"EPSG:{str(epsgCode)}", geometry=geometry)

    tmp = None
    dfc = None
    
    #Optionnaly export to vector layer (format GPKG)

    if len(dstFile) != 0:
        gdf.to_file(dstFile)
    else:
        pass

    
    return gdf



def join_points_to_pixels(pointsFile,pointsLayer,rasterFile,epsgCode,dstFile):
    
    """
    Originally made to extract resampled discharge values based on their nearest position to gauge stations.
    Returns a geodataframe where geometry is the one of the closest-to-stations pixels and attributes are from both source files
    Makes a shadow call to functions extract_cellsValues() and cells_to_points()
    
    pointsFile: /path/to/vector/points/file.gpkg [string]
    pointsLayer: name of the column in pointsFile that will be retrieved in output [string]
    rasterFile: /path/to/raster/file.tif [string]
    epsgCode: EPSG Code of both pointsFile and rasterFile [int]
    
    dstFile: /path/to/destination/file.gpkg that is the point geometry geodataframe with merged attributes and coordinates taken from rasterFile's pixels [string]       
    """
    
    import geopandas as gpd
    from shapely import wkt
    
    #Convert rasterFile to geoDataFrame and create a 'uid' column
    raster = extract_cellsValues(rasterFile)
    gdf_raster = cells_to_points(raster,epsgCode)
    uids  = [i for i in range(len(gdf_raster))]
    gdf_raster['uid'] = uids
    
    #Exclude NaN values from gdf_raster
    gdf_raster_noNaN = gdf_raster.dropna(axis=0,subset='values')
    
    #Open pointsFile and cut dataframe to only the following columns ['pointsLayer','geometry','code_station'] 
    points = gpd.read_file(pointsFile)
    gdf_points = points[[f"{str(pointsLayer)}",'geometry','code_station']]
    
    #Spatial joining and merging
    join = gpd.sjoin_nearest(gdf_points,gdf_raster_noNaN,how='left',lsuffix='points', rsuffix='raster')
    
    #Retrieve the coordinates of cooresponding pixels in rasterFile
    gdf_raster_geom = gdf_raster[['geometry','uid']]
    merge = join.merge(gdf_raster_geom,left_on='uid',right_on='uid',how='left',suffixes=('_points', '_raster'))
    
    #Convert above dataframe to geodataframe based on raster geometry
    merge['station_coordinates'] = merge['geometry_points'] 
    merge['station_coordinates'] = merge['station_coordinates'].astype("string")
    merge.drop(['geometry_points'], axis=1, inplace=True)
    merge['geometry'] = merge['geometry_raster']
    merge.drop('geometry_raster', axis=1, inplace=True)
    gdf = gpd.GeoDataFrame(merge, crs=f"EPSG:{str(epsgCode)}", geometry='geometry')
    
    #Write gdf to disk
    gdf.to_file(dstFile)  
    
    return gdf


def convert_to_geotiff(srcFile,dstFile,epsgCode):

    """
    Convert scalar .map raster to a contiuous .tif raster
    """

    from osgeo import gdal
    gdal.UseExceptions()
    import os

    #GDAL Translate
    cmd = f"gdal_translate -a_srs EPSG:{str(epsgCode)} -of GTiff {str(srcFile)} {str(dstFile)}"
    os.system(cmd)

    return