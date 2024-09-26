
import pathlib
import requests
import json

"""
url de l'api : https://api.gouv.fr/documentation/api_hubeau_prelevements 
"""

#%% FUNCTIONS

def extract_Rasterbbox_list(srcFile) -> list:

    """
    srcFile: path/to/source/raster/file.tif [string]

    output : bbox of srcFile as a list [xmin, ymin, xmax, ymax] [list]
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

    bbox = [llx, lly, urx, ury] #xmin (llx) ymin (lly) xmax (urx) ymax (ury)

    return bbox

def extract_Vectorbbox_list(srcFile) -> list:

    """
    srcFile: path/to/source/vector/file.gpkg [string] or any fiona-driver compatible format

    output : bbox of srcFile as a list [xmin, ymin, xmax, ymax] [list]
    """

    import geopandas as gpd

    gdf = gpd.read_file(srcFile)
    bbox = [gdf.total_bounds[0], gdf.total_bounds[1], gdf.total_bounds[2], gdf.total_bounds[3]]

    del gdf

    return bbox

def get_code_ouvrage(obj):
    return obj['properties']['code_ouvrage']

def binary_search(chroniques, x) -> dict:
    low = 0
    high = len(chroniques) - 1
    mid = 0
    D = {}
    while low <= high:
 
        mid = (high + low) // 2
 
        # If x is greater, ignore left half
        if chroniques[mid]['properties']['code_ouvrage'] < x:
            low = mid + 1
 
        # If x is smaller, ignore right half
        elif chroniques[mid]['properties']['code_ouvrage'] > x:
            high = mid - 1

        # means x is present at mid
        else:
            minimum = mid
            maximum = mid
            D['code_usage'] = [chroniques[mid]['properties']['code_usage']]

            while minimum-1 >= 0 and chroniques[minimum-1]['properties']['code_ouvrage'] == x :
                minimum -= 1
            while maximum+1 <= len(chroniques)-1 and chroniques[maximum+1]['properties']['code_ouvrage'] == x :
                maximum += 1
            else :
                for n in range(minimum, maximum+1) :
                    D[f"{chroniques[n]['properties']['annee']}"] = chroniques[n]['properties']['volume']
                    if chroniques[n]['properties']['code_usage'] not in D['code_usage'] :
                        D['code_usage'].append(chroniques[n]['properties']['code_usage'])
                return D
 
    # If we reach here, then the element was not present
    return {}
    
def ouvrages(milieu : list, bbox = None, depart = None, format = "geojson", size = 20000) -> json :

    """
    Request to "https://hubeau.eaufrance.fr/api/v1/prelevements/referentiel/ouvrages", to retrieve all installations interfering with natural water flows, within the AoI
    Return a geojson that can be directly used in QGIS.

    milieu : the water type studied, can be 'CONT' for surface water, 'SOUT' for underground water, 'LIT' for sea or oceanic water.
    bbox : spatial extent of the Area of Interest = [xmin, ymin, xmax, ymax]
    depart : department of interest, only if bbox = None. e.g. ['69', '38', '42', '01']
    format : format of created file. "geojson" by default. Can be "json" or "geojson".
    size : number of data retrieved. max size = 20000.
    """

    link = f"https://hubeau.eaufrance.fr/api/v1/prelevements/referentiel/ouvrages"

    params = {
        'code_departement': depart,
        'code_type_milieu' : milieu,
        'format': format,
        'size': size
    
    }
    if type(bbox) is list :
        params['bbox'] = bbox

    if type(depart) is list :
        params['code_departement'] = depart

    try :
        if bbox == None and depart == None :
            if input("No location (bbox = None, depart = None) was precised.\nContinue ? (y/n) -->") != "y" :
                raise Exception("Code was aborted by user.")

        ouvrages_request = requests.get(link, params=params)
        if ouvrages_request.status_code == 200 or ouvrages_request.status_code == 206 :
            ouvrages_json = json.loads(ouvrages_request.text)
            if len(ouvrages_json['features']) == 0 :
                raise Exception(f"In request to hubeau.eaufrance.fr/api/v1/prelevements/referentiel/ouvrages :\nNo installation was found. Please Check Inputs.")
            else :
                return ouvrages_json
        else :
            raise Exception(f"In request to hubeau.eaufrance.fr/api/v1/prelevements/referentiel/ouvrages :\nstatus code = {ouvrages_request.status_code}")
    except Exception as e :
        raise e

def chroniques(annees : list, bbox = None, depart = None, ouv_list = None, format = "geojson", size = 20000) -> json :

    """
    Request to "https://hubeau.eaufrance.fr/api/v1/prelevements/chroniques", to retrieve the volumes (per year and per installation) of surface water used by humans within the AoI
    Return a geojson that can be directly used in QGIS.

    annees : years of interest. e.g. [2012, 2013, 2014, 2020]
    bbox : spatial extent of the Area of Interest = [xmin, ymin, xmax, ymax]
    depart : department of interest, if bbox = None. e.g. ['69', '38', '42', '01']
    ouv_list : to request on specific installations. list of installation id with max size of 200.
    format : format of created file. "geojson" by default. Can be "json" or "geojson".
    size : number of data retrieved. max size = 20000.
    """

    link = f"https://hubeau.eaufrance.fr/api/v1/prelevements/chroniques"
    fields = ['code_ouvrage', 'nom_ouvrage','latitude', 'longitude', 'code_type_milieu', 'code_usage', 'nom_commune', 'code_commune_insee', 'annee', 'volume']
    params = {
        'annee' : annees,
        'format': format,
        'size': size,
        'code_ouvrage': ouv_list,
        'fields': fields
    }

    if type(bbox) is list :
        params['bbox'] = bbox

    if type(depart) is list :
        params['code_departement'] = depart

    try :
        if bbox == None and depart == None :
            if input("No location (bbox = None, depart = None) was precised.\nContinue ? (y/n) -->") != "y" :
                raise Exception("Code was aborted by user.")
        chroniques_request = requests.get(link, params=params)
        # print(chroniques_request.url)
        if chroniques_request.status_code == 200 or chroniques_request.status_code == 206 :
            chroniques_json = json.loads(chroniques_request.text)
            if len(chroniques_json['features']) == 0 :
                raise Exception(f"In request to {link} :\nNo installation was found. Please Check Inputs.")
            else :
                return chroniques_json
        else :
            raise Exception(f"In request to {link} :\nstatus code = {chroniques_request.status_code}")
    except Exception as e :
        raise e

def extract_ouvrages(features : list, vlim = 200) -> list :
    
    """
    Extract a list of id for the installations describe in the 'features' list extracted from a request to https://hubeau.eaufrance.fr/api/v1/prelevements/
    Return a list with lists of installation id of size = vlim.

    features : 'features' attribute from dict coming from request to api
    vlim : size of the lists within the final list. by default = max size = 200
    """

    ouv_final = []
    ouv_extracted = []
    
    for ouv in features :
        code_ouv = get_code_ouvrage(ouv)
        n = len(ouv_extracted)
        if ouv == features[-1] :
            ouv_extracted.append(code_ouv)
            ouv_final.append(ouv_extracted)
            return ouv_final
        elif n < vlim :
            ouv_extracted.append(code_ouv)
        else :
            ouv_final.append(ouv_extracted)
            ouv_extracted = [code_ouv]

def multi_chroniques(ouv_multi : list, annees : list, bbox = None, depart = None, format = "geojson") -> json :
    
    """
    See function "chroniques" help for more information.
    Make sure that all data are retrieved using multiple requests, when the size of the response is > 20000.
    Return a geojson that can be directly used in QGIS.

    ouv_multi : list of lists of installations id. ouv_multi = [ouv_list1, ouv_list2, etc.] with ouv_list of max size = 200.
    annees : years of interest. e.g. [2012, 2013, 2014, 2020]
    bbox : spatial extent of the Area of Interest = [xmin, ymin, xmax, ymax]
    depart : department of interest, only if bbox = None. e.g. ['69', '38', '42', '01']
    format : format of created file. "geojson" by default. Can be "json" or "geojson".
    """

    features = []
    n=0
    try :
        for ouv_list in ouv_multi :
            if len(ouv_list) > 200 :
                raise Exception(f"In multi_chroniques function : list number {n} in ouv_multi is out of range (length={len(ouv_list)}). List length should be <= 200.")
            C = chroniques(annees, bbox=bbox, depart=depart, ouv_list=ouv_list, format=format)
            features += C['features']
            n+=1
        C['features'] = features
    except Exception as e :
        raise(e)

    return C

def chroniques_ouvrages(path : str, Chro : json, Ouv : json, annees : list) -> json:
    
    """
    Coupling chroniques response and ouvrages response to have a geojson compliant with further analysis in QGIS with volumes per year associated for every installation.
    Download a geojson that can be directly used in QGIS (at the given path).

    path : path to the file to write, or overwrite, with .geojson extension
    Chro : a json which has the same structure than a response from "https://hubeau.eaufrance.fr/api/v1/prelevements/chroniques"
    Ouv : a json which has the same structure than a response from "https://hubeau.eaufrance.fr/api/v1/prelevements/referentiel/ouvrages"
    annees : years of interest, which will appear in the geojson result. e.g. [2012, 2013, 2014, 2020]
    """
    try :
        if path[-8:] != '.geojson' :
            raise Exception("In chroniques_ouvrages function : Path extension must be .geojson !")

        C_f = sorted(Chro['features'], key=get_code_ouvrage)

        for o in Ouv['features'] :

            AnnualMMF = 0
            nb_a = 0
            code_o = get_code_ouvrage(o)
            D = binary_search(C_f, code_o)

            if "code_usage" in D :
                o['properties']['code_usage'] = D['code_usage'][0]

            for a in annees :
                if f"{a}" in D :
                    nb_a += 1
                    o['properties'][f"{a} AnnualFlow"] = D[f"{a}"]
                    AnnualMMF += D[f"{a}"]

                else :
                    o['properties'][f"{a} AnnualFlow"] = None
            if nb_a != 0 :
                #Mean over the years
                AnnualMMF = AnnualMMF/nb_a

                #Monthly mean
                AnnualMMF = AnnualMMF/12

                #m3/month to L/s
                AnnualMMF = AnnualMMF/2629.8

                o['properties']['AnnualMMF'] = AnnualMMF

        with open(path, 'w') as f :
            json.dump(Ouv, f)
        
        return Ouv

    except Exception as e :
        raise(e)

def count_points(polygons_path : str, points_path : str, output_path : str, join_field : str, name_field : str) :

    """
    Counting points in polygons among a field.
    Save the gpkg file in output_path.
    """
    import geopandas as gpd
    import os

    polygons = gpd.read_file(polygons_path)
    points = gpd.read_file(points_path)

    # Ensure the coordinate reference systems match
    points = points.to_crs(polygons.crs)

    # Perform spatial join to count points within each polygon
    joined = gpd.sjoin(points, polygons, how='inner', op='within')

    # Group by polygon and count points
    point_counts = joined.groupby('index_right')[join_field].sum()

    # Merge point counts back to polygons
    polygons[name_field] = polygons.index.map(point_counts).fillna(0)

    polygons.to_file(output_path, driver='GPKG')

    print(f"New subcatchment layer including withdrawal saved as {output_path}.")

def cut_function(territory_path : str, subcatchments_path : str) :
    
    """
    Clipping a territory with subcatchments.
    Return the corresponding dataframe.
    """
    import geopandas as gpd

    territory = gpd.read_file(territory_path)
    subcatchments = gpd.read_file(subcatchments_path)

    subcatchments = subcatchments.to_crs(territory.crs)

    # Perform the clipping operation (intersection)
    result = gpd.overlay(subcatchments, territory, how='intersection')

    return result

#%% MAIN PART
# AoI_filePath = "C:/Users/ITR2276/Documents/EAU_CODE/HYDRO/Emprises/Emprise_Adapte.gpkg" #path to file with extension
# AoI_bbox = extract_Vectorbbox_list(AoI_filePath)
# print(f"AoI_bbox : {AoI_bbox}")

# milieu = ['CONT', 'SOUT']

# annees = [y for y in range(2012, 2022)]

# Ouv = ouvrages(milieu, bbox=AoI_bbox)
# O_f = Ouv['features']

# Ouv_multi = extract_ouvrages(O_f)

# Chro_multi = multi_chroniques(Ouv_multi, annees, bbox = AoI_bbox)

# result_path = "C:/Users/ITR2276/Documents/EAU_CODE/HYDRO/CODE_FINAL/data/BNPE_all.geojson"

# chroniques_ouvrages(result_path, Chro_multi, Ouv, annees)


#%% TESTING PART

####################
# C = chroniques([2015], ouv_list=['OPR0000059870'])
# print(C)

####################

