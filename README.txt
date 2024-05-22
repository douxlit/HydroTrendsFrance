(c) Quentin DASSIBAT <q.dassibat@emse.fr>
-
Ecole des Mines de Saint-Etienne (EMSE)
Ecole Nationale des Travaux Publics de l'Etat (ENTPE)
Ecole Urbaine de Lyon (EUL) 

################
# Introduction #
################

This repository aims to ease the treatment of hydrographic data available on hubeau.fr API. The data consist of mean daily flows captured by a network of gauging stations operating throughout France, known as the Vigicrue network.

It is structured in several modules:

# MODULE 0
- for each station within the user-specified Area of Interest (AoI), it downloads the location, code, openning and closing date of each station
- for each station, it dowloads the mean daily flows over a user-specified Period of Interest
- it downloads the Digital Elevation Model (DEM) of the AoI
- API for these operations are : hubeau.eaufrance.fr and portal.opentopography.org

# MODULE 1
- creates the subcatchment corresponding to each station within the AoI
- associates to each subcatchment the corresponding station code for further uses

# MODULE 2
- needs modules 1 and 2 to be completed
- computes elaborated data (mean monthly flows and mean monthly flow deviations)
- exports them as 3 files : a point-geometry .gpkg vector file, a polygon-geometry .gpkg vector file (i.e. subcatchments), a .csv file
- optionally generates scatter plots of elaborated data for each station

###########
# Content #
###########

# main.py
- Python3 file containing the main script
- user is required to set the parameters contained in the "set parameters" section
- user can customize the "set directories" section but can leave it to default

# hydrofunc.py
- Python3 file containing the core functions that runs main.py 

# HydroTrendsFrance_documentation.pdf (in progress)
- documentation explaining each executable MODULE in main.py 

############################
# Environment requirements #
############################

The scripts have be developped for Python3 (with Python 3.10.12)

The scripts need the following additional libraries (with their version used for development): 
- geopandas (0.3.12)
- pandas (2.1.1)
- json (0.17.0)
- shapely (2.0.1)
- requests (2.31.0)
- numpy (1.26.0)
- pcraster (4.4.1)

These libraries can be easilly installed using "pip install lib_name" or "conda install lib_name".
Except PCRaster, which needs to be installed following the doc: https://pcraster.geo.uu.nl/pcraster/4.4.0/documentation/pcraster_project/install.html 