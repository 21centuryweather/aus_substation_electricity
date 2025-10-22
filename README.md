Australian Substation Electricity Data
--------------------------------------

Project to analyse NEAR substation-leve electricity data

Contributions
-------------

- Pia Vassallo: Data analysis, project management
- Ailie Gallant: Supervision, experiment design
- Mathew Lipson: Original processing code, quality checks and basic gap-filling, planning discussions

Description
-----------

This repository holds code to process Australian standardised substation electricity demand data, as released through the CSIRO (now ended) National Energy Analytics Research Program (NEAR). The data extends aproximately from 2004-2018 at half hour intervals, for hundreds of substations across Australia (approximately covering a suburb level).

Website: https://near.csiro.au
To find for demand data by network zone, search on that site with the keyword 'Standardised substation data'.

Processing code example in `import_substation.py`

This example works with some weather data from Sydney. Either remove that code or extract additional data for other locations. 

Instructions
------------

1. Extract this repository to gadi with:  

 `git clone git@github.com:21centuryweather/aus_substation_electricity.git`

2. Copy link of desired NEAR electricty dataset, download to gadi and extract, e.g.:

`curl https://near.csiro.au//public/assets/5bc8975f-5c81-4d49-82b0-f21a5b11fa36/5bc8975f-5c81-4d49-82b0-f21a5b11fa36.zip > ./data/ausgrid/ausgrid.zip`

`unzip ./data/ausgrid/ausgrid.zip -d ./data/ausgrid`

3. Activate Gadi analysis environment:

`module use /g/data/xp65/public/modules; module load conda/analysis3`

4. Load ipython and run the script:

`ipython`

`run import_substation.py`

5. Variables in memory:

 - **demand**: a dataframe of half hourly substation values
 - **info**: substation characteristics info


Substation zone characteristics 
-------------------------------

As well as the half-hourly demand data, CSIRO NEAR also prepared metadata for each substation area, which includes information such as the 

- approximate area serviced
- land uses
- population
- number of dwellings

The metadata zone charactericts are included in this repository at [./data/DNSP_Zone_Substation_Characteristics.csv](./data/DNSP_Zone_Substation_Characteristics.csv), copied from:
https://near.csiro.au/public/aremi/dataset/dnsp_zs_characteristics.csv

The following metadata description were copied from: https://nationalmap.gov.au


### Dataset Description
Here we present a characterisation of Zone Substations in terms of Australian Bureau of Statistics (ABS) Mesh Block (MB) land use categories, dwellings and persons. Areal interpolation has been used to reaggregate data from the ABS Mesh Block geometry to the Zone Substation geometry, using the intersection of polygons. This method has a number of limitations:
Assumption of distribution: MBs areas are small, but there may be instances where a MB intersects with multiple Zone Substations. Statistics have been calculated using the proportionality of the area of intersection. This may not reflect the physical distribution, such as the location of dwellings and persons.
Accuracy of spatial data: The ABS geometry is generally accurate than the DNSP geometry.
Mesh Block Categories are expressed as percentages - Area given to Category / Zone Substation area.

### Estimation of Persons and Dwellings according to ABS 2016 Mesh Block Counts. Display variables:
- DNSP: Name of the Distribution Network Service Provider
- State/Territory: State or Territory of the Distribution Network Service Provider
- Zone Substation Name: Name of the Zone Substation
- Zone Substation ID: DNSP's identifier for the Zone Substation
- Zone Substation Area (km2): Zone Substation area in square kilometres
- Dwellings: Approximate number of dwellings in the Zone Substation area
- Persons: Approximate population of the Zone Substation area
- Residential: Houses, apartments, duplexes, etc
- Commercial: Businesses, shops, etc, generally zero population count
- Industrial: Businesses, generally zero population count
- Primary Production: > 50% primary production, formerly known as Agricultural
- Education: Schools, universities, etc
- Hospital/Medical: Hospital or medical facilities, aged care
- Transport: Road and rail features
- Parkland: Parkland, nature reserves, public open space, etc
- Water: Water bodies, lakes, etc
- Other: Not easily categorised, high mixed use

### Dataset Custodian
National Energy Analytics Research Program (NEAR), from data provided by Australian distribution network businesses.

### Data Currency and Updates
Spatial data for Zone Substation Boundaries and ABS Mesh Blocks as at 2016.


### Licensing, Terms & Conditions
ABS Data

