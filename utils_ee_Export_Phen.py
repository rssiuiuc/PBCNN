# using the phenology dataset to mask the satellite images and export the data to drive
import ee
import time

#calculate evi2
def addevi(image):
    evi = image.expression(
        '2.5 * float(nir - red)*0.0001 / float(nir*0.0001 + 2.4 * red*0.0001 + 1)',
        {
            'red': image.select('Nadir_Reflectance_Band1'),    # 620-670nm, RED
            'nir': image.select('Nadir_Reflectance_Band2'),    # 841-876nm, NIR
            #blue: image.select('Nadir_Reflectance_Band3')    # 459-479nm, BLUE
        });
    return image.addBands(evi.rename(['EVI2']))

#masking images by the phenology stages
def GetPeriodCol(TimeSeries, phen_dataset):
    # TimeSeries is the complete time series
    # phen_dataset is a two-band image, band 1 is the start of the phenological period, band 2 is the end

    ClassDate = ee.Date('1970-1-01')

    def filterPeriod(CurrentImage):
        # Extract the date from the current image
        Date = ee.Date(CurrentImage.get('system:time_start'))
        
        # Calculate the difference in days between the current date and the reference date
        Date_Differ = Date.difference(ClassDate, 'day')
        
        # Calculate the difference between the current date difference and the start and end of the phenological period
        T1 = ee.Image.constant(Date_Differ).subtract(phen_dataset.select([0])).toFloat()
        T2 = ee.Image.constant(Date_Differ).subtract(phen_dataset.select([1])).toFloat()

        # Create a mask where the current date falls within the phenological period
        PeriodTag = T1.multiply(T2).lte(0)

        # Apply the mask to the current image and return the masked image
        # return CurrentImage.updateMask(PeriodTag).selfMask().set('system:time_start', Date)
        return CurrentImage.updateMask(PeriodTag).set('system:time_start', Date)

    # Map the filter function over the time series and return the result
    return TimeSeries.map(filterPeriod)

#export to drive
def export_to_drive(collection, description, folder, year, phen_start, phen_end, varName):
    task = ee.batch.Export.table.toDrive(
        collection=collection,
        description=description+'_'+year+'_'+phen_start+'_'+phen_end+'_'+varName,
        folder=folder,
        fileFormat='GeoJSON'
    )
    task.start()

def main(Satellite, countyCol, SoyMap, Country, year, varName,reducer='mean'):
    
    #map the variable
    def soybeans_map(img):
        return img.updateMask(SoyMap.gte(0.50))

    Satellite = Satellite.map(soybeans_map).select(varName)

    #select the phenology dataset
    phen_list=['Greenup_1','MidGreenup_1','Maturity_1','Peak_1','Senescence_1','MidGreendown_1','Dormancy_1']
    for i in range(len(phen_list)-1):
        phen_start=phen_list[i]
        phen_end=phen_list[i+1]

        phen_dataset= ee.ImageCollection('MODIS/061/MCD12Q2')\
                        .filter(ee.Filter.date(year+'-1-1',year+'-12-31')).select([phen_start,phen_end]).first()

        #masking by the phenology stages

        filteredSatellite = GetPeriodCol(Satellite,phen_dataset)
        # filteredSatellite = Satellite

        # mean reduce collection to one image
        if reducer=='sum':
            reducedImg = filteredSatellite.sum()
        else:
            reducedImg = filteredSatellite.mean()

        # Convert ImageCollection to FeatureCollection
        fc = reducedImg.reduceRegions(
                collection=countyCol,
                reducer=ee.Reducer.percentile([99, 90, 80, 70, 60, 50, 40, 30, 20, 10, 1]),
                scale=500,
                crs='SR-ORG:6974'
            )

        def drop_geometry(feature):
            return feature.setGeometry(None)

        fc = fc.map(drop_geometry)

        while True:
            try:
                # Export the data
                export_to_drive(
                    collection=fc,
                    description=Country+'_Regions',
                    folder='GEE_exports_'+Country,
                    year=year,
                    phen_start=phen_start,
                    phen_end=phen_end,
                    varName=varName)
            except:
                print('Error, waiting 60 seconds to try again...')
                time.sleep(60)
                continue
            break