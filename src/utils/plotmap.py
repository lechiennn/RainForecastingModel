import gdal
import geopandas as gpd
import os
import matplotlib.pyplot as plt
import numpy as np

BOUNDS = [103.44, 108.04, 17.5, 21.1]

def save(band, max_rain, bounds, output_path, norm=None):
    # Create a matplotlib figure
    plt.figure(figsize=(10, 5))

    # Read the shapefile containing borders
    border = gpd.read_file("src/utils/shapefile/btb.shp")

    image = plt.imshow(band, cmap='jet', norm=norm, extent=bounds)

    # Plot the borders from the shapefile on the same figure
    border.plot(ax=plt.gca(), facecolor='none', edgecolor='black', linewidth=1)

    # Remove x and y ticks
    plt.xticks([])
    plt.yticks([])

    # Create a color bar for the image
    sm = plt.cm.ScalarMappable(cmap='jet')
    sm.set_array([])
    cbar = plt.colorbar(image, extend='both', ticks=[0, 1, 2, 3, 5, 10, 20, 40, 60, 80, 100])
    cbar.mappable.set_clim(0, max_rain)


    # Create the output directory if it doesn't exist
    if not os.path.exists(output_path.split('Radar')[0]):
        os.makedirs(output_path.split('Radar')[0])

    # Save the plot as a PNG image
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    # Close the plot to release memory
    plt.close()

def plot_ref_pred(target_array, predict_array, name, norm=None):
    '''
    predict_array: output của model
    norm: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
    '''
    # # Open the raster dataset using GDAL
    # ref = gdal.Open(file_name)


    # # Get raster metadata
    # geotransform = ref.GetGeoTransform()
    # bounds = (geotransform[0], geotransform[0] + geotransform[1] * ref.RasterXSize,
    #           geotransform[3] + geotransform[5] * ref.RasterYSize, geotransform[3])
    

    # band = ref.GetRasterBand(1).ReadAsArray()
    target_array = target_array.cpu().numpy().squeeze()
    predict_array = predict_array.cpu().numpy().squeeze()   


    max = np.max(target_array) if np.max(target_array) > np.max(predict_array) else np.max(predict_array)

    output_path = name.replace('data/', 'outputs_image/').replace('.tif', '.png')


    save(target_array, max_rain=max, bounds=BOUNDS, output_path=output_path, norm=norm)
    save(predict_array, max_rain=max, bounds=BOUNDS, output_path=output_path.replace('.png', '-predict.png'), norm=norm)
    
def plot_array(predict_array, name='src/utils/test.png',norm=None):

    save(predict_array, max_rain=np.max(predict_array),bounds=BOUNDS, output_path=name, norm=norm)

if __name__ == '__main__':

    # plot_ref_pred('/home/lechiennn/lab/thesis/RainForecastingModel/data/test/2020/09/02/Radar_20200902010000.tif', np.random.rand(90,250), None)
    img = '/home/lechiennn/lab/thesis/RainForecastingModel/data/test/2020/09/02/Radar_20200902010000.tif'
    img = gdal.Open(img).ReadAsArray()
    plot_array(img, name='src/utils/Radar_test.png')
