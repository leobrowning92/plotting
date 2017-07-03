import os
import gwyutils
def save_outputs(c,filebase):
   # save fixed file
   gwy.gwy_file_save(c, filebase+"_fixed.gwy", gwy.RUN_NONINTERACTIVE)

   # export datafield from container c specified by name to png file
   gwyutils.save_dfield_to_png(c, "/0/data", filebase+"_fixed.png", gwy.RUN_NONINTERACTIVE)

directory="/home/leo/drive/Work/COL500_measuredCNTs"
for name in os.listdir(directory):
    fullpath=os.path.join(directory, name)
    c=gwy.gwy_app_file_load(fullpath)
    gwy.gwy_app_data_browser_add(c)
    gwy.gwy_app_data_browser_select_data_field(c, 0)
    print(c)
    # get active container
    c = gwy.gwy_app_data_browser_get_current(gwy.APP_CONTAINER)
    print(c)
    # get filename of active container
    filename = c.get_string_by_name("/filename")
    # remove extension from filename
    filebase = filename[0:-4]
    # set colors of first datafield in active container
    c.set_string_by_name("/0/base/palette", "Gwyddion.net")
    
    
    # call 'polylevel' process module to subtract polynomial
    gwy.gwy_process_func_run("polylevel", c, gwy.RUN_IMMEDIATE)
    #correct for line mismatch using median leveling
    gwy.gwy_process_func_run("align_rows", c, gwy.RUN_IMMEDIATE)
    save_outputs(c,filebase)
