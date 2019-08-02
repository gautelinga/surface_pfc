import os
from utilities.xdmf_utils import list_xdmf_files, write_combined_xdmf


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test visualization.")
    parser.add_argument("folder", help="Folder")
    args = parser.parse_args()

    folder = args.folder

    tsfolder = os.path.join(folder, "Timeseries")
    geofolder = os.path.join(folder, "Geometry")

    tsfilenames = list_xdmf_files(tsfolder)
    geofilenames = list_xdmf_files(geofolder)
    field_filenames = tsfilenames + geofilenames
    xyz_filename = os.path.join(geofolder, "xyz.xdmf")

    text = write_combined_xdmf(field_filenames, xyz_filename)
    with open(os.path.join(folder, "visualize.xdmf"), "w") as outfile:
        outfile.write(text)
