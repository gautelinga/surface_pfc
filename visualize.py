import os
import numpy as np
from lxml import etree
from collections import defaultdict

header = """<?xml version="1.0"?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="3.0" xmlns:xi="http://www.w3.org/2001/XInclude">
  <Domain>
    <Grid Name="{name}" GridType="Collection" CollectionType="Temporal">"""

grid_1st_begin = """
      <Grid Name="mesh" GridType="Uniform">"""

grid_ref_begin = """
      <Grid>"""

mesh_1st = """
        <Topology NumberOfElements="{num_elem}" TopologyType="Triangle" NodesPerElement="3">
          <DataItem Dimensions="{num_elem} 3" NumberType="UInt" Format="HDF">{topology_filename}:{topology_loc}</DataItem>
        </Topology>
        <Geometry GeometryType="XYZ">
          <DataItem Dimensions="{num_vert} 3" Format="HDF">{xyz_filename}:{xyz_loc}</DataItem>
        </Geometry>"""

mesh_ref = """
        <xi:include xpointer="xpointer(//Grid[@Name=&quot;{name}&quot;]/Grid[1]/*[self::Topology or self::Geometry])" />"""

timestamp = """
        <Time Value="{time}" />"""

attrib_scalar = """
        <Attribute Name="{field}" AttributeType="Scalar" Center="Node">
          <DataItem Dimensions="{num_vert} 1" Format="HDF">{field_filename}:{field_loc}</DataItem>
        </Attribute>"""

attrib_vector = """
      <Attribute Name="{field}" AttributeType="Vector" Center="Node">
        <DataItem Dimensions="{num_vert} 3" Format="HDF">{field_filename}:{field_loc}</DataItem>
      </Attribute>"""

grid_end = """
      </Grid>"""

footer = """
    </Grid>
  </Domain>
</Xdmf>"""


def etree_to_dict(t):
    d = {t.tag: {} if t.attrib else None}
    children = list(t)
    if children:
        dd = defaultdict(list)
        for dc in map(etree_to_dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {t.tag: {k: v[0] if len(v) == 1 else v
                     for k, v in dd.items()}}
    if t.attrib:
        d[t.tag].update(('@' + k, v)
                        for k, v in t.attrib.items())
    if t.text:
        text = t.text.strip()
        if children or t.attrib:
            if text:
                d[t.tag]['#text'] = text
        else:
            d[t.tag] = text
    return d


def parse_xyz_xdmf(xml_file):
    tree = etree.parse(xml_file)
    root = tree.getroot()
    prop = etree_to_dict(root)

    grid = prop["Xdmf"]["Domain"]["Grid"]
    geometry_address = grid["Geometry"]["DataItem"]["#text"].split(":")
    topology_address = grid["Topology"]["DataItem"]["#text"].split(":")
    xyz_address = grid["Attribute"]["DataItem"]["#text"].split(":")
    topology_shape = tuple(
        [int(a) for a in
         grid["Topology"]["DataItem"]["@Dimensions"].split(" ")])
    geometry_shape = tuple(
        [int(a) for a in
         grid["Geometry"]["DataItem"]["@Dimensions"].split(" ")])
    xyz_shape = tuple(
        [int(a) for a in
         grid["Attribute"]["DataItem"]["@Dimensions"].split(" ")])

    return (geometry_address, topology_address, xyz_address,
            geometry_shape, topology_shape, xyz_shape)


def parse_timeseries_xdmf(xml_file):
    tree = etree.parse(xml_file)
    root = tree.getroot()
    prop = etree_to_dict(root)
    grid = prop["Xdmf"]["Domain"]["Grid"]["Grid"]

    dsets = dict()
    for step in grid:
        name = step["Attribute"]["@Name"]
        if name not in dsets:
            dsets[name] = []
        time = int(step["Time"]["@Value"])
        address = step["Attribute"]["DataItem"]["#text"].split(":")
        field_type = step["Attribute"]["@AttributeType"]
        dsets[name].append((time, address, field_type))

    return dsets


def write_combined_xdmf(fields, xyz_filename):
    is_1st = True

    (geometry_address, topology_address, xyz_address,
     geometry_shape, topology_shape, xyz_shape) = parse_xyz_xdmf(xyz_filename)

    is_vector = dict()
    dsets = dict()
    for field, filename in fields.items():
        dsets.update(parse_timeseries_xdmf(filename))
        is_vector[field] = dsets[field][0][2] == "Vector"

    keys = dict(
        name="Timeseries",
        num_vert=geometry_shape[0],
        num_elem=topology_shape[0],
        xyz_filename=os.path.join("Geometry", xyz_address[0]),
        xyz_loc=xyz_address[1],
        topology_filename=os.path.join("Geometry", topology_address[0]),
        topology_loc=topology_address[1]
    )

    text = header.format(**keys)

    field_names = list(fields.keys())
    times = [a[0] for a in dsets[field_names[0]]]
    for field_name in field_names[1:]:
        times_alt = [a[0] for a in dsets[field_name]]
        assert(np.all(np.array(times) == np.array(times_alt)))

    for i, time in enumerate(times):
        if is_1st:
            text += grid_1st_begin.format(**keys)
            text += mesh_1st.format(**keys)
            is_1st = False
        else:
            text += grid_ref_begin.format(**keys)
            text += mesh_ref.format(**keys)
        text += timestamp.format(time=time, **keys)
        for field in field_names:
            dset = dsets[field][i]
            field_filename, field_loc = dset[1]
            if not is_vector[field]:
                attrib = attrib_scalar
            else:
                attrib = attrib_vector

            text += attrib.format(field=field,
                                  field_filename=os.path.join("Timeseries", field_filename),
                                  field_loc=field_loc,
                                  **keys)

        text += grid_end.format(**keys)
    text += footer
    return text


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test visualization.")
    parser.add_argument("folder", help="Folder")
    args = parser.parse_args()

    folder = args.folder

    tsfolder = os.path.join(folder, "Timeseries")
    geofolder = os.path.join(folder, "Geometry")

    fields = dict(
        psi=os.path.join(tsfolder, "psi_from_tstep_0.xdmf"),
        nu=os.path.join(tsfolder, "nu_from_tstep_0.xdmf"),
        nuhat=os.path.join(tsfolder, "nuhat_from_tstep_0.xdmf")
    )
    xyz_filename = os.path.join(geofolder, "xyz.xdmf")

    text = write_combined_xdmf(fields, xyz_filename)
    with open(os.path.join(folder, "visualize.xdmf"), "w") as outfile:
        outfile.write(text)
