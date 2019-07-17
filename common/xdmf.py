import h5py
import os
import numpy as np

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
          <DataItem Dimensions="{num_elem} 3" NumberType="UInt" Format="HDF">{xyz_filename}:/Mesh/0/mesh/topology</DataItem>
        </Topology>
        <Geometry GeometryType="XYZ">
          <DataItem Dimensions="{num_vert} 3" Format="HDF">{xyz_filename}:/Mesh/0/mesh/geometry</DataItem>
        </Geometry>"""

mesh_ref = """
        <xi:include xpointer="xpointer(//Grid[@Name=&quot;{name}&quot;]/Grid[1]/*[self::Topology or self::Geometry])" />"""

timestamp = """
        <Time Value="{tstep}" />"""

attrib_scalar = """
        <Attribute Name="{field}" AttributeType="Scalar" Center="Node">
          <DataItem Dimensions="{num_vert} 1" Format="HDF">{field_filename}:/VisualisationVector/{tstep}</DataItem>
        </Attribute>"""

attrib_vector = """
      <Attribute Name="{field}" AttributeType="Vector" Center="Node">
        <DataItem Dimensions="{num_vert} 3" Format="HDF">{field_filename}:/VisualisationVector/{tstep}</DataItem>
      </Attribute>"""

grid_end = """
      </Grid>
      <Grid>"""

footer = """
      </Grid>
    </Grid>
  </Domain>
</Xdmf>"""


def write_xdmf(xdmf_filename, fields, tsteps):
    is_1st = True

    with h5py.File(fields["xyz"], "r") as h5f:
        mesh = h5f["Mesh/mesh"]
        geometry = mesh["geometry"]
        topology = mesh["topology"]
        num_vert, dim_vert = np.shape(geometry)
        num_elem, dim_elem = np.shape(topology)

    is_vector = dict()
    for field, filename in fields.items():
        with h5py.File(filename, "r") as h5f:
            data_vec = h5f["VisualisationVector"]
            id_1st = sorted([
                int(a) for
                a in list(data_vec.keys())])[0]
            num_vec_loc, dim = np.shape(data_vec[str(id_1st)])
        is_vector[field] = dim == 3

    keys = dict(
        name="test",
        num_elem=num_elem,
        num_vert=num_vert
    )
        
    text = header.format(**keys)
    for tstep in tsteps:
        if is_1st:
            text += grid_1st_begin.format(**keys)
            text += mesh_1st.format(xyz_filename=fields["xyz"], **keys)
            is_1st = False
        else:
            text += grid_ref_begin.format(**keys)
            text += mesh_ref.format(**keys)
        text += timestamp.format(tstep=tstep)
        for field, filename in fields.items():
            if not is_vector[field]:
                text += attrib_scalar.format(field=field,
                                             field_filename=filename,
                                             tstep=tstep,
                                             **keys)
            else:
                text += attrib_vector.format(field=field,
                                             field_filename=filename,
                                             tstep=tstep,
                                             **keys)
        text += grid_end
    text += footer
    return text


if __name__ == "__main__":
    folder = "../"
    fields = dict(
        psi=os.path.join(folder, "psi.h5"),
        nu=os.path.join(folder, "nu.h5"),
        nuhat=os.path.join(folder, "nuhat.h5"),
        xyz=os.path.join(folder, "xyz.h5")
    )
    tsteps = range(1)
    fname = "test.xdmf"

    text = write_xdmf(fname, fields, tsteps)
    print(text)
