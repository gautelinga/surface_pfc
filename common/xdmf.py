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

timestamp = """
        <Time Value="{timestep}" />"""

attrib_scalar = """
        <Attribute Name="{field}" AttributeType="Scalar" Center="Node">
          <DataItem Dimensions="{num_vert} 1" Format="HDF">{field_filename}:/VisualisationVector/0</DataItem>
        </Attribute>"""

grid_end = """
      </Grid>
      <Grid>"""

mesh_ref = """
        <xi:include xpointer="xpointer(//Grid[@Name=&quot;{name}&quot;]/Grid[1]/*[self::Topology or self::Geometry])" />"""

footer = """
      </Grid>
    </Grid>
  </Domain>
</Xdmf>"""

def write_xdmf(xdmf_filename, fields, xyz, tsteps):
    is_1st = True
    for tstep in tsteps:
        pass
