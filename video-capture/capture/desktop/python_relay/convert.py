import pycolmap
reconstruction = pycolmap.Reconstruction(r"E:\qhack\SplatMesh-Core\test\sparse\0")
reconstruction.export_PLY(r"E:\qhack\SplatMesh-Core\test\sparse\0\model.ply")
print("Done! Open model.ply in Blender.")