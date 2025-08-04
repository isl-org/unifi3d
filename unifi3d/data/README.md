# Data

### Data iterators

There is one DataIterator per dataset *folder*, e.g. one for a test dataset with a single mesh, one for ShapeNet, one for the preprocessed version of ShapeNet, etc. The data iterators are responsible for returning the **paths** to all object files and preprocessed data files. The data iterators implement the `__getitem__` method, so they can be indexed. Each item of the iterator is a dictionary of paths and metadata. 

Compulsory dict items:
* `id`: Object id
* `text`: text descriptor
* `file_path`: path to the original `.obj` file (before any preprocessing was done)

Optional dict items (e.g. only returned by iterator of preprocessed dataset):
* `base_dir`: Directory of all data for this preprocessed object
* `mesh_path`: Path to preprocessed mesh (as npz file)
* `pointcloud_path`: Path to pointcloud npz file
* `points_iou_path`: Path to points IoU npz file
* `sdf_octree_path`: Path to SDF values of Octree (currently depth 6)
* `sdf_grid128_path`: Path to grid SDF of resolution 128
* `sdf_grid256_path`: Path to grid SDF of resolution 256

Example: 
```
itera = ShapenetPreprocessedIterator()
print(itera[0])

>> {
     'id': '521eab9363fdc2a07209009cfb89d4bd', 
     'text': 'airliner', 
     'file_path': 'datasets/ShapeNetCore.v1/02691156/521eab9363fdc2a07209009cfb89d4bd/model.obj', 
     'base_dir': 'dataset/02691156/521eab9363fdc2a07209009cfb89d4bd', 
     'sdf_grid128_path': 'dataset/02691156/521eab9363fdc2a07209009cfb89d4bd/sdf_grid_128.npz', 
     'sdf_grid256_path': 'dataset/02691156/521eab9363fdc2a07209009cfb89d4bd/sdf_grid_256.npz', 
     'sdf_octree_path': 'dataset/02691156/521eab9363fdc2a07209009cfb89d4bd/sdf_octree_depth_6.npz', 
     'pointcloud_path': 'dataset/02691156/521eab9363fdc2a07209009cfb89d4bd/pointcloud.npz', 
     'points_iou_path': 'dataset/02691156/521eab9363fdc2a07209009cfb89d4bd/points_iou.npz', 
     'mesh_path': 'dataset/02691156/521eab9363fdc2a07209009cfb89d4bd/mesh.npz'
     'synsetId': 2691156,
     'subSynsetId': 2690373, 
     'split': 'test', 
     'category': 'airplane', 
     }
```

**NOTES**:
* Only for ShapeNet and Objaverse, there is a proper data split implemented. Passing the `mode` argument to the `DatasetIterator` will return only the files of the train / test / val data respectively.
