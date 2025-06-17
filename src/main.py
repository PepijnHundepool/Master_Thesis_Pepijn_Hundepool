import os
import sys


# ensure we are in the correct directory
import bpycv

# ensure we are in the correct directory
root_dir_name = 'Blender'
root_dir_path = os.path.abspath(__file__).split(root_dir_name)[0] + root_dir_name
os.chdir(root_dir_path)
sys.path.extend([os.path.join(root_dir_path, dir) for dir in os.listdir(root_dir_path)])

# add all the subdirectories to the path
dirs  = os.listdir()
root = os.getcwd()
for dir in dirs:
    sys.path.append(os.path.join(root, dir))
sys.path.append(os.getcwd())
import generate_dataset
from category_information import category_information
import numpy as np
from importlib import reload
import custom_render_utils
import blender_python_code.data_gen_utils as data_gen_utils
from category_information import category_information

import bpy

# force a reload of object_placement_utils to help1 during development
reload(data_gen_utils)
reload(custom_render_utils)
reload(bpycv)
reload(generate_dataset)


# these are the per ablation run parameters
nr_of_images = 1
overwrite_data = False
empty_folders = True
minimum_overlap_percentage_for_visible = 0.1
objects_to_add_percentage = 0.6666
objects_to_remove_percentage = 0.333
object_to_move_percentage = 0 # true object to move percentage = object_to_move_percentage * objects_to_add_percentage
force_object_visibility = ['walls','LiDAR_Sphere'] # categorie(s) that should always be visible in the map
max_shift_distance =.5
output_parent_folder = r"data\test1\wheredidthepillarsgo"
output_map_resolution = [280,280] # the pixels of the map

bpy.data.materials["chairs display blainder"].use_fake_user = True  # Prevents deletion
bpy.data.materials["doors blainder"].use_fake_user = True  # Prevents deletion
bpy.data.materials["pillars display blainder"].use_fake_user = True  # Prevents deletion
bpy.data.materials["tables display blainder"].use_fake_user = True  # Prevents deletion
bpy.data.materials["walls blainder"].use_fake_user = True  # Prevents deletion

ablate_over_parameters = [
                            # {"wall width":'mean'}, 
                        #   {"wall nr x":0, "wall nr y":0}, 
                        #   {"low freq noise variance":'mean'},
                        #   {"wall density":'mean'},
                        #   {"min door width":1.0, "max door width":1.0},
                        #   {'max door rotation':0},
                        #   {'max wall randomness':0},
                        #   {'door density':'mean'},
                        #   {'chair width':'mean', 'chair length':'mean'},
                        #   {"table_legs":4},
                        #   {"table x width":'mean', "table y width":'mean'},
                        #   {"leg radius":'mean'},
                        #   {"high freq noise variance":0},
                        #   {"low freq noise variance":0}                  
                          ] # the parameters that will be ablated over one by one 

obj_ids = category_information

walls_doors_modifiers = {
    "wall width": (0.1, 0.3),
    "wall nr x": (0, 2),
    "wall nr y": (0, 2),
    "wall density": (0.7, 1),
    "seed": (0, 10000),
    "min door width": 0.7,
    "max door width": 1.3,
#    "max wall randomness": (0, 1.6),
    "max wall randomness": (0, 0.8),
    "max door rotation": (np.pi, 1.75*np.pi),
    "door density": (0.999, 1),
    "height":3,
    "pillar min dist": (2,5),
    "chair min dist": (1,5),
    "table min dist": (1,5),
}

chair_size = (0.8, 1.2)
chairs_modifiers = {
    "chair width": chair_size,
    "chair length": "chair width",
    "leg height": chair_size,
    "back rest offset": (0.3,0.6),
    "back rest height": (0.3,0.8),
    "leg width": (0.05, 0.1),
    "circular legs": np.random.choice([True, False]),
    "leg type": (0,1),
    "leg center offset percentage":(0.5,1)
}

table_size = (0.5, 0.9)
round_table_modifiers = {
    "table legs": (4),
    "table height": 1, 
    "table x width": table_size,
    "table y width": table_size,
    "leg radius": (0.05, 0.12),
    
}

pillar_modifiers = {
    "width": (0.3, 1.3),
    "height":3,
    "round/square": np.random.choice([True, False]),
}
LiDAR_height = (0.2, 0.21) # height from the bottom of the walls
# LiDAR_height = (0.2)

#raytrace_modifiers = {"high freq noise variance": (0, 0.1), 
#                      "low freq noise variance": (0, 0.44),
#                      "lidar block size":(0.05,0.2),
#                      "nr lidar points": (500,512)
#                      }
                      
raytrace_3D_modifiers = {"lidar block size":(0.05,0.2),
                      "nr lidar points": (500,512)
                      }

# these colors are used for the map not for the annotations
set_colors = {
            "walls": (0,0, 0, 255),  # Black
            "chairs display": (0, 255, 0, 255),  # Green
            "tables display": (0, 0, 255, 255),  # Blue
            "pillars display": (255, 255, 0, 255),  # Yellow
            "doors": (255, 0, 255, 255),  # Magenta
            "LiDAR_Sphere": (255, 0, 0, 255),  # Red
        }

output_parent_folder = os.path.join(os.getcwd(), output_parent_folder)

if ablate_over_parameters == []:
    ablate_over_parameters = [{}]
    
# Pepijn
import bpy

# Function to add 3D LiDAR to the scene
def add_lidar_to_scene(lidar_position=(0, 0, 1), lidar_group_name="raytrace_3D", lidar_material_name="LiDAR_Sphere"):
    # Add a new sphere to the scene
    bpy.ops.mesh.primitive_uv_sphere_add(location=lidar_position, radius=5)
    lidar_sphere = bpy.context.object
    lidar_sphere.name = "LiDAR_Sphere"
    
    # Add a Geometry Nodes modifier
    geo_nodes_modifier = lidar_sphere.modifiers.new(name="LiDAR", type='NODES')
    if lidar_group_name in bpy.data.node_groups:
        geo_nodes_modifier.node_group = bpy.data.node_groups[lidar_group_name]
    else:
        raise ValueError(f"Geometry node group '{lidar_group_name}' not found. Please create it in Blender.")
    
    # Assign the material to the LiDAR_Sphere object
    if lidar_material_name in bpy.data.materials:
        lidar_sphere.data.materials.append(bpy.data.materials[lidar_material_name])
    else:
        raise ValueError(f"Material '{lidar_material_name}' not found. Please create it in Blender.")
    
    # Move LiDAR_Sphere to the "Collection" collection
#    collection_collection = bpy.data.collections.get("Collection")
#    if collection_collection:
#        # Remove from all other collections, if necessary
#        for col in lidar_sphere.users_collection:
#            col.objects.unlink(lidar_sphere)
#        # Link to "Collection"
#        collection_collection.objects.link(lidar_sphere)
#    else:
#        raise ValueError("Collection collection not found. Please ensure the collection exists in Blender.")
        
#    # Assign the raytrace_3D node group to the modifier
#    if lidar_group_name in bpy.data.node_groups:
#        lidar_modifier.node_group = bpy.data.node_groups[lidar_group_name]
#    else:
#        raise ValueError(f"Node group '{lidar_group_name}' not found in the Blender file.")
    
    return lidar_sphere


## Pepijn Add LiDAR to the center of the room
#lidar_position = (0.7, 18.8, LiDAR_height[0])  # Set the LiDAR height from the configuration
#add_lidar_to_scene(lidar_position=lidar_position)


for fixed_modifier in ablate_over_parameters:
    
    # Pepijn Cleanup existing LiDAR objects
    def cleanup_lidar():
        lidar_name = "LiDAR_Sphere"  # Or the name of your 3D LiDAR object
        for obj in bpy.data.objects:
            if obj.name.startswith(lidar_name):
                bpy.data.objects.remove(obj, do_unlink=True)

    
    # try: 
    gen = generate_dataset.generate_dataset(nr_of_images=nr_of_images, 
                    folder_name=os.path.join(output_parent_folder,f"{list(fixed_modifier)}"),
                    overwrite_data=overwrite_data,
                    empty_folders=empty_folders, 
                    minimum_overlap_percentage_for_visible=minimum_overlap_percentage_for_visible, 
                    objects_to_add_percentage=objects_to_add_percentage, 
                    objects_to_remove_percentage=objects_to_remove_percentage, 
                    object_to_move_percentage=object_to_move_percentage,
                    force_object_visibility=force_object_visibility, 
                    max_shift_distance=max_shift_distance, 
                    walls_modifiers=walls_doors_modifiers, 
                    chairs_modifiers=chairs_modifiers, 
                    round_table_modifiers=round_table_modifiers,
                    pillars_modifiers=pillar_modifiers,
#                    raytrace_modifiers=raytrace_modifiers, 
                    raytrace_3D_modifiers=raytrace_3D_modifiers,
                    set_colors=set_colors, 
                    ablation_parameter=fixed_modifier,
                    map_resolution=output_map_resolution,
                    LiDAR_height=LiDAR_height
                    )
        
    cleanup_lidar()  # Ensure no leftover LiDAR objects from previous runs    
        
    # Pepijn Add LiDAR to the scene
    lidar_position = (2, 20, 0.4)  # Position at the center of the room
    add_lidar_to_scene(lidar_position=lidar_position)
    
    # except Exception as e:
    #     # when an error occurs write it to the error log but continue with the next 
    #     # ablation parameter
        
    #     import datetime
    #     errorlog_location = os.path.join(os.getcwd(), "error_log.txt")
    #     with open(os.path.join(output_parent_folder,("error_log.txt")), "a") as f:
    #         f.write(f"Error in fixed modifier {fixed_modifier} \n")
    #         f.write(f"time: {datetime.datetime.now()} \n")
    #         f.write("\n")
    #         f.write(f"{e} \n")
    #         f.write("\n")



