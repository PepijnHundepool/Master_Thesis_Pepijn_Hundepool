import bpy

def cleanup_scene():
    """ Remove unnecessary objects from the scene (2D LiDAR, 3D LiDAR, and placable objects). """
    
    # Delete 2D and 3D LiDAR objects and old camera
    lidar_objects = ["raytrace", "LiDAR_Sphere", "Camera"]
    for obj_name in lidar_objects:
        obj = bpy.data.objects.get(obj_name)
        if obj:
            bpy.data.objects.remove(obj, do_unlink=True)

    # Delete all objects in the "placable objects" collection
    collection_name = "placable objects"
    if collection_name in bpy.data.collections:
        for obj in bpy.data.collections[collection_name].objects:
            bpy.data.objects.remove(obj, do_unlink=True)

    print("✅ Cleaned up unnecessary objects!")
    
def remove_custom_properties():
    """ Remove the 'inst_id' property from all objects in the scene. """
    for obj in bpy.data.objects:
        if "inst_id" in obj:
            del obj["inst_id"]
            print(f"Removed inst_id from {obj.name}")

def assign_blainder_materials():
    """ Assign BLAINDER materials to relevant meshes based on object names. """
    
    # Define a mapping from standard materials to BLAINDER materials
    material_mapping = {
        "chairs display": "chairs display blainder",
        "doors": "doors blainder",
        "pillars display": "pillars display blainder",
        "tables display": "tables display blainder",
        "walls": "walls blainder",
        "Plane": "walls blainder",
        "Ceiling": "walls blainder"
    }

    # Assign materials to objects based on name matching
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.data.materials:
            base_name = obj.name.split(".")[0]  # Extract base name (e.g., "chairs display" from "chairs display.001")
            new_material_name = material_mapping.get(base_name)

            if new_material_name and new_material_name in bpy.data.materials:
                obj.data.materials.clear()  # Remove any existing materials
                obj.data.materials.append(bpy.data.materials[new_material_name])
                print(f"✅ Assigned {new_material_name} to {obj.name}")
            else:
                print(f"⚠️ No matching material found for {obj.name}")
                
def assign_custom_properties():
    """ Assign custom properties for categoryID and partID to all objects in the scene. """
    for obj in bpy.data.objects:
        if obj.type == 'MESH':  # Ensure we only assign properties to mesh objects
            obj["categoryID"] = obj.name  # Set categoryID as object name
            obj["partID"] = obj.name  # Copy categoryID to partID
#            obj["inst_id"] = obj.pass_index  # Ensure unique instance ID if needed
            
            print(f"Assigned categoryID={obj.name}, partID={obj.name} to {obj.name}")

def run_setup():
    cleanup_scene()
    remove_custom_properties()
    assign_blainder_materials()
    assign_custom_properties()
    print("✅ Scene preparation for BLAINDER LiDAR completed!")

run_setup()
