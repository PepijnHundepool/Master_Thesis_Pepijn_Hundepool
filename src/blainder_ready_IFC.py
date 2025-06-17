import bpy

def assign_walls_material_to_ifc_objects():
    material_name = "walls blainder"
    temp_coll = bpy.data.collections.get("temp")

    if not temp_coll:
        print("[ERROR] 'temp' collection not found.")
        return

    if material_name not in bpy.data.materials:
        print(f"[ERROR] Material '{material_name}' not found.")
        return

    mat = bpy.data.materials[material_name]
    count = 0

    for obj in temp_coll.all_objects:
        if obj.type == 'MESH':
            obj.data.materials.clear()
            obj.data.materials.append(mat)
            count += 1
            print(f"[INFO] Assigned '{material_name}' to {obj.name}")

    print(f"\n✅ Assigned 'walls blainder' to {count} objects.")
    
def assign_custom_properties():
    """ Assign custom properties for categoryID and partID to all objects in the scene. """
    for obj in bpy.data.objects:
        if obj.type == 'MESH':  # Ensure we only assign properties to mesh objects
            obj["categoryID"] = obj.name  # Set categoryID as object name
            obj["partID"] = obj.name  # Copy categoryID to partID
#            obj["inst_id"] = obj.pass_index  # Ensure unique instance ID if needed
            
            print(f"Assigned categoryID={obj.name}, partID={obj.name} to {obj.name}")

assign_walls_material_to_ifc_objects()
assign_custom_properties()
