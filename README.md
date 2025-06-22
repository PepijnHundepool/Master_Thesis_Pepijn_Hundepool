# Master_Thesis_Pepijn_Hundepool
 This repository contains all source code related to the thesis: Synthetic Data-Driven Mismatch Detection for Updating BIM-Based Digital Twins from a Single Viewpoint.

 This project has leveraged the parametric dataset generator created by Pim van den Akker, see https://github.com/PimAkker/Final-Thesis-Pim-van-den-Akker, and the BLAINDER range scanner tool, see https://github.com/ln-12/blainder-range-scanner. Furthermore, the Bonsai extension was used to open IFC files inside Blender, see https://extensions.blender.org/add-ons/bonsai/. The dataset generator requires Blender 3.6, while BLAINDER and Bonsai require Blender 4.2. 
 
 Inside the blender_files folder are all the blender files that have been used throughout the project, with the most important one being room_generation.blend, which is the parametric dataset generator and thus needs to be ran from blender 3.6 by selecting main.py inside the scripting window and clicking the run button. All other blender files can be opened in Blender 4.2 with the BLAINDER tool installed. This allows you to see the scans that have been made and to regenerate scans (in .csv format) if needed. 

 Inside the code folder is all code that has been used throughout the project. All code files inside the src subfolder are universal across all specialized models, with test_master.py being the script that is used for real deployment and config.py containing all configurable settings for this total script. All other scripts are either helper scripts for test_master.py or are used during the data generation and preprocessing stage. In the other subfolders of the code folder are specialized scripts that have been used in different testing stages throughout the project. It must be noted that many files require you to insert the right file path, which is different for your pc. To make things easier, when you use ctrl + f and type: "change", you can easily find which lines of code you are allowed to change, not only for changing file paths, but also for configurable settings in some scripts. 

 Inside the datasets folder is all generated and preprocessed data that has been used throughout the project. This includes both training and testing data. It is recommended to use the same foldering and file naming logic as done here. You may note that the csv_files folders are empty. This is because the files are too large in size for Git. If you do want these .csv files, you must open the corresponding blender file, see blender_files folder, and regenrate the scan yourself. 

 Inside the thesis_and_presentation folder are the thesis and presentation. 

 Inside the trained_models folder are the final versions of all four specialzed mismatch detection models. 

# Manual for Data Generation:
 1. Generate a room in blender 3.6 using main.py inside room_generation.blend.
 2. Save it and open it in blender 4.2
 3.	Delete or hide objects that intersect.
 4. Delete objects that are underneath the generated room.
 5.	Run blainder_ready.py inside blender 4.2 to ready the file for point cloud generation by forcing a specific material property that is compatible with the BLAINDER tool. 
 6. Move 3D sensor (camera) if necessary.
 7. Unhide the ceiling. 
 8. Inside the BLAINDER menu, set 80 HFOV, 55 VFOV, width=1280, height=800, optionally add noise, export as .csv and change the filename. Then scroll up and click "generate" to obtain the DT scan. 
 9. (optional) Run pointcloud_coloring_V2.py (not inside blender anymore) to view a semantically segmented pointcloud based on the categoryID’s in the .csv file. 
 10. Copy the room blender file and manually induce a mismatch on a structural element, thus creating a modified version of the room. When copying an element, make sure to assign the right material to it by running blainder_ready.py inside blender again. 
 11. Repeat earlier steps to get the robot scan. It is most wise to keep the DT scan and robot scan pair together in the same folder. 

# Manual for Data Preprocessing
 1.	Generate rooms and scans in Blender, see above. 
 2. Run object_filtering.py for all robot scans that have occlusion (chairs or tables). This script works the same for all four scenarios. The object-filtered robot scans are saved in the object_filtered_csv_files folder. 
 3. Check the object_filtered robot scans again with pointcloud_coloring_V2.py to make sure the occluding objects are really removed. 
 4. Copy all the dt scan files from the csv_files folder to the object_filtered_csv_files folder and add the suffix "_object_filtered" to it so that it has the same naming format as the object filtered robot scans in that folder. 
 5. Run pointcloud_folder_noise_preprocessing_csv.py for all files inside the object_filtered_csv_files folder. This script works the same for all four scenarios. The files are then saved inside the preprocessed_csv_files folder. 
 6. Make sure there are no augmented samples in stage yet. 
 7. Check the preprocessed csv files again with pointcloud_coloring_V2.py.
 8. Run generate_labels_csv_[scenario type].py to get both correct and incorrect labels, depending on the scenario. Files are saved inside the labels_csv folder. 
 9. Check first version of labels with view_labels_csv_[scenario type].py.
 10. Run label_filtering_csv_[scenario type].py only for occluded object-removed cases. Files are saved inside the labels_csv_filtered folder. All other cases should just be copy-pasted into this folder. 
 11. Check final version of labels again with view_labels_csv_[scenario type].py.
 12. (optional) Run data_augmentation.py on the labels_csv_filtered folder.  
 13. Run convert_csv_to_npy_labels_[scenario type].py to get npy files instead of csv. The files are stored inside the npy_files_filtered folder and the labels_npy_filtered folder. 
 14. Check files with view_labels_npy_filtered_[scenario type].py.
 15. Train a model with train_[scenario type].py. 

 # Manual for Optimal Workflow
 The order of source code files for the entire pipeline is:
 1. pointcloud_coloring.py
 2. object_filtering.py
 3. pointlcoud_folder_noise_preprocessing_csv.py
 4. generate_labels_csv_multi_type.py
 5. view_labels_csv_multi_type.py
 6. label_filtering_csv_multi_type.py
 7. (in the case of both an mismatched pillar and wall in the same can) label_filtering_csv_multi_type_walls.py
 8. label_addition.py
 9. convert_csv_to_npy_labels_multi_type.py
 10. view_labels_npy_filtered_multi_type.py
 11. train_master.py
 12. plot_training.py
 13. config.py
 14. test_master.py, which uses bounding_box_helpers.py and MD_performance_helpers.py as helper scripts. 
