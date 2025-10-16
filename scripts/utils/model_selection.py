import os
import mujoco

# Define all objects and their required files here
OBJECTS = {
    'box': {
        'xml_path': '../assets/object_sim/box.xml',
        'assets': []
    },
    'flashlight': {
        'xml_path': '../assets/object_sim/flash.xml',
        'assets': ['../assets/meshes/flashlight.stl']
    },
    'bunny': {
        'xml_path': '../assets/object_sim/stanfordbunny.xml',
        'assets': ['../assets/meshes/stanfordbunny.stl']
    },
    'box_exp': {
        'xml_path': '../assets/object_sim/box_exp.xml',
        'assets': [],
        'com': [-0.05, 0.05, 0.15]
    },
}

def select_model(main_xml_path: str, object_name: str):
    """
    Loads a MuJoCo model with a dynamically chosen object.

    This function uses MuJoCo's virtual file system (VFS) to insert an object's
    XML definition into a main template environment at runtime.

    Args:
        main_xml_path (str): The file path to the main template XML file.
                             This file should contain an <include file="object_to_load.xml"/>.
        object_choice (str): The key for the desired object (e.g., 'box', 'bunny', 'flashlight').
                             Must be a key in the OBJECT_DEFINITIONS dictionary.

    Returns:
        A tuple containing the loaded (mujoco.MjModel, mujoco.MjData).
        
    Raises:
        ValueError: If the object_choice is not valid.
        FileNotFoundError: If any of the required XML or asset files are not found.
    """
    if object_name is not None:

        print('This shouldnt be printing')
        if object_name not in OBJECTS:
            raise ValueError(f"Invalid object choice: '{object_name}'. "
                            f"Valid choices are: {list(OBJECTS.keys())}")

        # --- 1. Get the definition for the chosen object ---
        definition = OBJECTS[object_name]
        object_xml_path = definition['xml_path']
        obj_com = definition['com'] if 'com' in definition else [0, 0, 0]

        # --- 2. Build the assets dictionary for the VFS ---
        assets = {}

        # First, add the object's main XML content to the VFS.
        # The key 'object_to_load.xml' must match the filename in the <include> tag
        # of your main environment XML.
        try:
            with open(object_xml_path, 'r') as f:
                assets['object_to_load.xml'] = f.read().encode()
        except FileNotFoundError:
            print(f"Error: Could not find the object XML file at '{object_xml_path}'")
            raise

        # Next, add any other required asset files (like meshes) to the VFS.
        for asset_path in definition['assets']:
            try:
                with open(asset_path, 'rb') as f:
                    print(os.getcwd())
                    # The key for the asset must be just the filename, not the full path.
                    asset_filename = os.path.basename(asset_path)
                    assets[asset_filename] = f.read()
            except FileNotFoundError:
                print(f"Error: Could not find the required asset file at '{asset_path}'")
                raise
        
        # --- 3. Load the model using the assets dictionary ---
        print(f"Loading environment '{main_xml_path}' with object '{object_name}'...")
        try:
            model = mujoco.MjModel.from_xml_path(
                main_xml_path,
                assets=assets
            )
            data = mujoco.MjData(model)
            print("Model loaded successfully.")
            return model, data, obj_com
        except Exception as e:
            print(f"An error occurred during model compilation: {e}")
            raise

    else:
        object_name = 'none'
        assets = {}

        print(f"Loading environment '{main_xml_path}' with no object...")
        try:
            model   = mujoco.MjModel.from_xml_path(main_xml_path)
            data    = mujoco.MjData(model)
            print("Environment loaded successfully.")
            return model, data, [0, 0, 0]
        except Exception as e:
            print(f"An error occurred during model compilation: {e}")
            raise

    