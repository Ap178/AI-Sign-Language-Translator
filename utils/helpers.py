# Common functions:

# ```python
# create_folder()
# load_json()
# save_numpy()
# ```
import os
import numpy as np
import pickle



def create_directory(path):

    """
    Create folder if not exists
    """

    if not os.path.exists(path):

        os.makedirs(path)



def save_numpy(data, path):

    """
    Save numpy array
    """

    create_directory(
        os.path.dirname(path)
    )

    np.save(
        path,
        data
    )



def load_numpy(path):

    """
    Load numpy array
    """

    return np.load(path)



def save_pickle(obj,path):

    with open(
        path,
        "wb"
    ) as f:

        pickle.dump(
            obj,
            f
        )



def load_pickle(path):

    with open(
        path,
        "rb"
    ) as f:

        return pickle.load(f)



def list_files(folder, extension):

    files=[]


    for root,dirs,names in os.walk(folder):

        for name in names:

            if name.endswith(extension):

                files.append(
                    os.path.join(
                        root,
                        name
                    )
                )


    return files