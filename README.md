# German-Traffic-Signs-Detector

# Quickstart

Please check requirements.txt before starting. 

**1. Download Dataset** 

To download the dataset and get the training and test images, should be executed as follows:

    python app.py download 

**2. Train Models** 

Before making predictions with the images of the test and user folders, you must train the models, in this way you will save the models in the "models" folder,should be executed as follows:

    python app.py train -m [chosen model] -d [directory with trainig data]


  
**3. Test Models** 

Before executing this command, make sure you have completed step 2, should be executed as follows:

    python app.py test -m [chosen model] -d [directory with trainig data]

**4. Infer new images**

Before executing this command, make sure you have completed step 2, should be executed as follows:
  
    python app.py infer -m [chosen model] -d [directory with user data]
