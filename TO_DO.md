# TO DO LIST for the master project

## Relevance - Project Understanding
- [ ] Read the two papers and write summary on Overleaf (Week 19-20)

        1. [x] 3d dataset format- Voxel
        2. [x] how relevant is the order of sequence: They are relevant, but different sequence may generate the same object
        3. [x] How many points in the dataset
        4. [x] What metrics have beem used to evaluate the DL model : Some metrics in the voxel space

        Week 20-21
        5. [x] What does the data mean. What each parameter of a shape depicts. Add this in the documentation "Reademe.md" or "Latex file" : created in the latex


## Relevance - Deep Learning Approaches
- [ ] Train a sinmple CNN model for CAD classification 

        1. [x] Extract all voxel represtations for one-ops expressions
        2. [x] Make a classification dataset with following classes : (SP,SP) (SP,CU) (CU, CU) (CY,CY) (CY,SP) (CY,CU)

        Week 20-21
        3. [x] Train a simple 3d CNN model and obtain classification accuracy --> Task 7
        4. [x] Upload the code for training and creating dataset 
        5. [x] Upload the voxel dataset --> Task 6 

        Week 23-24
        6. [x] Upload the large datasets on seafile: Partially Uploaded: Deprecate: Use scripts to make dataset
        7. [x] Add milestones and add results of the classifcation there --> Task 9
        8. [x] Organize code into files and folders for better meintenance. Use clean code principles, Remove deprecated : partially
        9. [x] Add proper information about which dataset, also add plots instead of raw text
        10. [x] Add a readme file with instructions to download data and train data with arguments : Almost done
        11. [ ] Add plots for confusion matrix : not so important 
        12. [x] add the scripts to make voxel datasets

        week 25:
        13. [x] Update Readme file for training data flags. --check it :Saurbah
        14. [x] Add evaluation in terms of Bleu score 
        15. [x] Add another approach for CNN-LSTM that predicts the whole sequence at once and not just one step and report similarly as one step
        16. [ ] Study Chamfer distance

        week 26:
        17. [x] Check tzhe bleu score implementation. Check if accuracy is also dropping there 
        18. [x] Try another model with bigger capayity to see if the model improves --> it improves the bleu score

        week 29:
        19. [ ] Comparison table for the charts you have created
        20. [ ] Comparison chart for the transformer and Cnv-LSTM model

- [ ] VorPlanML dataset
        week 29:
        21. [ ] Data preprocessing of all the excel files and create one csv preprocessed file
        22. [ ] Check if all corresponding binvox files are present in the the dataset
        23. [ ] Check if you can use the data loder provided in sea file
         
- [ ] train a CNN-LSTM based model in a typical Image captioning setup

        Week 23-24 
        1. [x] Extract all voxel representations for 3-ops expressions
        2. [x] Make a captioning kind of dataset where for each datapoint you have a label as a sequence: partiallly done: just add the Voxel filename as a column
        3. [x] Implement 3D CNN-LSTM based architecture 
        4. [x] Train it on the created dataset, report accuracy  
        5. [x] Upload the code

        week 25:
        6. [ ] Add code and documentation for evaluation in terms of loss and accuracy and plots for them 


- [ ] train a transformer based model in an Image captioning setup

        Week 29:
        7. [ ] Setup a transformer model :::: we can have a call ? 
        8. [ ] Train the model to predict sequences
        9. [ ] Report Bleu score and accuracy


 

## Relevace - Code Issues

        Week 25
        1. [x] Update the CNN-LSTM milestone to show a few predictions on the test data---see 2

        Week 29
        2. [ ] In Milestone 3 add some predictions of the data....

## Relevance - Data Loaders and Data Handling
- [x] Search for a simple tool (e.g. VTK ) to visualize voxel represesentation : Matplotlib  



