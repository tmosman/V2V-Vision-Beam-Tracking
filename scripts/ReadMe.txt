Steps to generate datasets and train ML models;

1) Generate the polar cordinates from the YOLO coordinates, for all the sequences of datasets.
 NB: The datasets from the DeepSense6G scenario 36 has seq_idx 1 to 13. But after further data inspection and cleaning, we utilized data samples with the seq_index 1, 5, 6, 9, and 12.
 --> Script to perform step 1; ./scripts/generate_polar_cord_for_all_seqs.py

2) Train and test a fully-connected neural network for the Transmitter Identification task, using the scripts in the directory; ./scripts/TX_ID_Problem/
 (i) ./FC_model_training_TX_ID_v1.py -- Can be used train a model to map normalized power vector to the polar distance/angle. Change the IND value to select dist/angle
 (ii)./FC_model_training_TX_ID_v2.py -- Can be used train a model to map normalized power vector to both polar distance&angle.

3) Transmitter Identifaction and tracking in all the data points. Scripts in directory ./scripts/TX_Tracking_Problem/
(i) Use the FC model from step 2 to predict the TX coordinates in all the data samples.
 --> ./Use_FC_model_to_predict_TX_coordinate_all_seq.py
(ii) Perform Transmitter coordinate tracking with window size, r=5.
 --> ./create_sequential_points_fromTxTracking_v1.py


4) Train and test LSTM model for beam prediction. 
 (i) Create a sequential datasets for the LSTM model using ./scripts/TX_Tracking_Problem/create_dataset_for_LSTM_training.py
 (ii) Run LSTM_model_training_v1.py to train the LSTM model.

5) Plot the results using the scripts in the directory; ./scripts/Plot_results/
