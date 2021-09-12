
# Molecule Generation

## Directory structure
- `tokenizer`: tokenize SMILES from a dataset
- `configure`: this file contains the default configuration
- `train`: train model on your own dataset
- `sample`: generate molecules by sampling from the latent space



## Model training

To generate tokenizer file 

	./tokenizer.py

To train a new model

	./train.py
	
```
iter  | epoch | loss    
-----------------------  
  200   1.00  | 194.484    
  400   2.00  | 162.136    
  600   3.00  | 161.940    
  800   4.00  | 156.227    
 1000 * 5.00  | 156.765    
 1200   6.00  | 153.533    
 1400   7.00  | 153.566    
 1600   8.00  | 151.898    
 1800   9.00  | 151.989    
 2000 * 10.00  | 149.846    
 2200   11.00  | 151.761    
 2400   12.00  | 146.787    
 2600   13.00  | 144.065    
 2800   14.00  | 141.365    
 3000 * 15.00  | 140.018    
 3200   16.00  | 138.615    
 3400   17.00  | 135.574    
 3600   18.00  | 134.483    
 3800   19.00  | 132.043    
 4000 * 20.00  | 131.511
```
	  

