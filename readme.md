
# Molecule Generation

## Directory structure
- `tokenizer`: tokenize SMILES from a dataset
- `configure`: this file contains the default configuration
- `train`: train model on your own dataset
- `sample`: generate molecules by sampling from the latent space



## Model training

1. fit tokenizer 

	    ./tokenizer.py

2. train a new model

	    ./train.py
	```
	iter   | epoch | loss  
	-----------------------
	000200 | 0001  | 193.60  
	000400 | 0002  | 164.02  
	000600 | 0003  | 161.56  
	000800 | 0004  | 158.10  
	001000 | 0005  | 154.25  
	001200 | 0006  | 152.85  
	001400 | 0007  | 152.83  
	001600 | 0008  | 151.17 
	```

3. generate molecules from a trained model
   
		./sample.py
	  ```
	input: COc1cc2c(c3ccccc3n2C)c4c(Cl)nccc14
	output: COc1cc2ccccccccc2))cccc(()ccccc14
	input: CC(C)Sc1ccc(CC2CCN(CC2)C3CCN(CC3)C(=O)c4cccc5cccnc45)cc1
	output: CC(C)Cc1ccc(CC2CCNCCC2CC3CCCCCCCCCC())c4cccccccccc5))cc1
	input: [Br-].Oc1ccc(cc1)C(=O)CCN2CC[N+]3(CCCC3)CC2
	output: [Br-].Oc1ccc(cc1)C(=O)CC2CCCCCCCCCCCCC2
	input: C[C@@H]([C@H](CS)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O)c3ccc(cc3)c4ccccc4
	output: C[C@@H](C(()NC(=O)C(CCc1ccc2ccccc12CC=OO)Occcccccccc)c4cccccc4
	```


