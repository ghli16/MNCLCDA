Predicting circRNA-drug sensitivity associations by using mixed neighbourhood information and contrastive learning

## Code
### Environment Requirement
The code has been tested running under Python 3.6.8. The required packages are as follows
- python == 3.7.13
- pandas == 0.25.1
- numpy == 1.17.0
- scipy == 1.7.3
- pytorch == 1.9.0+cpu
- torch-geometric== 2.0.4

Files:
dataset
 1.gene_seq_sim.csv stores drug similarity matrix .
 2. association.csv stores circRNA-drug association information;
 3. drug_str_sim.csv stores drug similarity matrix .

src
 1.model.py：the MACLCDA framework;
 2.main.py: the training module.


