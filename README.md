# Genetic-Selection-Cheminformatics

This project contains feature selection by genetic algorithm for cheminformatics applications (also applicable to other datasets). We have provided sample application on the following 4 data sets (2 chemical, 2 toy datasets).

Classification:
- BBBP [1] (Blood-Brain Barrier Penetration dataset)
- Iris (Iris plants dataset)

Regression:
- AqSolDB [2] (Aqueous Solubility dataset)
- Boston (Boston house prices dataset)

Genetic algorithm is a heuristic method of optimization, inspired by evolution. Basically, it aims to create more successful individuals by transferring the best genes to new generations (see image below). In this application, we have implemented the genetic algorithm to select the features that best represent the target.

![alt text](https://raw.githubusercontent.com/mcsorkun/Genetic-Selection-Cheminformatics/master/images/genetic-score-change.png)




![alt text](https://raw.githubusercontent.com/mcsorkun/Genetic-Selection-Cheminformatics/master/images/genetic-algorithm-feature-selection.jpg)



### Dependencies

- rdkit==2020.03.2
- mordred==1.2.0
- scikit-learn==0.23.1
- pandas==1.0.3

### References

[1] Martins, Ines Filipa, et al. "A Bayesian approach to in silico blood-brain barrier penetration modeling." Journal of chemical information and modeling 52.6 (2012): 1686-1697.

[2] Sorkun, Murat Cihan, Abhishek Khetan, and Süleyman Er. "AqSolDB, a curated reference set of aqueous solubility and 2D descriptors for a diverse set of compounds." Scientific data 6.1 (2019): 1-8.

[3] Moriwaki, Hirotomo, et al. "Mordred: a molecular descriptor calculator." Journal of cheminformatics 10.1 (2018): 4.

