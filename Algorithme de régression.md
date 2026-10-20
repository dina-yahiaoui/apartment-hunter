# **Principaux algorithms de régression**



### Régression linéaire



La régression linéaire permet de prédire la valeur d'une variable dépendante (y) en fonction d'une ou plusieurs variables indépendantes données ( , ,…). Cette technique de régression établit donc une relation entre x (l'entrée) et y (la sortie).



#### **a. Régression linéaire simple**



Dans le cas d’une régression linéaire simple, le modèle de régression établit une relation entre la variable d’entrée indépendante, soit la variable explicative (x), et la variable de sortie continue, soit la variable à expliquer (y).



La régression linéaire simple est le modèle le plus utilisé en finance. En effet, il peut être utilisé pour la gestion de portefeuille, l’évaluation des actifs et l’optimisation. Par exemple, il peut être employé pour prévoir les retours et les performances opérationnelles d’une entreprise. Cependant, il existe certains cas pour lesquels la variable à expliquer ne dépendent pas que d’un seul facteur. Ainsi, nous aurons besoin d’un modèle pour évaluer des situations un peu plus complexes : nous utiliserons la régression linéaire multiple. 



Exemple : 



1\. Prédire le prix d'une maison en fonction de sa superficie (m²). Plus la maison est grande, plus le prix augmente, de façon assez linéaire.





2\. Prédire le poids d'une personne en fonction de sa taille, ou la tension artérielle en fonction de l'âge.



###### ***A retenir*** : 



1. Il n'y a qu'une seule variable explicative (sinon on utilise la régression linéaire multiple)

2\. La relation entre x et y est à peu près linéaire (une ligne droite représente bien la tendance)



#### **b. Régression linéaire multiple**





Dans certains cas, la variable de sortie (y), que nous essayons de prédire, dépend de plus d’une variable. Ainsi, un modèle plus élaboré, qui prend en compte cette dimension supérieure, est nécessaire. C’est ce qu’on appelle la régression linéaire multiple. L’utilisation d’un plus grand nombre de variables indépendantes peut permettre d’améliorer la précision du modèle, tant que les variables ajoutées sont pertinentes à celui-ci.



Exemple : 



1. Prédire le prix d'une maison selon la superficie, le nombre de chambres, l'âge de la maison, et la distance du centre-ville — plusieurs facteurs à la fois.



2\. Prédire le risque de maladie cardiaque en fonction de l'âge, du poids, de la tension artérielle et du taux de cholestérol.



###### ***A retenir :***



1. On utilise la régression multiple dès que plusieurs facteurs influencent y en même temps, ce qui est le cas le plus fréquent dans la vraie vie, peu de phénomènes dépendent d'une seule variable.



#### **c. Régression polynomiale**



La régression polynomiale peut être considérée comme un cas particulier de la régression linéaire multiple, dans lequel la distribution des données est plus complexe qu’une simple distribution linéaire. En d’autres mots, la relation entre la variable dépendante x et la variable indépendante y est modélisée comme le N-ième degré polynomial de x. Cet algorithme peut générer une courbe pour des données non-linéaires.



Exemple :



1. La croissance d'une plante n'est pas constante dans le temps : elle est lente au début, rapide au milieu, puis ralentit, une courbe, pas une droite.



2\.





