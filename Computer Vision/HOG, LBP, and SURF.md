
En reconnaissance des formes, une image brute n’est generalement pas directement utilisee par un classifieur classique. Elle est d’abord transformee en une representation numerique appelee vecteur de caracteristiques. La chaıne generale ressemble a:

$$ Image \rightarrow Descripteur \rightarrow Caracteristique \rightarrow MLP \rightarrow Classe $$

- **HU**: forme globale;
- **H**istogram of **O**riented **G**radients: structure des contours et orientations des gradients;
- **L**ocal **B**inary **P**attern: texture locale;
- **S**peeded-**U**p **R**obust **F**eature: caracteristiques locales et points d'interet.

The