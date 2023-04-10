# k-Nearest-Neighbor and Linear Regression for predicting type and quality of 3-D-printed material
A prediction on the type and quality of material printed by a 3D printer based on the chosen setup.

## Description
The data contains eight input variables (variables that define the setup of the 3D printer), three output variables (measured characteristics of the produced material) and one target variable (= the produced material itself).<br/>
A k-Nearest-Neighbor algorithm is applied in two different forms: First, the algorithm is fed with the input variables only in order to predict the produced material. Second, the algorithm is fed with both input and output variables in order to predict the produced material.<br/>
Beyond this, a linear regression is fitted on the input variables in order to separately predict the value of each of the three output variables.

## Data
Dataset: https://www.kaggle.com/datasets/afumetto/3dprinter

## Assessment Methods
For assessing the kNN algorithm, accuracy is computed, while for linear regression, mean squared error is applied.

## Results
kNN on input variables only: 90.00 %<br/>
kNN on input and output variables: 70.00 %<br/>
Predicting roughness of material with linear regression fitted on input variables: 0.17 µm<br/>
Predicting elongation of material with linear regression fitted on input variables: 0.22 mm<br/>
Predicting tension strength of material with linear regression fitted on input variables: 0.11 MPa<br/>

## Limitations
Predictions are based on only 40 training instances due to the small size of the dataset.

## Package Versions
python---3.7.6<br/>
pandas---1.0.1<br/>
numpy---1.21.6<br/>
matplotlib---3.1.3<br/>
seaborn---0.10.0<br/>

## Copyright Notice
No license is offered. Copyrights belong to the owner of this repository. The software provider does not represent or warrant that it has any rights whatsoever in the data used. Neither the software provider, nor any upstream software or data provider shall have any liability for any direct, indirect, incidental, special, exemplary, or consequential damages (including without limitation lost profits), however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the software, the data used or the produced results, even if advised of the possibility of such damages.
