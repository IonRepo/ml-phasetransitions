# ml-phasetransitions

In the pursuit of energy-efficient and environmentally friendly cooling technologies, barocaloric materials have emerged as promising candidates due to their substantial thermal changes under applied pressure. The discovery of new materials with enhanced thermodynamic-related effects is essential for the advancement of solid-state cooling devices. To address this challenge, we introduce a novel approach based on Graph Convolutional Neural Networks (GCNNs) to predict the phase transitions of materials in large datasets such as the Materials Project database.
    
Our GCNN model leverages a large and diverse dataset of ~6000 harmonic DFT phonon calculations, extracting valuable insights from the underlying crystallographic and structural relationships.
    
The predictive power of our GCNN-based approach is demonstrated through extensive validation on benchmark datasets, showcasing its ability to accurately forecast the barocaloric behavior of both known and undiscovered materials. The integration of GCNNs into the materials discovery pipeline holds great promise for the development of next-generation solid-state refrigeration technologies, introducing here a framework which allows predicting on any customized dataset.

## Features

- Predicts Helmholtz free energies for crystal materials.
- Detects first-order phase transitions and estimates their temperature and entropy changes.
- Analyzes transitions between polymorphs, including centrosymmetry and ferromagnetism detection.

## Usage

Detailed explanations on the utilities of the library are shown in the examples folder.

### Example steps:

1. **Data Preparation**: Ensure your dataset includes crystal structures in POSCAR format and associated ground state energy (EPA) files. Materials must include atomic information such as mass, charge, and ionization energy.
   
2. **Training/Prediction**: The script allows for both training a new model and using a pretrained GCNN model. You can configure the training settings through the provided notebook interface.

3. **Run Model**: Once data is prepared and the model is trained, use the example notebook to make predictions on your dataset.

4. **Analysis**: Visualization tools are included to plot phase transitions, entropy changes, and the transition temperatures for materials in the dataset.

## Installation

To download the repository and install the required dependencies:

```bash
git clone https://github.com/YourUsername/ml-phasetransitions.git
cd ml-phasetransitions
pip3 install -r requirements.txt
```

## Authors

This repository is being developed by:

- **Cibrán López Álvarez** - Lead Developer and Researcher
- **Claudio Cazorla Silva** - Supervisor

## Contact, Questions, and Contributing

For any questions, issues, or contributions, feel free to contact:

- Cibrán López Álvarez: [cibran.lopez@upc.edu](mailto:cibran.lopez@upc.edu)

Feel free to open issues or submit pull requests for bug fixes, improvements, or feature suggestions.
