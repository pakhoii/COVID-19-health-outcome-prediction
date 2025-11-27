# COVID-19-health-outcome-prediction
HCMIU Data Mining Project

## Project Structure
```
COVID-19-HEALTH-OUTCOME-PREDICTION/
│
├── code/
│   └── src/
│       ├── main/
│       │   └── java/
│       │       └── com/hop/
│       │           ├── App.java                 # Primary entry point of the application.
│       │           │                             # Responsible for initializing the system,
│       │           │                             # invoking preprocessing modules, and running
│       │           │                             # predictive algorithms.
│       │           │
│       │           ├── algo1/                    # Module implementing Algorithm 1.
│       │           │   ├── README.md             # Documentation describing Algorithm 1,
│       │           │   │                         # usage, expected inputs, and output formats.
│       │           │   └── algo1.java            # Java class encapsulating the core logic of
│       │           │                             # Algorithm 1.
│       │           │
│       │           ├── algo2/                    # Module implementing Algorithm 2.
│       │           │   ├── README.md             # Documentation detailing Algorithm 2,
│       │           │   │                         # conceptual background, usage examples,
│       │           │   │                         # and integration notes.
│       │           │   └── algo2.java            # Java class providing the full implementation
│       │           │                             # of Algorithm 2.
│       │           │
│       │           ├── preprocess/               # Preprocessing module for dataset cleaning,
│       │           │   │                         # normalization, transformation, and feature
│       │           │   │                         # preparation prior to algorithm execution.
│       │           │   ├── README.md             # Detailed explanation of preprocessing steps,
│       │           │   │                         # supported methods, and data requirements.
│       │           │   └── preprocess.java       # Java class implementing all preprocessing
│       │           │                             # functions used in the workflow.
│       │           │
│       │           └── datasets/                 # Directory containing dataset metadata,
│       │                                           documentation, or schema descriptions.
│       │               └── README.md             # Notes describing datasets utilized in the
│       │                                         # project, including source, structure, and
│       │                                         # field definitions.
│       │
│       └── test/
│           └── java/
│               └── com/hop/
│                   └── AppTest.java              # Unit tests validating the functionality of
│                                                 # core application components.
│
├── data/                                         # Raw CSV datasets used for model training,
│   │                                             # evaluation, and exploratory analysis.
│   ├── comorbidity.csv
│   ├── covid.csv
│   ├── symptoms.csv
│
├── pom.xml                                       # Maven configuration file specifying project
│                                                 # dependencies, build plugins, and packaging.
│
├── target/                                       # Automatically generated build artifacts
│                                                 # (compiled .class files, packaged JARs, etc.).
│
├── .gitignore                                    # Git ignore rules preventing unnecessary or
│                                                 # large files from being committed.
│
├── Data_Mining_Project_Proposal.pdf              # Project proposal document outlining objectives,
│                                                 # methodology, and expected deliverables.
│
├── LICENSE                                       # License governing the usage and distribution
│                                                 # of the project’s source code.
│
└── README.md                                     # High-level project overview, setup
                                                  # instructions, execution steps, and
                                                  # documentation links.
```