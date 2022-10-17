# freelancer-com-diabetes-mellitus

![Contributors](https://img.shields.io/github/contributors/walidsi/freelancer-com-diabetes-mellitus?style=plastic)
![Forks](https://img.shields.io/github/forks/walidsi/freelancer-com-diabetes-mellitus)
![Downloads](https://img.shields.io/github/downloads/walidsi/freelancer-com-diabetes-mellitus/total)
![Stars](https://img.shields.io/github/stars/walidsi/freelancer-com-diabetes-mellitus)
![Licence](https://img.shields.io/github/license/walidsi/freelancer-com-diabetes-mellitus)
![Issues](https://img.shields.io/github/issues/walidsi/freelancer-com-diabetes-mellitus)

## Goal
The goal of this project is to predict whether a patient has diabetes mellitus or not. The project is a supervised learning project.

### Install

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)


### Code

### Run

In a terminal or command window, navigate to the top-level project directory (that contains this README) and run one of the following commands:

```bash
ipython notebook diabetes-mellitus-prediction.ipynb
```  
or
```bash
jupyter notebook diabetes-mellitus-prediction.ipynb
```

This will open the iPython Notebook software and project file in your browser.

### Data

The dataset consists of approximately 79000 data points, with each datapoint having 87 features. 

**Features**
- `encounter_id`: 	Unique identifier associated with a patient unit stay
- `hospital_id`: 	Unique identifier associated with a hospital
- `age`: 	The age of the patient on unit admission
- `bmi`: 	The body mass index of the person on unit admission
- `elective_surgery`: 	Whether the patient was admitted to the hospital for an elective surgical operation
- `ethnicity`: 	The common national or cultural tradition which the person belongs to
- `gender`: 	The genotypical sex of the patient
- `height`: 	The height of the person on unit admission
- `icu_type`: 	A classification which indicates the type of care the unit is capable of providing
- `readmission_status`: 	Whether the current unit stay is the second (or greater) stay at an ICU within the same hospitalization
- `weight`: 	The weight (body mass) of the person on unit admission
- `bilirubin_apache`: 	The bilirubin concentration measured during the first 24 hours which results in the highest APACHE III score
- `creatinine_apache`: 	The creatinine concentration measured during the first 24 hours which results in the highest APACHE III score
- `gcs_eyes_apache`: 	The eye opening component of the Glasgow Coma Scale measured during the first 24 hours which results in the highest APACHE III score
- `gcs_motor_apache`: 	The motor component of the Glasgow Coma Scale measured during the first 24 hours which results in the highest APACHE III score
- `heart_rate_apache`: 	The heart rate measured during the first 24 hours which results in the highest APACHE III score
- `intubated_apache`: 	Whether the patient was intubated at the time of the highest scoring arterial blood gas used in the oxygenation score
- `map_apache`: 	The mean arterial pressure measured during the first 24 hours which results in the highest APACHE III score
- `paco2_apache`: 	The partial pressure of carbon dioxide from the arterial blood gas taken during the first 24 hours of unit admission which produces the highest APACHE III score for oxygenation
- `ventilated_apache`: 	Whether the patient was invasively ventilated at the time of the highest scoring arterial blood gas using the oxygenation scoring algorithm, including any mode of positive pressure ventilation delivered through a circuit attached to an endo-tracheal tube or tracheostomy
- `wbc_apache`: 	The white blood cell count measured during the first 24 hours which results in the highest APACHE III score
- `d1_heartrate_max`: 	The patient's highest heart rate during the first 24 hours of their unit stay
- `h1_spo2_max`: 	The patient's highest peripheral oxygen saturation during the first hour of their unit stay
- `h1_temp_max`: 	The patient's highest core temperature during the first hour of their unit stay, invasively measured
- `h1_temp_min`: 	The patient's lowest core temperature during the first hour of their unit stay
- `d1_albumin_max`: 	The lowest albumin concentration of the patient in their serum during the first 24 hours of their unit stay
- `d1_albumin_min`: 	The lowest albumin concentration of the patient in their serum during the first 24 hours of their unit stay
- `d1_bilirubin_max`: 	The highest bilirubin concentration of the patient in their serum or plasma during the first 24 hours of their unit stay
- `d1_bilirubin_min`: 	The lowest bilirubin concentration of the patient in their serum or plasma during the first 24 hours of their unit stay
- `d1_bun_max`: 	The highest blood urea nitrogen concentration of the patient in their serum or plasma during the first 24 hours of their unit stay
- `d1_bun_min`: 	The lowest blood urea nitrogen concentration of the patient in their serum or plasma during the first 24 hours of their unit stay
- `d1_glucose_min`: 	The lowest glucose concentration of the patient in their serum or plasma during the first 24 hours of their unit stay
- `d1_hco3_max`: 	The highest bicarbonate concentration for the patient in their serum or plasma during the first 24 hours of their unit stay
- `d1_hco3_min`: 	The lowest bicarbonate concentration for the patient in their serum or plasma during the first 24 hours of their unit stay
- `d1_inr_max`: 	The highest international normalized ratio for the patient during the first 24 hours of their unit stay
- `d1_inr_min`: 	The lowest international normalized ratio for the patient during the first 24 hours of their unit stay
- `d1_lactate_max`: 	The highest lactate concentration for the patient in their serum or plasma during the first 24 hours of their unit stay
- `d1_lactate_min`: 	The lowest lactate concentration for the patient in their serum or plasma during the first 24 hours of their unit stay
- `d1_platelets_max`: 	The highest platelet count for the patient during the first 24 hours of their unit stay
- `d1_platelets_min`: 	The lowest platelet count for the patient during the first 24 hours of their unit stay
- `d1_potassium_max`: 	The highest potassium concentration for the patient in their serum or plasma during the first 24 hours of their unit stay
- `d1_potassium_min`: 	The lowest potassium concentration for the patient in their serum or plasma during the first 24 hours of their unit stay
- `d1_sodium_max`: 	The highest sodium concentration for the patient in their serum or plasma during the first 24 hours of their unit stay
- `d1_sodium_min`: 	The lowest sodium concentration for the patient in their serum or plasma during the first 24 hours of their unit stay
- `d1_wbc_max`: 	The highest white blood cell count for the patient during the first 24 hours of their unit stay
- `d1_wbc_min`: 	The lowest white blood cell count for the patient during the first 24 hours of their unit stay
- `h1_albumin_max`: 	The lowest albumin concentration of the patient in their serum during the first hour of their unit stay
- `h1_albumin_min`: 	The lowest albumin concentration of the patient in their serum during the first hour of their unit stay
- `h1_bilirubin_max`: 	The highest bilirubin concentration of the patient in their serum or plasma during the first hour of their unit stay
- `h1_bilirubin_min`: 	The lowest bilirubin concentration of the patient in their serum or plasma during the first hour of their unit stay
- `h1_bun_max`: 	The highest blood urea nitrogen concentration of the patient in their serum or plasma during the first hour of their unit stay
- `h1_bun_min`: 	The lowest blood urea nitrogen concentration of the patient in their serum or plasma during the first hour of their unit stay
- `h1_calcium_max`: 	The highest calcium concentration of the patient in their serum during the first hour of their unit stay
- `h1_calcium_min`: 	The lowest calcium concentration of the patient in their serum during the first hour of their unit stay
- `h1_creatinine_max`: 	The highest creatinine concentration of the patient in their serum or plasma during the first hour of their unit stay
- `h1_creatinine_min`: 	The lowest creatinine concentration of the patient in their serum or plasma during the first hour of their unit stay
- `h1_glucose_max`: 	The highest glucose concentration of the patient in their serum or plasma during the first hour of their unit stay
- `h1_glucose_min`: 	The lowest glucose concentration of the patient in their serum or plasma during the first hour of their unit stay
- `h1_hco3_max`: 	The highest bicarbonate concentration for the patient in their serum or plasma during the first hour of their unit stay
- `h1_hco3_min`: 	The lowest bicarbonate concentration for the patient in their serum or plasma during the first hour of their unit stay
- `h1_hematocrit_max`: 	The highest volume proportion of red blood cells in a patient's blood during the first hour of their unit stay, expressed as a fraction
- `h1_hematocrit_min`: 	The lowest volume proportion of red blood cells in a patient's blood during the first hour of their unit stay, expressed as a fraction
- `h1_inr_max`: 	The highest international normalized ratio for the patient during the first hour of their unit stay
- `h1_inr_min`: 	The lowest international normalized ratio for the patient during the first hour of their unit stay
- `h1_lactate_max`: 	The highest lactate concentration for the patient in their serum or plasma during the first hour of their unit stay
- `h1_lactate_min`: 	The lowest lactate concentration for the patient in their serum or plasma during the first hour of their unit stay
- `h1_sodium_max`: 	The highest sodium concentration for the patient in their serum or plasma during the first hour of their unit stay
- `h1_sodium_min`: 	The lowest sodium concentration for the patient in their serum or plasma during the first hour of their unit stay
- `d1_arterial_po2_max`: 	The highest arterial partial pressure of oxygen for the patient during the first 24 hours of their unit stay
- `d1_arterial_po2_min`: 	The lowest arterial partial pressure of oxygen for the patient during the first 24 hours of their unit stay
- `d1_pao2fio2ratio_max`: 	The highest fraction of inspired oxygen for the patient during the first 24 hours of their unit stay
- `d1_pao2fio2ratio_min`: 	The lowest fraction of inspired oxygen for the patient during the first 24 hours of their unit stay
- `h1_arterial_pco2_max`: 	The highest arterial partial pressure of carbon dioxide for the patient during the first hour of their unit stay
- `h1_arterial_pco2_min`: 	The lowest arterial partial pressure of carbon dioxide for the patient during the first hour of their unit stay
- `h1_arterial_ph_max`: 	The highest arterial pH for the patient during the first hour of their unit stay
- `h1_arterial_ph_min`: 	The lowest arterial pH for the patient during the first hour of their unit stay
- `h1_arterial_po2_max`: 	The highest arterial partial pressure of oxygen for the patient during the first hour of their unit stay
- `h1_arterial_po2_min`: 	The lowest arterial partial pressure of oxygen for the patient during the first hour of their unit stay
- `h1_pao2fio2ratio_max`: 	The highest fraction of inspired oxygen for the patient during the first hour of their unit stay
- `h1_pao2fio2ratio_min`: 	The lowest fraction of inspired oxygen for the patient during the first hour of their unit stay
- `aids`: 	Whether the patient has a definitive diagnosis of acquired immune deficiency syndrome (AIDS) (not HIV positive alone)
- `cirrhosis`: 	Whether the patient has a history of heavy alcohol use with portal hypertension and varices, other causes of cirrhosis with evidence of portal hypertension and varices, or biopsy proven cirrhosis. This comorbidity does not apply to patients with a functioning liver transplant.
- `hepatic_failure`: 	Whether the patient has cirrhosis and additional complications including jaundice and ascites, upper GI bleeding, hepatic encephalopathy, or coma.
- `immunosuppression`: 	Whether the patient has their immune system suppressed within six months prior to ICU admission for any of the following reasons; radiation therapy, chemotherapy, use of non-cytotoxic immunosuppressive drugs, high dose steroids (at least 0.3 mg/kg/day of methylprednisolone or equivalent for at least 6 months).
- `leukemia`: 	Whether the patient has been diagnosed with acute or chronic myelogenous leukemia, acute or chronic lymphocytic leukemia, or multiple myeloma.
- `lymphoma`: 	Whether the patient has been diagnosed with non-Hodgkin lymphoma.
- `solid_tumor_with_metastasis`: 	Whether the patient has been diagnosed with any solid tumor carcinoma (including malignant melanoma) which has evidence of metastasis.


**Target Variable**
- `diabetes-mellitus`: 	Whether the patient has been diagnosed with diabetes mellitus, a chronic disease (1, 0)
