from datasets import load_dataset
from dataset.map_fn import sample_mapping_fn
from dataset.info_cons_format_map_fn import info_constrained_sample_mapping_fn, load_multi_format_dataset
import os


def load_hf_dataset(data_files: dict, 
                    llm_tokenizer, 
                    input_max_length, 
                    data_split:str,
                    pe_method: str = "raw", 
                    num_workers=None,
                    task: str = None):
    instruction = "" # empty instruction by default.
    
    # Task-specific prompts
    if task == "guo_readmission":
        task_prompt = """
        Predict 30-day readmission after discharge using all available EHR data at and 
        prior to discharge. Anchor your reasoning to validated risk domains (e.g., LACE 
        and HOSPITAL variables) and transitional-care best practices. Specifically, 
        consider: length of stay; acuity of admission; comorbidity burden (e.g., Charlson/ICD history); 
        prior ED visits/admissions in the last 6–12 months; discharge service; key discharge-day labs 
        (e.g., hemoglobin, sodium, creatinine), vital-sign stability, procedures during index stay; 
        high-risk medications and regimen changes (anticoagulants/insulin/opioids), polypharmacy; 
        diagnoses linked to higher readmission (HF, COPD, CKD, diabetes, cancer); 
        post-discharge plan quality (follow-up booked ≤7 days, pending results communication, home health); 
        and documented social needs (transport, housing, caregiver support, health literacy). 
        Do not use race/ethnicity or other protected characteristics as proxies. 
        Output: output JSON only: {"answer": "Your Answer”}.
        """
        instruction += task_prompt
    elif task == "new_pancan":
        task_prompt = """
        Predict the probability of a new primary cancer diagnosis within 1 year. 
        Ground your assessment in symptom, sign, and test patterns consistent with referral thresholds 
        from national suspected-cancer guidance (e.g., NG12). Consider: age/sex; alarm symptoms 
        (unexplained weight loss, persistent unexplained pain, dysphagia, rectal/uterine bleeding, 
        postmenopausal bleeding, persistent cough/hoarseness, lymphadenopathy); 
        lab flags (iron-deficiency anaemia in men/postmenopausal women, thrombocytosis, hypercalcaemia, 
        abnormal LFTs without clear cause, age-adjusted PSA, positive FIT); 
        imaging/pathology cues (suspicious lesions, unexplained masses or lytic lesions); 
        and documented high-risk history (strong family history, prior cancer) when present. 
        Avoid over-weighting non-specific tumour markers unless guideline-supported. 
        Output: output JSON only: {"answer": "Your Answer”}.
        """
        instruction += task_prompt
    elif task == "icu_mortality":
        task_prompt = """
        Predict in-ICU mortality for the current admission using time-updated physiology and organ-support data. 
        Use validated severity constructs: baseline and delta SOFA (respiratory PaO2/FiO2, coagulation platelets, 
        liver bilirubin, cardiovascular MAP/vasopressors, CNS GCS, renal creatinine/urine output); 
        APACHE-style admission severity (first-24h worst values, age, chronic health); 
        lactate trends; invasive/non-invasive ventilation with oxygenation indices; ARDS status and severity; 
        hemodynamics (vasopressor dose, fluid balance); renal replacement therapy; infection status and septic shock; 
        organ supports (ECMO/IABP), comorbidities, frailty/code status if documented. 
        Output: output JSON only: {"answer": "Your Answer”}.
        """
        instruction += task_prompt
    elif task == "icu_phenotyping":
        task_prompt = """
        Identify which of the 25 predefined ICU phenotypes apply by mapping EHR evidence to standard diagnostic 
        criteria where available. Use: Sepsis-3 (sepsis = infection + ↑SOFA ≥2; septic shock = vasopressors to 
        maintain MAP ≥65 mmHg AND lactate >2 mmol/L despite fluids); ARDS (Berlin/updated definitions: timing within 
        1 week, bilateral opacities not fully explained by cardiac failure/overload, and PaO2/FiO2 with PEEP≥5 to 
        grade severity); AKI (KDIGO: rise in creatinine ≥0.3 mg/dL in 48h or ≥1.5× baseline within 7 days, or 
        urine output <0.5 mL/kg/h for ≥6h; stage accordingly); pneumonia (clinical features plus new infiltrate on 
        imaging consistent with CAP/HAP when applicable); myocardial infarction (Universal Definition: acute myocardial 
        injury with troponin rise/fall AND evidence of ischaemia—symptoms, ECG, imaging, or angiography); heart failure 
        (documented HF with echo/NP support; classify HFrEF/HFmrEF/HFpEF when data allow); COPD/exacerbation 
        (history of COPD with post-BD FEV1/FVC<0.7; exacerbation = acute symptom worsening requiring treatment change). 
        For remaining phenotypes, combine guideline-consistent rules with ICD codes, procedures, cultures, imaging, 
        and clinician notes. 
        Output: output JSON only: {"answer": "Your Answer”}.
        """
        instruction += task_prompt

    
    # PE method prompts
    if pe_method == "raw":
        pass
    elif pe_method == "eb": # evidence based
        instruction += " Generate the answer with evidence and explanation."
    elif pe_method == "cot":
        instruction += " Think step by step."
    elif pe_method == "coe":
        instruction += " Think step-by-step and generate the answer with evidence and explanation."
    else:
        raise ValueError(f"Invalid pe_method: {pe_method}")
    
    dataset = load_dataset(
        "parquet",
        data_files=data_files,
        columns=['label', "data", "question"],
        cache_dir=os.path.dirname(list(data_files.values())[0],),
        split=data_split
    )

    dataset = dataset.map(
        sample_mapping_fn,
        batched=True, 
        fn_kwargs={
            "instruction": instruction,
            "llm_tokenizer": llm_tokenizer,
            "max_input_length": input_max_length,
        },
        num_proc=num_workers,
        remove_columns=['data', 'question']  # Remove unnecessary columns
    )
    return dataset


def load_info_constrained_hf_dataset(data_files: dict, llm_tokenizer, input_max_length,
                                    pe_method: str = "raw", 
                                    data_format: str = "nl",
                                    num_workers=None,
                                    task: str = None):
    instruction = "Now make your prediction."
    
    # Task-specific prompts
    if task == "guo_readmission":
        task_prompt = """
        Based on the patient's EHR data, predict whether this patient will be readmitted to the hospital within 30 days after discharge. 
        Consider factors such as comorbidities, medication adherence, discharge planning, and social determinants of health.
        """
        instruction += task_prompt
    elif task == "icu_mortality":
        task_prompt = """
        Based on the patient's EHR data, predict whether this patient will die during this ICU stay. 
        Consider vital signs, lab values, interventions, and the overall clinical trajectory.
        """
        instruction += task_prompt
    elif task == "icu_phenotyping":
        task_prompt = """
        Based on the patient's EHR data, identify which medical conditions this patient has from the available phenotype options. 
        Consider all relevant clinical indicators, lab values, and documented diagnoses.
        """
        instruction += task_prompt
    elif task == "new_pancan":
        task_prompt = """
        Based on the patient's EHR data, predict whether this patient has cancer. 
        Consider all relevant clinical indicators, lab values, imaging results, and documented symptoms.
        """
        instruction += task_prompt
    
    # PE method prompts
    if pe_method == "raw":
        pass
    elif pe_method == "cot":
        instruction += " Let's think step by step."
    else:
        raise ValueError(f"Invalid pe_method: {pe_method}")
    
    dataset = load_multi_format_dataset(data_files=data_files)

    dataset = dataset.map(
        info_constrained_sample_mapping_fn,
        fn_kwargs={
            "instruction": instruction,
            "llm_tokenizer": llm_tokenizer,
            "max_input_length": input_max_length,
            "data_format": data_format
        },
        num_proc=num_workers
    )
    return dataset

