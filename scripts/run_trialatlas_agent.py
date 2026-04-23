import sys
import os
from datetime import datetime

REPO_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_BASE_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(REPO_BASE_DIR, ".env"))

from biodsa.agents import TrialAtlasAgent

agent = TrialAtlasAgent(
    model_name="gpt-5",
    api_type="azure",
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    model_kwargs={
        "max_completion_tokens": 5000,
        "reasoning_effort": "minimal",
    },
    small_model_name="gpt-5",
    small_model_kwargs={
        "max_completion_tokens": 5000,
        "reasoning_effort": "minimal",
    },
    small_model_api_type="azure",
    small_model_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    small_model_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
)
agent.register_workspace()

execution_results = agent.go("""
Assess the design of the following trial:

```
Title:
  Clinical Trial to Investigate Efficacy and Safety of the IMP in Patients
  With Non Healing Wounds Originating From Ulcers

## Description Module
Brief Summary:
  The aim of this clinical trial is to investigate the efficacy (by
  monitoring the wound size reduction) and safety (by monitoring occurring
  adverse events) of the investigational medicinal product APZ2 after one
  single application on chronic venous leg ulcer wounds.
## Conditions Module
Conditions: Skin Ulcer Venous Stasis Chronic
Keywords:
- phase I/IIa
- mesenchymal stem cells
- ABCB5+
- APZ2
- chronic venous ulcer
- varicose ulcer
- skin ulcer
- advanced therapy medicinal product
- APZ2-II-01
- autologous use
- somatic cell therapy
## Design Module
Study Type: INTERVENTIONAL
### Design Info
Allocation: NA
Intervention Model: SINGLE_GROUP
Primary Purpose: TREATMENT
#### Masking Info
Masking: NONE
### Enrollment Info
Count: 13
Phases: PHASE1, PHASE2
## Arms Interventions Module
### Arm Groups
Label: APZ2 application
Type: EXPERIMENTAL
Description: Topical, single application of APZ2; 500000 cells per square cm;
Intervention Names: Biological: APZ2 application
### Interventions
Type: BIOLOGICAL
Name: APZ2 application
Description: Application of IMP on patients wound.
Arm Group Labels: APZ2 application
## Eligibility Module
Eligibility Criteria:
  Inclusion Criteria:
  1. Male or female patients aged 18 to 85 years;
  2. Chronic venous leg ulcer (as defined by the current AWMF guidelines:
  therapy resistant ulcer that shows no improvement within 3 months
  despite of optimal phlebological therapies or is not healed within 12
  months) for at least 6 weeks but shorter than 3 years diagnosed by
  doppler ultrasonography (DUS), ankle brachial index (ABI, 0.9-1.3),
  physical examination and dermatological review;
  3. Wound size between 5 and 50 square cm measured by a standardized
  photography at the screening visit;
  4. Wound location between knee and ankle;
  5. Patients suffering from 2 ulcers at the same extremity, as long as
  these ulcers are separated by a minimum bridge of 1 cm of epithelialized
  skin;
  6. Patients must agree to have at least one biopsy performed before
  treatment. In case IMP production from the first biopsy is not
  successful, a second biopsy will be taken;
  7. Body mass index (BMI) between 20 and 40 kg/m²;
  8. Patients understand the nature of the procedure and are providing
  written informed consent prior to any clinical trial procedure;
  9. Women of childbearing potential must have a negative blood pregnancy
  test at Screening and at Visit 5 before the IMP application;
  10. Women of childbearing potential and their partner must be willing to
  use highly effective contraceptive methods during the course of the
  clinical trial.
  Exclusion Criteria:
  General exclusion criteria
  1. Evidence of the ulcer extending to the underlying muscle, tendon, or
  bone;
  2. Current long-term use (more than 14 days) of steroid medication above
  Cushing-threshold dose (>7.5 mg/d prednisone or equivalent);
  3. Diabetes mellitus that has to be evaluated by blood test (Hemoglobin
  A1c [HbA1c] > 7.5%);
  4. Peripheral Artery Disease (PAD) including claudication with need of
  treatment;
  5. Acute deep vein thrombosis (maximum 30 days from diagnosis) or a
  still untrealed deep vein thrombosis;
  6. Unable to tolerate leg ulcer compression bandage;
  7. Infection of the target ulcer requiring treatment as judged
  clinically;
  8. Wound size <1.5 cm² measured by a standardized photography at Visit
  5;
  9. Any chronic dermatological disorders diagnosed at the investigator's
  discretion;
  10. Skin disorders, unrelated to the ulcer, that are present adjacent to
  the target wound;
  11. Current use of medications that influence wound healing: systemic
  immunosuppressives, cytotoxic medicinal products, and systemic steroids
  (above Cushing-threshold level);
  12. Known abuse of alcohol, drugs, or medicinal products;
  13. Cancerous or pre-cancerous lesions adjacent to the target wound;
  14. Patients anticipated to be unwilling or unable to comply with the
  requirements of the protocol;
  15. Pregnant or lactating women;
  16. Systemic infectious disease diagnosed by serology testing for
  syphilis (acute), human immunodeficiency virus (HIV˗1, HIV-2), hepatitis
  B (acute) or C infection at Screening or at Visit 2;
  17. Any known allergies to components of the IMP;
  18. Prior surgical procedures such as bypass or mesh-graft treatment
  within 2 months prior to IMP application;
  19. Treatment with active wound care agents (e.g. Iruxol, local
  antibiotics or silver dressings), which have not been paused 14 days
  before IMP application;
  20. Current or previous (within 30 days of enrollment) treatment with
  another IMP, or participation and/or under follow-up in another clinical
  trial;
  21. Previous participation in this clinical trial;
  22. Evidence of any other medical conditions (such as psychiatric
  illness, physical examination, or laboratory findings) that may
  interfere with the planned treatment, affect the patient's compliance,
  or place the patient at high risk of complications related to the
  treatment;
  23. Employees of the sponsor, or employees or relatives of the
  investigator.
  Exclusion criteria for efficacy assessments
  1. A wound size enlargement of more than 25% between the wound
  assessment at the screening visit and the wound assessment at Visit 5;
  2. A wound size reduction of more than 50% between the wound assessment
  at the screening visit and wound assessment at Visit 5.
Healthy Volunteers: False
Sex: ALL
Minimum Age: 18 Years
Maximum Age: 85 Years
Std Ages: ADULT, OLDER_ADULT
```
""")

# Save outputs to outputs/<timestamp>/
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(REPO_BASE_DIR, "outputs", timestamp)
os.makedirs(output_dir, exist_ok=True)

print(execution_results)

# Save JSON
json_path = os.path.join(output_dir, "results.json")
execution_results.to_json(output_path=json_path)
print(f"JSON saved: {json_path}")

# Download sandbox artifacts
artifacts = execution_results.download_artifacts(output_dir=output_dir)
print(f"Artifacts downloaded: {artifacts}")

# Generate PDF report
pdf_path = execution_results.to_pdf(output_dir=output_dir)
print(f"PDF saved: {pdf_path}")

agent.clear_workspace()
print(f"\nAll outputs saved to: {output_dir}")
