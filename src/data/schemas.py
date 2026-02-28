BOLD_REQUIRED_COLUMNS = ["sao2", "spo2"]

ENCODE_TABLES = [
    "PERSON",
    "VISIT_OCCURRENCE",
    "MEASUREMENT",
    "CONCEPT",
]

OMOP_KEY_COLUMNS = {
    "PERSON": ["person_id"],
    "VISIT_OCCURRENCE": ["visit_occurrence_id", "person_id"],
    "MEASUREMENT": ["measurement_id", "person_id"],
    "CONCEPT": ["concept_id", "concept_name"],
}
