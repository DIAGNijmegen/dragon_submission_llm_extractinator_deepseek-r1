import pandas as pd
from dragon_baseline import DragonBaseline


def load_predictions(task_name, is_synthetic, expected_len):
    if is_synthetic:
        path = f"/opt/app/results/synthetic/{task_name}/nlp-predictions-dataset.json"
        print(f"[Synthetic] Loading predictions from: {path}")
        return pd.read_json(path)
    else:
        val_path = (
            f"/opt/app/results/validation/{task_name}/nlp-predictions-dataset.json"
        )
        test_path = f"/opt/app/results/test/{task_name}/nlp-predictions-dataset.json"
        print(f"[Normal] Trying validation: {val_path}")
        print(f"[Normal] Trying test: {test_path}")

        val_predictions = pd.read_json(val_path)
        test_predictions = pd.read_json(test_path)

        # Choose the predictions matching the expected length
        if len(val_predictions) == expected_len:
            print(f"[Normal] Using validation predictions")
            return val_predictions
        elif len(test_predictions) == expected_len:
            print(f"[Normal] Using test predictions")
            return test_predictions
        else:
            raise ValueError(
                f"Prediction count mismatch. Expected {expected_len}, "
                f"got {len(val_predictions)} (val) and {len(test_predictions)} (test)"
            )


def sort_predictions(df):
    df["case_num"] = df["uid"].str.extract(r"case(\d+)").astype(int)
    df = df.sort_values("case_num").reset_index(drop=True)
    df = df.drop(columns="case_num")
    return df


if __name__ == "__main__":
    baseline = DragonBaseline()
    baseline.load()

    taskname = baseline.task.task_name

    synthetic_tasklist = [
        "Task101_Example_sl_bin_clf-fold0",
        "Task102_Example_sl_mc_clf-fold0",
        "Task103_Example_mednli-fold0",
        "Task104_Example_ml_bin_clf-fold0",
        "Task105_Example_ml_mc_clf-fold0",
        "Task106_Example_sl_reg-fold0",
        "Task107_Example_ml_reg-fold0",
        "Task108_Example_sl_ner-fold0",
        "Task109_Example_ml_ner-fold0",
    ]

    normal_tasklist = [
        "Task001_adhesion_clf-fold0",
        "Task002_nodule_clf-fold0",
        "Task003_kidney_clf-fold0",
        "Task004_skin_case_selection_clf-fold0",
        "Task005_recist_timeline_clf-fold0",
        "Task006_pathology_tumor_origin_clf-fold0",
        "Task007_nodule_diameter_presence_clf-fold0",
        "Task008_pdac_size_presence_clf-fold0",
        "Task009_pdac_diagnosis_clf-fold0",
        "Task010_prostate_radiology_clf-fold0",
        "Task011_prostate_pathology_clf-fold0",
        "Task012_pathology_tissue_type_clf-fold0",
        "Task013_pathology_tissue_origin_clf-fold0",
        "Task014_textual_entailment_clf-fold0",
        "Task015_colon_pathology_clf-fold0",
        "Task016_recist_lesion_size_presence_clf-fold0",
        "Task017_pdac_attributes_clf-fold0",
        "Task018_osteoarthritis_clf-fold0",
        "Task019_prostate_volume_reg-fold0",
        "Task020_psa_reg-fold0",
        "Task021_psad_reg-fold0",
        "Task022_pdac_size_reg-fold0",
        "Task023_nodule_diameter_reg-fold0",
        "Task024_recist_lesion_size_reg-fold0",
        "Task025_anonymisation_ner-fold0",
        "Task026_medical_terminology_ner-fold0",
        "Task027_prostate_biopsy_ner-fold0",
        "Task028_skin_pathology_ner-fold0",
    ]

    all_tasks = synthetic_tasklist + normal_tasklist

    # Find the task name corresponding to the jobid
    task_name = None
    for task in all_tasks:
        if task.startswith(f"{taskname}"):
            task_name = task
            break

    if task_name is None:
        raise ValueError(f"No task found for {taskname}")

    is_synthetic = task_name in synthetic_tasklist
    expected_len = len(baseline.df_test)
    predictions = load_predictions(task_name, is_synthetic, expected_len)
    predictions = sort_predictions(predictions)

    print(f"{task_name}: Loaded {len(predictions)} predictions. Saving...")

    baseline.save(predictions)
    baseline.verify_predictions()
