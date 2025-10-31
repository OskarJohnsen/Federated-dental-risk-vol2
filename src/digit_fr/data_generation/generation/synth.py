from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from ..rules.decision.extraction import decide_row
from ..rules.decision.removal import compute_removal_decision
from ..rules.decision.risk import compute_risk_from_evidence
from ..rules.noise.apply import add_feature_noise

def _get_profile(client_profiles: Dict[int, Any], client_id: int) -> Dict[str, Any]:
    return client_profiles.get(client_id, {"name": f"Clinic_{client_id}", "prevalence_shift": {}, "score_scale": {1: 1, 2: 1, 3: 1, 4: 1}, "missingness": {}})

def _generate_binary(n: int, p: float) -> np.ndarray:
    return np.random.choice([0, 1], size=n, p=[1 - p, p])

def _generate_categorical(n: int, categories, probs) -> np.ndarray:
    return np.random.choice(categories, size=n, p=probs)

def _generate_age(n: int, mu: int = 28, sigma: int = 7, lo: int = 16, hi: int = 60) -> np.ndarray:
    return np.clip(np.random.normal(loc=mu, scale=sigma, size=n).astype(int), lo, hi)

# moved otseo and bisphos outside of loop
def prob_osteoporosis_given_age_sex(age: int, sex: int) -> float:
            if sex == 0:
                if age < 40: return 0.002
                if age < 50: return 0.010
                return 0.030
            else:
                if age < 40: return 0.005
                if age < 50: return 0.020
                return 0.070

def prob_bisphosphonates_given_age_osteoporosis(age: int, has_osteoporosis: bool) -> float:
            if has_osteoporosis:
                base = 0.45
                if age >= 50: base += 0.10
                return min(base, 0.70)
            else:
                return 0.003

def generate_dataset(configs: Dict[str, Any]) -> pd.DataFrame:
    gen = configs["generation"]
    extraction_cfg = configs["extraction_types"]
    binary_cfg = configs["extraction_binary"]
    client_profiles_raw = configs["client_profiles"]

    # Normalize client profile keys to int and score_scale keys to int
    client_profiles: Dict[int, Any] = {}
    for cid, prof in client_profiles_raw.items():
        p = dict(prof)
        if "score_scale" in p:
            p["score_scale"] = {int(k): v for k, v in p["score_scale"].items()}
        client_profiles[int(cid)] = p

    n_clients = gen["dataset"]["n_clients"]
    patients_per_client = gen["dataset"]["patients_per_client"]

    rows = []
    for c in range(1, n_clients + 1):
        prof = _get_profile(client_profiles, c)
        n = patients_per_client

        Age_mu = prof["prevalence_shift"].get("Age_mu", 28)
        Proximity_p = prof["prevalence_shift"].get("Proximity_Nerve_p", 0.60)
        Depth_probs = prof["prevalence_shift"].get("Impaction_Depth", [0.45, 0.45, 0.1])

        age = _generate_age(n, mu=Age_mu)
        sex = _generate_binary(n, 0.5)
        mandibular_maxillary = _generate_binary(n, 0.48)

        pain = _generate_binary(n, 0.5)
        swelling = _generate_binary(n, 0.30)
        trismus = _generate_binary(n, 0.20)
        pericoronitis = _generate_binary(n, 0.40)

        caries_w = _generate_binary(n, 0.20)
        caries_adj = _generate_binary(n, 0.15)
        perio = _generate_categorical(n, [1, 2, 3], [0.6, 0.3, 0.1])
        root_dev = _generate_categorical(n, [1, 2, 3], [0.2, 0.7, 0.1])
        mobility = _generate_categorical(n, [0, 1, 2], [0.7, 0.2, 0.1])

        ang = _generate_categorical(n, [1, 2, 3, 4, 5], [0.30, 0.40, 0.20, 0.09, 0.01])
        depth = _generate_categorical(n, [1, 2, 3], Depth_probs)

        gingival_cov = np.full(n, np.nan)
        mask_depth2 = (depth == 2)
        gingival_cov[mask_depth2] = _generate_binary(mask_depth2.sum(), p=0.60)

        prox_nerve = _generate_binary(n, Proximity_p)
        root_count = _generate_categorical(n, [1, 2, 3, 4], [0.1, 0.5, 0.3, 0.1])
        root_curve = _generate_binary(n, 0.70)
        bone_density = _generate_categorical(n, [1, 2, 3], [0.4, 0.4, 0.2])

        cyst = _generate_binary(n, 0.10)

        diabetes = _generate_binary(n, 0.10)
        clotting = _generate_binary(n, 0.02)
        smoking = _generate_binary(n, 0.25)
        prev_issue = _generate_binary(n, 0.10)

        osteoporosis = np.array([
            np.random.rand() < prob_osteoporosis_given_age_sex(int(age[i]), int(sex[i])) 
            for i in range(n)
        ], dtype=int)

        bisphosph = np.array([
            np.random.rand() < prob_bisphosphonates_given_age_osteoporosis(int(age[i]), bool(osteoporosis[i])) 
            for i in range(n)
        ], dtype=int)

        for i in range(n):
            rows.append({
                "Client": c, "Patient": i + 1,
                "Age": age[i], "Sex": sex[i], "Mandi_Maxi": mandibular_maxillary[i],
                "Pain": pain[i], "Swelling": swelling[i], "Trismus": trismus[i], "Pericoronitis": pericoronitis[i],
                "Caries_Wisdom": caries_w[i], "Caries_Adjacent": caries_adj[i],
                "Periodontal_Status": perio[i], "Root_Development": root_dev[i], "Tooth_Mobility": mobility[i],
                "Tooth_Angulation": ang[i], "Impaction_Depth": depth[i], "PartialBony_GingivalCoverage": (None if np.isnan(gingival_cov[i]) else int(gingival_cov[i])),
                "Proximity_Nerve": prox_nerve[i],
                "Root_Count": root_count[i], "Root_Curvature": root_curve[i], "Bone_Density": bone_density[i],
                "Cyst": cyst[i],
                "Diabetes": diabetes[i], "Osteoporosis": osteoporosis[i], "Clotting_Disorder": clotting[i],
                "Smoking": smoking[i], "Bisphosphonates": bisphosph[i],
                "Prev_Extraction_Issue": prev_issue[i],
            })

    df = pd.DataFrame(rows)

    # Optional missingness per profile
    for c in range(1, n_clients + 1):
        miss = _get_profile(client_profiles, c)["missingness"]
        idx = df["Client"] == c
        for col, rate in miss.items():
            mask = (np.random.rand(idx.sum()) < rate)
            df.loc[idx, col] = df.loc[idx, col].mask(mask)

    # Apply feature noise before computing risks
    for c in range(1, n_clients + 1):
        df = add_feature_noise(df, c, client_profiles, configs["noise"])

    # Decisions, removal, risks (seed is controlled by caller/CLI)
    temperature = gen["decision_model"]["temperature"]
    noise_sd = gen["decision_model"]["noise_sd"]

    decisions, score1, score2, score3, p1, p2, p3, subtypes = [], [], [], [], [], [], [], []
    removal_decisions, removal_probs = [], []
    alveolar_risks, infection_risks, nerve_risks, bleeding_risks = [], [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing decisions and risks"):
        d, s, p = decide_row(row, extraction_cfg, client_profiles, temperature, noise_sd)
        decisions.append(d)
        score1.append(s[1]); score2.append(s[2]); score3.append(s[3])
        p1.append(p[0]); p2.append(p[1]); p3.append(p[2])

        if d == 2:
            depth = row["Impaction_Depth"]
            if depth == 1: subtypes.append(1)
            elif depth == 2: subtypes.append(2)
            else: subtypes.append(3)
        else:
            subtypes.append(None)

        removal_decision, removal_prob = compute_removal_decision(row, binary_cfg)
        removal_decisions.append(removal_decision)
        removal_probs.append(removal_prob)

        alveolar_risk = compute_risk_from_evidence(row, "AlveolarOsteitis", configs["risks"], d)
        infection_risk = compute_risk_from_evidence(row, "SecondaryInfection", configs["risks"], d)
        nerve_risk = compute_risk_from_evidence(row, "NerveDysesthesia", configs["risks"], d)
        bleeding_risk = compute_risk_from_evidence(row, "Bleeding", configs["risks"], d)

        alveolar_risks.append(alveolar_risk)
        infection_risks.append(infection_risk)
        nerve_risks.append(nerve_risk)
        bleeding_risks.append(bleeding_risk)

    df["Surgical_Extraction_Type"] = decisions
    df["Score_1"] = score1; df["Score_2"] = score2; df["Score_3"] = score3
    df["Prob_1"] = p1; df["Prob_2"] = p2; df["Prob_3"] = p3
    df["Surg_2_Subtype"] = subtypes

    df["Removal_Indicated"] = removal_decisions
    df["Removal_Prob"] = removal_probs
    df["Risk_AlveolarOsteitis"] = alveolar_risks
    df["Risk_SecondaryInfection"] = infection_risks
    df["Risk_NerveDysesthesia"] = nerve_risks
    df["Risk_Bleeding"] = bleeding_risks

    return df