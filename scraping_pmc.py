import json
import os
import re
from datetime import datetime
from xml.etree import ElementTree as ET

import requests

DISEASES = [
    "heart disease", "hypertension", "stroke", "arrhythmia", "angina",
    "myocardial infarction", "heart failure", "atherosclerosis", "aortic aneurysm",
    "peripheral arterial disease", "venous thromboembolism", "pulmonary hypertension",
    "endocarditis", "myocarditis", "pericarditis", "valvular heart disease",
    "congenital heart disease", "cardiomyopathy", "deep vein thrombosis", "pulmonary embolism",
    "asthma", "pneumonia", "chronic obstructive pulmonary disease", "bronchitis",
    "tuberculosis", "covid-19", "influenza", "cystic fibrosis", "emphysema",
    "acute respiratory distress syndrome", "pulmonary fibrosis", "mesothelioma",
    "sleep apnea", "laryngitis", "pharyngitis", "bronchiectasis", "atelectasis",
    "hypersensitivity pneumonitis", "sarcoidosis", "occupational lung disease",
    "aspergillosis", "blastomycosis", "histoplasmosis", "legionnaires disease",
    "diabetes", "type 2 diabetes", "type 1 diabetes", "obesity", "metabolic syndrome",
    "thyroid disease", "hyperthyroidism", "hypothyroidism", "thyroid cancer",
    "goiter", "graves disease", "hashimoto thyroiditis", "adrenal insufficiency",
    "cushing syndrome", "pituitary disorders", "growth hormone deficiency",
    "hypogonadism", "polycystic ovary syndrome", "preeclampsia", "gestational diabetes",
    "hyperlipidemia", "fatty liver disease", "pancreatitis", "diabetic ketoacidosis",
    "lung cancer", "breast cancer", "colorectal cancer", "prostate cancer",
    "pancreatic cancer", "leukemia", "lymphoma", "hodgkin lymphoma", "multiple myeloma",
    "melanoma", "ovarian cancer", "uterine cancer", "cervical cancer",
    "gastric cancer", "esophageal cancer", "liver cancer", "hepatocellular carcinoma",
    "bladder cancer", "renal cell carcinoma", "brain tumor", "glioblastoma",
    "sarcoma", "osteosarcoma", "alzheimer disease", "parkinson disease", "epilepsy",
    "multiple sclerosis", "amyotrophic lateral sclerosis", "migraine", "dementia",
    "vascular dementia", "lewy body dementia", "huntington disease", "essential tremor",
    "ataxia", "restless leg syndrome", "narcolepsy", "guillain-barre syndrome",
    "myasthenia gravis", "peripheral neuropathy", "transverse myelitis", "hiv",
    "malaria", "measles", "mumps", "rubella", "hepatitis a", "hepatitis b",
    "hepatitis c", "dengue", "zika virus", "ebola", "yellow fever", "west nile virus",
    "whooping cough", "diphtheria", "tetanus", "meningitis", "encephalitis",
    "poliomyelitis", "rabies", "lyme disease", "depression",
    "major depressive disorder", "anxiety disorder", "generalized anxiety disorder",
    "panic disorder", "bipolar disorder", "schizophrenia", "post-traumatic stress disorder",
    "autism spectrum disorder", "attention deficit hyperactivity disorder",
    "alcohol use disorder", "substance use disorder", "eating disorder",
    "anorexia nervosa", "bulimia nervosa", "rheumatoid arthritis", "lupus",
    "systemic lupus erythematosus", "crohn disease", "ulcerative colitis", "celiac disease",
    "psoriasis", "psoriatic arthritis", "ankylosing spondylitis", "scleroderma",
    "sjogren syndrome", "primary biliary cholangitis", "autoimmune hepatitis",
    "pemphigus vulgaris", "vitiligo", "gastroesophageal reflux disease",
    "peptic ulcer disease", "gastroparesis", "irritable bowel syndrome",
    "inflammatory bowel disease", "diverticulitis", "appendicitis", "cholecystitis",
    "cirrhosis", "hepatic encephalopathy", "gallstone", "hemorrhoid", "osteoarthritis",
    "gout", "fibromyalgia", "osteoporosis", "osteomyelitis", "septic arthritis",
    "bursitis", "tendinitis", "rotator cuff tear", "herniated disc", "sciatica",
    "spinal stenosis", "scoliosis", "eczema", "atopic dermatitis", "contact dermatitis",
    "rosacea", "acne", "folliculitis", "impetigo", "cellulitis", "herpes zoster",
    "alopecia areata", "lichen planus", "urticaria", "sickle cell disease",
    "thalassemia", "hemophilia", "von willebrand disease", "marfan syndrome",
    "muscular dystrophy", "spinal muscular atrophy", "fragile x syndrome",
    "down syndrome", "urinary tract infection", "prostatitis",
    "benign prostatic hyperplasia", "nephrolithiasis", "chronic kidney disease",
    "acute kidney injury", "glomerulonephritis", "endometriosis", "uterine fibroids",
    "pelvic inflammatory disease", "ovarian cyst", "vulvovaginal candidiasis",
    "bacterial vaginosis", "dysmenorrhea", "amenorrhea", "anemia",
    "iron deficiency anemia", "hemolytic anemia", "thrombocytopenia",
    "disseminated intravascular coagulation", "antiphospholipid syndrome",
    "cerebral palsy", "developmental delay", "childhood asthma", "croup",
    "bronchiolitis", "rotavirus", "chickenpox", "scarlet fever", "glaucoma",
    "cataracts", "age-related macular degeneration", "diabetic retinopathy",
    "retinal detachment", "dry eye syndrome", "hearing loss", "tinnitus", "vertigo",
    "meniere disease", "otitis media", "otitis externa",
]

DOWNLOAD_DIR = "datasets_pmc"
RETMAX = 50
REQUEST_HEADERS = {"User-Agent": "MedicalChatbotScraper/1.0"}
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def create_folder(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_article_ids(disease_name: str) -> list[str]:
    params = {
        "db": "pmc",
        "term": f'"{disease_name}"[Title]',
        "retmax": str(RETMAX),
        "retmode": "json",
    }

    try:
        response = requests.get(ESEARCH_URL, params=params, headers=REQUEST_HEADERS, timeout=30)
        response.raise_for_status()
    except requests.RequestException:
        return []

    return response.json().get("esearchresult", {}).get("idlist", [])


def get_article_record(article_id: str, disease_name: str) -> dict | None:
    params = {
        "db": "pmc",
        "id": article_id,
        "retmode": "xml",
    }

    try:
        response = requests.get(EFETCH_URL, params=params, headers=REQUEST_HEADERS, timeout=30)
        response.raise_for_status()
    except requests.RequestException:
        return None

    try:
        root = ET.fromstring(response.text)
    except ET.ParseError:
        return None

    article = root.find("article")
    if article is None:
        return None

    title_node = article.find(".//article-title")
    title = " ".join(title_node.itertext()).strip() if title_node is not None else article_id

    body = article.find("body")
    if body is not None:
        content = "\n".join(
            text.strip()
            for text in body.itertext()
            if text and text.strip()
        )
    else:
        content = ""

    if not content:
        return None

    return {
        "disease": disease_name,
        "title": title,
        "publisher_url": f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{article_id}/",
        "publisher_content": content,
        "publisher_status": "200",
        "scraped_at": datetime.now().isoformat(),
    }


def safe_filename(article_id: str) -> str:
    slug = re.sub(r"[^\w.-]", "_", article_id)
    return f"PMC{slug[:120]}.json"


def _print_progress(current: int, total: int) -> None:
    width = 30
    filled = int(width * current / total) if total else 0
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r    [{bar}] {current}/{total}", end="", flush=True)


def scrape_disease(disease_name: str) -> int:
    print(f"  * {disease_name}", end="", flush=True)
    create_folder(DOWNLOAD_DIR)
    article_ids = get_article_ids(disease_name)

    if not article_ids:
        print(" (no results)")
        return 0

    total_articles = len(article_ids)
    print(f" ({total_articles} results)")
    saved = 0

    for index, article_id in enumerate(article_ids, start=1):
        record = get_article_record(article_id, disease_name)
        if record:
            filepath = os.path.join(DOWNLOAD_DIR, safe_filename(article_id))
            with open(filepath, "w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2, ensure_ascii=False)
            saved += 1
        _print_progress(index, total_articles)

    print(f" -> {saved} saved")
    return saved


def main() -> None:
    print("\n" + "=" * 80)
    print("MEDICAL SCRAPER  --  PMC E-utilities".center(80))
    print("=" * 80)
    print(f"Diseases      : {len(DISEASES)}")
    print(f"Articles/disease: {RETMAX}")
    print("=" * 80)

    create_folder(DOWNLOAD_DIR)

    total = 0
    for index, disease in enumerate(DISEASES, start=1):
        print(f"\n[{index:3d}/{len(DISEASES)}]", end="")
        total += scrape_disease(disease)

    print(f"\n{'=' * 80}")
    print("COMPLETED".center(80))
    print(f"Total articles saved : {total}".center(80))
    print(f"Location : {os.path.abspath(DOWNLOAD_DIR)}".center(80))
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
