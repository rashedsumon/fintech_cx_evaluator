import os
from datasets import Dataset, load_dataset, DatasetDict

DATASET_NAME = "financial_cx_spain_mystery_shopping"

def get_fallback_dataset() -> DatasetDict:
    """Generates synthetic dataset for Spanish Fintech CX evaluation if remote dataset is unavailable."""
    sample_data = {
        "text": [
            "Intenté verificar mi DNI cargando una foto. El sistema dio error 'Documento borroso' 4 veces consecutivas sin indicar qué parte fallaba.",
            "El proceso de alta pidiéndome el NIE fue rápido, pero al transferir 10 EUR la app se congeló y tardó 15 minutos en reflejar el saldo.",
            "Contacté al soporte por chat sobre las comisiones de cambio EUR/USD. El agente tardó 22 minutos en responder y dio información engañosa.",
            "Completé la prueba de vida facial sin problemas. La interfaz en español es clara y la verificación KYC tardó menos de 2 minutos."
        ],
        "category": [
            "KYC_Bottleneck",
            "UX_Friction",
            "Service_Quality_Gap",
            "Compliant_Success"
        ],
        "severity": [3, 2, 3, 0]
    }
    raw_dataset = Dataset.from_dict(sample_data)
    return DatasetDict({"train": raw_dataset, "test": raw_dataset})

def load_or_download_dataset(hf_dataset_path: str = None) -> DatasetDict:
    """Auto-downloads dataset from Hugging Face Hub or builds local baseline data."""
    if hf_dataset_path:
        try:
            print(f"Downloading dataset '{hf_dataset_path}' from Hugging Face...")
            return load_dataset(hf_dataset_path)
        except Exception as e:
            print(f"Failed to fetch remote dataset: {e}. Falling back to baseline loader.")
    
    print("Loading baseline Spanish Mystery Shopping CX evaluation dataset...")
    return get_fallback_dataset()

if __name__ == "__main__":
    ds = load_or_download_dataset()
    print("Dataset successfully loaded:")
    print(ds)