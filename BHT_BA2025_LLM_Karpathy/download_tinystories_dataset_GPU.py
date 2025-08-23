"""
Gezielter Download der ersten vier TinyStories JSON-Dateien
für wissenschaftliche Reproduzierbarkeit der Bachelorthesis

Angepasst und erstellt mit Unterstützung von Claude Opus 4
"""

import os
import json
import requests
from datasets import load_dataset
from tqdm import tqdm

def download_tinystories_subset(output_dir="test_test", max_files=4, max_entries_per_file=10000):
    """
    Lädt die ersten vier JSON-Dateien des TinyStories Datasets herunter
    
    Args:
        output_dir: Zielverzeichnis für JSON-Dateien
        max_files: Anzahl der zu erstellenden JSON-Dateien (4)
        max_entries_per_file: Maximale Einträge pro Datei
    """
    
    # Debug: Zeige aktuelles Arbeitsverzeichnis
    current_dir = os.getcwd()
    print(f"Aktuelles Arbeitsverzeichnis: {current_dir}")
    
    # Vollständiger Pfad zum Output-Verzeichnis
    full_output_path = os.path.abspath(output_dir)
    print(f"Erstelle Verzeichnis: {full_output_path}")
    
    # Erstelle Output-Verzeichnis
    os.makedirs(output_dir, exist_ok=True)
    
    # Überprüfe ob Verzeichnis erstellt wurde
    if os.path.exists(output_dir):
        print(f"✓ Verzeichnis '{output_dir}' erfolgreich erstellt")
    else:
        print(f"✗ Fehler: Verzeichnis '{output_dir}' konnte nicht erstellt werden")
        return 0
    
    print("Lade TinyStories Dataset von Hugging Face...")
    
    # Lade nur den Trainings-Split des Datasets
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    current_file = 0
    current_entries = 0
    current_data = []
    
    print(f"Erstelle {max_files} JSON-Dateien mit jeweils max. {max_entries_per_file} Einträgen...")
    
    # Iteriere durch das Dataset
    for i, item in enumerate(tqdm(dataset, desc="Verarbeite Einträge")):
        # Füge Story zum aktuellen Batch hinzu
        current_data.append({"story": item["text"]})
        current_entries += 1
        
        # Prüfe ob aktuelle Datei voll ist
        if current_entries >= max_entries_per_file:
            # Speichere aktuelle Datei
            filename = os.path.join(output_dir, f"data{current_file:02d}.json")
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(current_data, f, ensure_ascii=False, indent=2)
            
            print(f"Gespeichert: {filename} ({len(current_data)} Einträge)")
            
            # Reset für nächste Datei
            current_file += 1
            current_entries = 0
            current_data = []
            
            # Stoppe wenn genug Dateien erstellt wurden
            if current_file >= max_files:
                break
    
    # Speichere letzte Datei falls nicht leer
    if current_data and current_file < max_files:
        filename = os.path.join(output_dir, f"data{current_file:02d}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(current_data, f, ensure_ascii=False, indent=2)
        print(f"Gespeichert: {filename} ({len(current_data)} Einträge)")
    
    print(f"\nDownload abgeschlossen! {max_files} JSON-Dateien erstellt in '{output_dir}'")
    
    # Validierung der erstellten Dateien
    total_entries = 0
    for i in range(max_files):
        filename = os.path.join(output_dir, f"data{i:02d}.json")
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
                total_entries += len(data)
                print(f"data{i:02d}.json: {len(data)} Einträge")
    
    print(f"Gesamt: {total_entries} Einträge")
    return total_entries

def verify_compatibility():
    """
    Überprüft ob die erstellten Dateien mit dem bestehenden Script kompatibel sind
    """
    test_dir = "test_test"
    if not os.path.exists(test_dir):
        print("Fehler: Verzeichnis 'test_test' nicht gefunden")
        return False
    
    # Prüfe ob alle vier Dateien existieren
    required_files = [f"data{i:02d}.json" for i in range(4)]
    existing_files = []
    
    for filename in required_files:
        filepath = os.path.join(test_dir, filename)
        if os.path.exists(filepath):
            existing_files.append(filename)
            # Prüfe Dateistruktur
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    if isinstance(data[0], dict) and "story" in data[0]:
                        print(f"✓ {filename}: Gültige Struktur ({len(data)} Einträge)")
                    else:
                        print(f"✗ {filename}: Ungültige Struktur")
                        return False
                else:
                    print(f"✗ {filename}: Leere oder ungültige Datei")
                    return False
    
    if len(existing_files) == 4:
        print("✓ Alle vier JSON-Dateien sind vorhanden und kompatibel")
        return True
    else:
        print(f"✗ Nur {len(existing_files)}/4 Dateien gefunden: {existing_files}")
        return False

def main():
    """
    Hauptfunktion für wissenschaftlich reproduzierbaren Download
    """
    print("TinyStories Subset Download für Bachelorthesis")
    print("=" * 50)
    
    # Konfiguration für reproduzierbare Ergebnisse
    config = {
        "output_dir": "test_test",
        "max_files": 4,
        "max_entries_per_file": 10000,  # Anpassbar je nach Bedarf
        "seed": 123  # Für Reproduzierbarkeit
    }
    
    print("Konfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Download durchführen
    try:
        total_entries = download_tinystories_subset(
            output_dir=config["output_dir"],
            max_files=config["max_files"],
            max_entries_per_file=config["max_entries_per_file"]
        )
        
        print("\nVerifiziere Kompatibilität...")
        if verify_compatibility():
            print("\n✓ Download erfolgreich! Die Dateien sind bereit für c_convert_TinyStories_npy_gleicherSatz.py")
        else:
            print("\n✗ Kompatibilitätsprüfung fehlgeschlagen")
            
    except Exception as e:
        print(f"\nFehler beim Download: {e}")
        print("Stelle sicher, dass du eine Internetverbindung hast und die 'datasets' Bibliothek installiert ist:")
        print("pip install datasets tqdm")

if __name__ == "__main__":
    main()