import json
import subprocess
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from pathlib import Path
import sys

def extract_radon_metrics(path):
    """Extrahiert Radon-Metriken für einen gegebenen Pfad"""
    try:
        print(f"Führe radon-Analyse für Pfad aus: {path}")
        
        # Komplexitätsmetriken extrahieren
        print("Extrahiere Komplexitätsmetriken...")
        cc_result = subprocess.run(['radon', 'cc', path, '--json'], 
                                  capture_output=True, text=True, check=True)
        
        if cc_result.stdout.strip():
            complexity_data = json.loads(cc_result.stdout)
        else:
            print("Warnung: Keine Komplexitätsdaten erhalten")
            complexity_data = {}
        
        # Maintainability Index
        print("Extrahiere Maintainability Index...")
        mi_result = subprocess.run(['radon', 'mi', path, '--json'], 
                                  capture_output=True, text=True, check=True)
        
        if mi_result.stdout.strip():
            mi_data = json.loads(mi_result.stdout)
        else:
            print("Warnung: Keine MI-Daten erhalten")
            mi_data = {}
        
        # Halstead-Metriken (optional)
        print("Extrahiere Halstead-Metriken...")
        try:
            hal_result = subprocess.run(['radon', 'hal', path, '--json'], 
                                       capture_output=True, text=True, check=True)
            if hal_result.stdout.strip():
                halstead_data = json.loads(hal_result.stdout)
            else:
                halstead_data = {}
        except:
            print("Warnung: Halstead-Metriken konnten nicht extrahiert werden")
            halstead_data = {}
        
        print(f"Extrahierte Daten:")
        print(f"  Komplexität: {len(complexity_data)} Dateien")
        print(f"  Maintainability: {len(mi_data)} Dateien")
        print(f"  Halstead: {len(halstead_data)} Dateien")
        
        return complexity_data, mi_data, halstead_data
        
    except subprocess.CalledProcessError as e:
        print(f"Fehler beim Ausführen von radon für {path}: {e}")
        print(f"stdout: {e.stdout if hasattr(e, 'stdout') else 'N/A'}")
        print(f"stderr: {e.stderr if hasattr(e, 'stderr') else 'N/A'}")
        return {}, {}, {}
    except json.JSONDecodeError as e:
        print(f"Fehler beim Parsen der JSON-Ausgabe für {path}: {e}")
        print(f"Rohe Ausgabe: {cc_result.stdout if 'cc_result' in locals() else 'N/A'}")
        return {}, {}, {}
    except Exception as e:
        print(f"Unerwarteter Fehler: {e}")
        return {}, {}, {}

def process_complexity_data(complexity_data):
    """Verarbeitet Komplexitätsdaten zu DataFrame"""
    complexity_scores = []
    file_names = []
    
    print("Debug: Struktur der Komplexitätsdaten:")
    print(f"Type: {type(complexity_data)}")
    
    if not complexity_data:
        print("Warnung: Keine Komplexitätsdaten gefunden")
        return pd.DataFrame({'file': [], 'complexity': []})
    
    try:
        for file_path, file_data in complexity_data.items():
            print(f"Verarbeite Datei: {file_path}")
            print(f"Datentyp: {type(file_data)}")
            
            # Prüfen ob file_data eine Liste ist
            if isinstance(file_data, list):
                for item in file_data:
                    if isinstance(item, dict) and 'type' in item and 'complexity' in item:
                        if item['type'] in ['function', 'method', 'class']:
                            complexity_scores.append(item['complexity'])
                            file_names.append(Path(file_path).name)
            elif isinstance(file_data, dict):
                # Falls file_data direkt ein Dict ist
                if 'complexity' in file_data:
                    complexity_scores.append(file_data['complexity'])
                    file_names.append(Path(file_path).name)
            else:
                print(f"Unerwarteter Datentyp für {file_path}: {type(file_data)}")
    
    except Exception as e:
        print(f"Fehler bei der Datenverarbeitung: {e}")
        print("Rohdaten:")
        print(complexity_data)
    
    if not complexity_scores:
        print("Warnung: Keine gültigen Komplexitätswerte gefunden")
        # Dummy-Daten für Visualisierung
        return pd.DataFrame({'file': ['dummy'], 'complexity': [1]})
    
    return pd.DataFrame({
        'file': file_names,
        'complexity': complexity_scores
    })

def process_maintainability_data(mi_data):
    """Verarbeitet Maintainability-Daten zu DataFrame"""
    mi_scores = []
    file_names = []
    ranks = []
    
    print("Debug: Struktur der Maintainability-Daten:")
    print(f"Type: {type(mi_data)}")
    
    if not mi_data:
        print("Warnung: Keine Maintainability-Daten gefunden")
        return pd.DataFrame({'file': [], 'maintainability': [], 'rank': []})
    
    try:
        for file_path, mi_score in mi_data.items():
            print(f"Verarbeite Datei: {file_path}")
            
            # Prüfen ob mi_score ein Dictionary mit 'mi' und 'rank' ist
            if isinstance(mi_score, dict) and 'mi' in mi_score:
                mi_value = mi_score['mi']
                rank_value = mi_score.get('rank', 'Unknown')
                
                if isinstance(mi_value, (int, float)):
                    mi_scores.append(mi_value)
                    file_names.append(Path(file_path).name)
                    ranks.append(rank_value)
                    print(f"  ✓ MI: {mi_value:.2f}, Rank: {rank_value}")
                else:
                    print(f"  ✗ Ungültiger MI-Wert: {mi_value}")
            elif isinstance(mi_score, (int, float)):
                # Falls es direkt ein numerischer Wert ist
                mi_scores.append(mi_score)
                file_names.append(Path(file_path).name)
                ranks.append('Unknown')
                print(f"  ✓ MI: {mi_score:.2f}")
            else:
                print(f"  ✗ Unerwartete Struktur: {mi_score} (Type: {type(mi_score)})")
    
    except Exception as e:
        print(f"Fehler bei der MI-Datenverarbeitung: {e}")
        print("Rohdaten:")
        print(mi_data)
    
    if not mi_scores:
        print("Warnung: Keine gültigen MI-Werte gefunden")
        # Dummy-Daten für Visualisierung
        return pd.DataFrame({'file': ['dummy'], 'maintainability': [50.0], 'rank': ['C']})
    
    print(f"Erfolgreich verarbeitet: {len(mi_scores)} MI-Werte")
    
    return pd.DataFrame({
        'file': file_names,
        'maintainability': mi_scores,
        'rank': ranks
    })

def create_comparison_plots(raschka_path, karpathy_path, output_dir="plots"):
    """Erstellt Vergleichsplots zwischen Raschka und Karpathy Modellen"""
    
    # Output-Verzeichnis erstellen
    Path(output_dir).mkdir(exist_ok=True)
    
    print("Extrahiere Metriken für Raschka Modell...")
    comp_raschka, mi_raschka, hal_raschka = extract_radon_metrics(raschka_path)
    
    print("Extrahiere Metriken für Karpathy Modell...")
    comp_karpathy, mi_karpathy, hal_karpathy = extract_radon_metrics(karpathy_path)
    
    if not comp_raschka or not comp_karpathy:
        print("Fehler: Keine Daten für einen oder beide Modelle gefunden.")
        return
    
    # Daten verarbeiten
    df_comp_raschka = process_complexity_data(comp_raschka)
    df_comp_karpathy = process_complexity_data(comp_karpathy)
    df_mi_raschka = process_maintainability_data(mi_raschka)
    df_mi_karpathy = process_maintainability_data(mi_karpathy)
    
    # Modell-Labels hinzufügen
    df_comp_raschka['model'] = 'Raschka (Modular)'
    df_comp_karpathy['model'] = 'Karpathy (Monolithisch)'
    df_mi_raschka['model'] = 'Raschka (Modular)'
    df_mi_karpathy['model'] = 'Karpathy (Monolithisch)'
    
    # DataFrames kombinieren
    df_complexity = pd.concat([df_comp_raschka, df_comp_karpathy], ignore_index=True)
    df_maintainability = pd.concat([df_mi_raschka, df_mi_karpathy], ignore_index=True)
    
    # Plotting
    plt.style.use('seaborn-v0_8')
    fig3, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig3.suptitle('Architektur-Vergleich: Modulares vs. Monolithisches Design', fontsize=18, fontweight='bold')
    
    # 1. Komplexitäts-Boxplot Vergleich
    box_plot = sns.boxplot(data=df_complexity, x='model', y='complexity', ax=axes[0,0])
    axes[0,0].set_title('Zyklomatische Komplexität - Verteilung', fontweight='bold')
    axes[0,0].set_ylabel('McCabe Complexity Score')
    axes[0,0].set_xlabel('Architektur-Typ')
    axes[0,0].tick_params(axis='x', rotation=15)
    
    # Outliers explizit markieren für bessere Sichtbarkeit
    for i, model in enumerate(['Raschka (Modular)', 'Karpathy (Monolithisch)']):
        model_data = df_complexity[df_complexity['model'] == model]['complexity']
        q1 = model_data.quantile(0.25)
        q3 = model_data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = model_data[(model_data < lower_bound) | (model_data > upper_bound)]
        
        # Zusätzliche Annotation für multiple Outliers mit gleichem Wert
        outlier_counts = outliers.value_counts()
        for value, count in outlier_counts.items():
            if count > 1:
                axes[0,0].annotate(f'{count}x', 
                                 xy=(i, value), 
                                 xytext=(5, 5), 
                                 textcoords='offset points',
                                 fontsize=9, 
                                 color='red',
                                 weight='bold')
    
    # 2. Maintainability-Boxplot Vergleich
    sns.boxplot(data=df_maintainability, x='model', y='maintainability', ax=axes[0,1])
    axes[0,1].set_title('Maintainability Index - Vergleich', fontweight='bold')
    axes[0,1].set_ylabel('MI Score')
    axes[0,1].set_xlabel('Architektur-Typ')
    axes[0,1].tick_params(axis='x', rotation=15)
    
    # 3. Komplexitäts-Histogramm Überlagert
    axes[0,2].hist([df_comp_raschka['complexity'], df_comp_karpathy['complexity']], 
                   bins=20, alpha=0.7, label=['Raschka (Modular)', 'Karpathy (Monolithisch)'],
                   color=['skyblue', 'coral'])
    axes[0,2].set_title('Komplexitätsverteilung - Überlagerung', fontweight='bold')
    axes[0,2].set_xlabel('Complexity Score')
    axes[0,2].set_ylabel('Häufigkeit')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. Statistische Zusammenfassung
    summary_data = {
        'Raschka (Modular)': [
            df_comp_raschka['complexity'].mean(),
            df_mi_raschka['maintainability'].mean(),
            len(df_comp_raschka['file'].unique()),
            df_comp_raschka['complexity'].max()
        ],
        'Karpathy (Monolithisch)': [
            df_comp_karpathy['complexity'].mean(),
            df_mi_karpathy['maintainability'].mean(),
            len(df_comp_karpathy['file'].unique()),
            df_comp_karpathy['complexity'].max()
        ]
    }
    
    metrics = ['Ø Komplexität', 'Ø Maintainability', 'Anzahl Module', 'Max Komplexität']
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = axes[1,0].bar(x - width/2, summary_data['Raschka (Modular)'], width, 
                         label='Raschka (Modular)', alpha=0.8, color='skyblue')
    bars2 = axes[1,0].bar(x + width/2, summary_data['Karpathy (Monolithisch)'], width, 
                         label='Karpathy (Monolithisch)', alpha=0.8, color='coral')
    
    axes[1,0].set_title('Architektur-Metriken Vergleich', fontweight='bold')
    axes[1,0].set_ylabel('Score')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(metrics, rotation=45, ha='right')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Werte auf Balken anzeigen
    def autolabel(bars, ax):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    autolabel(bars1, axes[1,0])
    autolabel(bars2, axes[1,0])
    
    # 5. Separate Komplexitäts- und Maintainability-Vergleiche
    fig2, (ax_comp, ax_maint) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Komplexitäts-Metriken (Durchschnitt und Maximum)
    complexity_metrics = ['Ø Komplexität', 'Max Komplexität']
    raschka_complexity = [df_comp_raschka['complexity'].mean(), df_comp_raschka['complexity'].max()]
    karpathy_complexity = [df_comp_karpathy['complexity'].mean(), df_comp_karpathy['complexity'].max()]
    
    x_comp = np.arange(len(complexity_metrics))
    width = 0.35
    
    bars1 = ax_comp.bar(x_comp - width/2, raschka_complexity, width, 
                       label='Raschka (Modular)', color='skyblue', alpha=0.8)
    bars2 = ax_comp.bar(x_comp + width/2, karpathy_complexity, width, 
                       label='Karpathy (Monolithisch)', color='coral', alpha=0.8)
    
    ax_comp.set_title('Komplexitäts-Metriken Vergleich', fontweight='bold')
    ax_comp.set_ylabel('McCabe Complexity Score')
    ax_comp.set_xlabel('Komplexitäts-Typ')
    ax_comp.set_xticks(x_comp)
    ax_comp.set_xticklabels(complexity_metrics)
    ax_comp.legend()
    ax_comp.grid(True, alpha=0.3)
    
    # Werte auf Komplexitäts-Balken
    for bar in bars1:
        height = bar.get_height()
        ax_comp.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax_comp.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
    
    # Maintainability-Metriken
    maintainability_metrics = ['Ø Maintainability', 'Min Maintainability', 'Max Maintainability']
    raschka_maint = [df_mi_raschka['maintainability'].mean(), 
                    df_mi_raschka['maintainability'].min(),
                    df_mi_raschka['maintainability'].max()]
    karpathy_maint = [df_mi_karpathy['maintainability'].mean(),
                     df_mi_karpathy['maintainability'].min(),
                     df_mi_karpathy['maintainability'].max()]
    
    x_maint = np.arange(len(maintainability_metrics))
    
    bars3 = ax_maint.bar(x_maint - width/2, raschka_maint, width, 
                        label='Raschka (Modular)', color='lightgreen', alpha=0.8)
    bars4 = ax_maint.bar(x_maint + width/2, karpathy_maint, width, 
                        label='Karpathy (Monolithisch)', color='orange', alpha=0.8)
    
    ax_maint.set_title('Maintainability Index Vergleich', fontweight='bold')
    ax_maint.set_ylabel('MI Score')
    ax_maint.set_xlabel('Maintainability-Typ')
    ax_maint.set_xticks(x_maint)
    ax_maint.set_xticklabels(maintainability_metrics, rotation=15)
    ax_maint.legend()
    ax_maint.grid(True, alpha=0.3)
    
    # Werte auf Maintainability-Balken
    for bar in bars3:
        height = bar.get_height()
        ax_maint.annotate(f'{height:.1f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontweight='bold')
    
    for bar in bars4:
        height = bar.get_height()
        ax_maint.annotate(f'{height:.1f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    fig2.savefig(f'{output_dir}/komplexitaet_maintainability_vergleich.png', dpi=300, bbox_inches='tight')
    fig2.savefig(f'{output_dir}/komplexitaet_maintainability_vergleich.pdf', bbox_inches='tight')
    plt.show()
    
    # Debug: Komplexitätsdaten prüfen
    print(f"\n🔍 DEBUG - KOMPLEXITÄTSDATEN:")
    print(f"Raschka - Alle Komplexitätswerte: {sorted(df_comp_raschka['complexity'].tolist())}")
    print(f"Karpathy - Alle Komplexitätswerte: {sorted(df_comp_karpathy['complexity'].tolist())}")
    print(f"Raschka - Max: {df_comp_raschka['complexity'].max()}")
    print(f"Karpathy - Max: {df_comp_karpathy['complexity'].max()}")
    print(f"Raschka - Anzahl Funktionen: {len(df_comp_raschka)}")
    print(f"Karpathy - Anzahl Funktionen: {len(df_comp_karpathy)}")
    
    # 6. Architektur-Charakteristika Tabelle
    axes[1,2].axis('off')
    
    table_data = [
        ['Metrik', 'Raschka\n(Modular)', 'Karpathy\n(Monolithisch)', 'Bewertung'],
        ['Ø Komplexität', f'{df_comp_raschka["complexity"].mean():.1f}', 
         f'{df_comp_karpathy["complexity"].mean():.1f}', 
         '✓ Raschka' if df_comp_raschka['complexity'].mean() < df_comp_karpathy['complexity'].mean() else '✓ Karpathy'],
        ['Ø Maintainability', f'{df_mi_raschka["maintainability"].mean():.1f}', 
         f'{df_mi_karpathy["maintainability"].mean():.1f}',
         '✓ Raschka' if df_mi_raschka['maintainability'].mean() > df_mi_karpathy['maintainability'].mean() else '✓ Karpathy'],
        ['Max Komplexität', f'{df_comp_raschka["complexity"].max()}', 
         f'{df_comp_karpathy["complexity"].max()}',
         '✓ Raschka' if df_comp_raschka['complexity'].max() < df_comp_karpathy['complexity'].max() else '✓ Karpathy'],
        ['Anzahl Module', f'{len(df_comp_raschka["file"].unique())}', 
         f'{len(df_comp_karpathy["file"].unique())}',
         '✓ Raschka' if len(df_comp_raschka['file'].unique()) > len(df_comp_karpathy['file'].unique()) else '✓ Karpathy']
    ]
    
    table = axes[1,2].table(cellText=table_data[1:], colLabels=table_data[0], 
                           loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Header-Stil
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    axes[1,2].set_title('Quantitative Bewertung', fontweight='bold')
    
    plt.tight_layout()
    fig3.savefig(f'{output_dir}/architektur_vergleich_raschka_vs_karpathy.png', dpi=300, bbox_inches='tight')
    fig3.savefig(f'{output_dir}/architektur_vergleich_raschka_vs_karpathy.pdf', bbox_inches='tight')
    print(f"Vergleichsplots gespeichert in: {output_dir}/")
    plt.show()
    
    # Wissenschaftliche Zusammenfassung
    print("\n" + "="*80)
    print("QUANTITATIVE CODE-QUALITÄTS-ANALYSE")
    print("="*80)
    
    print(f"\n📊 STRUKTURELLE KOMPLEXITÄT:")
    print(f"{'Metrik':<25} {'Raschka':<15} {'Karpathy':<15}")
    print(f"{'-'*55}")
    print(f"{'Ø Komplexität':<25} {df_comp_raschka['complexity'].mean():<15.2f} {df_comp_karpathy['complexity'].mean():<15.2f}")
    print(f"{'Median Komplexität':<25} {df_comp_raschka['complexity'].median():<15.2f} {df_comp_karpathy['complexity'].median():<15.2f}")
    print(f"{'Max Komplexität':<25} {df_comp_raschka['complexity'].max():<15} {df_comp_karpathy['complexity'].max():<15}")
    print(f"{'Std. Abweichung':<25} {df_comp_raschka['complexity'].std():<15.2f} {df_comp_karpathy['complexity'].std():<15.2f}")
    
    print(f"\n🔧 WARTBARKEIT:")
    print(f"{'Metrik':<25} {'Raschka':<15} {'Karpathy':<15}")
    print(f"{'-'*55}")
    print(f"{'Ø Maintainability':<25} {df_mi_raschka['maintainability'].mean():<15.2f} {df_mi_karpathy['maintainability'].mean():<15.2f}")
    print(f"{'Min Maintainability':<25} {df_mi_raschka['maintainability'].min():<15.2f} {df_mi_karpathy['maintainability'].min():<15.2f}")
    print(f"{'Max Maintainability':<25} {df_mi_raschka['maintainability'].max():<15.2f} {df_mi_karpathy['maintainability'].max():<15.2f}")
    
    print(f"\n🏗️ ARCHITEKTUR:")
    print(f"{'Metrik':<25} {'Raschka':<15} {'Karpathy':<15}")
    print(f"{'-'*55}")
    print(f"{'Anzahl Module':<25} {len(df_comp_raschka['file'].unique()):<15} {len(df_comp_karpathy['file'].unique()):<15}")
    print(f"{'Anzahl Funktionen':<25} {len(df_comp_raschka):<15} {len(df_comp_karpathy):<15}")
    
    # Detaillierte Outlier-Analyse
    print(f"\n📈 KOMPLEXITÄTS-OUTLIERS (≥ 8):")
    raschka_outliers = df_comp_raschka[df_comp_raschka['complexity'] >= 8]
    karpathy_outliers = df_comp_karpathy[df_comp_karpathy['complexity'] >= 8]
    
    print(f"Raschka: {len(raschka_outliers)} von {len(df_comp_raschka)} Funktionen ({len(raschka_outliers)/len(df_comp_raschka)*100:.1f}%)")
    if len(raschka_outliers) > 0:
        outlier_counts_r = raschka_outliers['complexity'].value_counts().sort_index()
        for complexity, count in outlier_counts_r.items():
            print(f"  - Komplexität {complexity}: {count} Funktion(en)")
    
    print(f"Karpathy: {len(karpathy_outliers)} von {len(df_comp_karpathy)} Funktionen ({len(karpathy_outliers)/len(df_comp_karpathy)*100:.1f}%)")
    if len(karpathy_outliers) > 0:
        outlier_counts_k = karpathy_outliers['complexity'].value_counts().sort_index()
        for complexity, count in outlier_counts_k.items():
            print(f"  - Komplexität {complexity}: {count} Funktion(en)")
    
    return df_complexity, df_maintainability
    

    




    # Output-Verzeichnis erstellen
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"Extrahiere Metriken für {model_name}...")
    comp_data, mi_data, hal_data = extract_radon_metrics(model_path)
    
    if not comp_data:
        print("Fehler: Keine Daten gefunden.")
        return
    
    # Daten verarbeiten
    df_complexity = process_complexity_data(comp_data)
    df_maintainability = process_maintainability_data(mi_data)
    
    # Plotting
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Code-Metriken Analyse: {model_name}', fontsize=16, fontweight='bold')
    
    # 1. Komplexitäts-Balkendiagramm (statt Histogramm)
    complexity_values = df_complexity['complexity']
    complexity_counts = complexity_values.value_counts().sort_index()
    
    bars = axes[0,0].bar(complexity_counts.index, complexity_counts.values, 
                        alpha=0.8, color='skyblue', edgecolor='black', width=0.6)
    axes[0,0].set_title('Zyklomatische Komplexität - Verteilung')
    axes[0,0].set_xlabel('McCabe Complexity Score')
    axes[0,0].set_ylabel('Häufigkeit')
    axes[0,0].grid(True, alpha=0.3, axis='y')
    
    # X-Achsen-Ticks explizit setzen
    axes[0,0].set_xticks(complexity_counts.index)
    axes[0,0].set_xlim(complexity_counts.index.min() - 0.5, complexity_counts.index.max() + 0.5)
    
    # Werte auf Balken anzeigen
    for bar in bars:
        height = bar.get_height()
        axes[0,0].annotate(f'{int(height)}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3), textcoords="offset points",
                          ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 2. Maintainability-Balkendiagramm (statt Histogramm)
    mi_values = df_maintainability['maintainability']
    # Für kontinuierliche Werte wie MI, Bereiche erstellen
    mi_bins = np.linspace(mi_values.min(), mi_values.max(), 6)
    mi_labels = [f"{mi_bins[i]:.1f}-{mi_bins[i+1]:.1f}" for i in range(len(mi_bins)-1)]
    mi_digitized = np.digitize(mi_values, mi_bins) - 1
    mi_counts = pd.Series(mi_digitized).value_counts().sort_index()
    
    bars2 = axes[0,1].bar(range(len(mi_counts)), mi_counts.values, 
                         alpha=0.8, color='lightgreen', edgecolor='black', width=0.6)
    axes[0,1].set_title('Maintainability Index - Verteilung')
    axes[0,1].set_xlabel('MI Score Bereiche')
    axes[0,1].set_ylabel('Häufigkeit')
    axes[0,1].grid(True, alpha=0.3, axis='y')
    
    # X-Achsen-Labels für MI-Bereiche
    valid_indices = mi_counts.index
    axes[0,1].set_xticks(range(len(mi_counts)))
    axes[0,1].set_xticklabels([mi_labels[i] if i < len(mi_labels) else f">{mi_bins[-2]:.1f}" 
                              for i in valid_indices], rotation=45, ha='right')
    
    # Werte auf Balken anzeigen
    for bar in bars2:
        height = bar.get_height()
        if height > 0:  # Nur anzeigen wenn Wert > 0
            axes[0,1].annotate(f'{int(height)}',
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 3), textcoords="offset points",
                              ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 3. Komplexität pro Datei (Top 10)
    top_complex_files = df_complexity.groupby('file')['complexity'].max().sort_values(ascending=False).head(10)
    axes[1,0].barh(range(len(top_complex_files)), top_complex_files.values, color='coral')
    axes[1,0].set_yticks(range(len(top_complex_files)))
    axes[1,0].set_yticklabels([f[:20] + '...' if len(f) > 20 else f for f in top_complex_files.index])
    axes[1,0].set_title('Top 10 - Komplexeste Dateien')
    axes[1,0].set_xlabel('Max. Complexity Score')
    
    # 4. Zusammenfassungsstatistiken
    stats_data = {
        'Komplexität': [
            df_complexity['complexity'].mean(),
            df_complexity['complexity'].median(),
            df_complexity['complexity'].max(),
            df_complexity['complexity'].std()
        ],
        'Maintainability': [
            df_maintainability['maintainability'].mean(),
            df_maintainability['maintainability'].median(),
            df_maintainability['maintainability'].max(),
            df_maintainability['maintainability'].std()
        ]
    }
    
    x = np.arange(4)
    width = 0.35
    
    axes[1,1].bar(x - width/2, stats_data['Komplexität'], width, label='Komplexität', alpha=0.8)
    axes[1,1].bar(x + width/2, stats_data['Maintainability'], width, label='Maintainability', alpha=0.8)
    axes[1,1].set_title('Statistische Kennwerte')
    axes[1,1].set_ylabel('Score')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(['Mittelwert', 'Median', 'Maximum', 'Std.abw.'])
    axes[1,1].legend()
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name.lower().replace(" ", "_")}_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/{model_name.lower().replace(" ", "_")}_analysis.pdf', bbox_inches='tight')
    print(f"Plots gespeichert in: {output_dir}/")
    plt.show()
    
    # Detaillierte Statistische Zusammenfassung
    print(f"\n=== DETAILLIERTE ANALYSE: {model_name.upper()} ===")
    print(f"\n📊 KOMPLEXITÄTS-ANALYSE:")
    print(f"  Anzahl Funktionen/Methoden: {len(df_complexity)}")
    print(f"  Mittelwert: {df_complexity['complexity'].mean():.2f}")
    print(f"  Median: {df_complexity['complexity'].median():.2f}")
    print(f"  Standardabweichung: {df_complexity['complexity'].std():.2f}")
    print(f"  Minimum: {df_complexity['complexity'].min()}")
    print(f"  Maximum: {df_complexity['complexity'].max()}")
    
    # Komplexitätskategorien
    low_complex = len(df_complexity[df_complexity['complexity'] <= 5])
    med_complex = len(df_complexity[(df_complexity['complexity'] > 5) & (df_complexity['complexity'] <= 10)])
    high_complex = len(df_complexity[df_complexity['complexity'] > 10])
    
    print(f"\n  Komplexitätskategorien:")
    print(f"    Niedrig (≤5): {low_complex} ({low_complex/len(df_complexity)*100:.1f}%)")
    print(f"    Mittel (6-10): {med_complex} ({med_complex/len(df_complexity)*100:.1f}%)")
    print(f"    Hoch (>10): {high_complex} ({high_complex/len(df_complexity)*100:.1f}%)")
    
    print(f"\n🔧 MAINTAINABILITY-ANALYSE:")
    print(f"  Anzahl Dateien: {len(df_maintainability)}")
    print(f"  Mittelwert: {df_maintainability['maintainability'].mean():.2f}")
    print(f"  Median: {df_maintainability['maintainability'].median():.2f}")
    print(f"  Standardabweichung: {df_maintainability['maintainability'].std():.2f}")
    print(f"  Minimum: {df_maintainability['maintainability'].min():.2f}")
    print(f"  Maximum: {df_maintainability['maintainability'].max():.2f}")
    
    # Maintainability-Kategorien (basierend auf MI-Standards)
    excellent = len(df_maintainability[df_maintainability['maintainability'] >= 85])
    good = len(df_maintainability[(df_maintainability['maintainability'] >= 65) & (df_maintainability['maintainability'] < 85)])
    moderate = len(df_maintainability[(df_maintainability['maintainability'] >= 25) & (df_maintainability['maintainability'] < 65)])
    poor = len(df_maintainability[df_maintainability['maintainability'] < 25])
    
    print(f"\n  Maintainability-Kategorien:")
    print(f"    Exzellent (≥85): {excellent} ({excellent/len(df_maintainability)*100:.1f}%)")
    print(f"    Gut (65-84): {good} ({good/len(df_maintainability)*100:.1f}%)")
    print(f"    Moderat (25-64): {moderate} ({moderate/len(df_maintainability)*100:.1f}%)")
    print(f"    Problematisch (<25): {poor} ({poor/len(df_maintainability)*100:.1f}%)")
    
    # Radon-Ranking-Verteilung
    if 'rank' in df_maintainability.columns:
        print(f"\n  Radon-Ranking-Verteilung:")
        rank_counts = df_maintainability['rank'].value_counts()
        for rank, count in rank_counts.items():
            print(f"    Rang {rank}: {count} ({count/len(df_maintainability)*100:.1f}%)")
    
    # Detaillierte Datei-Bewertungen
    print(f"\n  📋 DETAILLIERTE DATEI-BEWERTUNGEN:")
    df_sorted = df_maintainability.sort_values('maintainability', ascending=False)
    for idx, row in df_sorted.iterrows():
        rank_info = f" (Rang: {row['rank']})" if 'rank' in row and pd.notna(row['rank']) else ""
        print(f"    {row['file']}: {row['maintainability']:.2f}{rank_info}")
    
    return df_complexity, df_maintainability

def analyze_single_model(model_path, model_name="Model", output_dir="plots"):
    """Analysiert ein einzelnes Modell und erstellt Visualisierungen"""
    
    # Output-Verzeichnis erstellen
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"Extrahiere Metriken für {model_name}...")
    comp_data, mi_data, hal_data = extract_radon_metrics(model_path)
    
    if not comp_data:
        print("Fehler: Keine Daten gefunden.")
        return
    
    # Daten verarbeiten
    df_complexity = process_complexity_data(comp_data)
    df_maintainability = process_maintainability_data(mi_data)
    
    # Plotting
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Code-Metriken Analyse: {model_name}', fontsize=16, fontweight='bold')
    
    # 1. Komplexitäts-Balkendiagramm (statt Histogramm)
    complexity_values = df_complexity['complexity']
    complexity_counts = complexity_values.value_counts().sort_index()
    
    bars = axes[0,0].bar(complexity_counts.index, complexity_counts.values, 
                        alpha=0.8, color='skyblue', edgecolor='black', width=0.6)
    axes[0,0].set_title('Zyklomatische Komplexität - Verteilung')
    axes[0,0].set_xlabel('McCabe Complexity Score')
    axes[0,0].set_ylabel('Häufigkeit')
    axes[0,0].grid(True, alpha=0.3, axis='y')
    
    # X-Achsen-Ticks explizit setzen
    axes[0,0].set_xticks(complexity_counts.index)
    axes[0,0].set_xlim(complexity_counts.index.min() - 0.5, complexity_counts.index.max() + 0.5)
    
    # Werte auf Balken anzeigen
    for bar in bars:
        height = bar.get_height()
        axes[0,0].annotate(f'{int(height)}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3), textcoords="offset points",
                          ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 2. Maintainability-Balkendiagramm (statt Histogramm)
    mi_values = df_maintainability['maintainability']
    # Für kontinuierliche Werte wie MI, Bereiche erstellen
    mi_bins = np.linspace(mi_values.min(), mi_values.max(), 6)
    mi_labels = [f"{mi_bins[i]:.1f}-{mi_bins[i+1]:.1f}" for i in range(len(mi_bins)-1)]
    mi_digitized = np.digitize(mi_values, mi_bins) - 1
    mi_counts = pd.Series(mi_digitized).value_counts().sort_index()
    
    bars2 = axes[0,1].bar(range(len(mi_counts)), mi_counts.values, 
                         alpha=0.8, color='lightgreen', edgecolor='black', width=0.6)
    axes[0,1].set_title('Maintainability Index - Verteilung')
    axes[0,1].set_xlabel('MI Score Bereiche')
    axes[0,1].set_ylabel('Häufigkeit')
    axes[0,1].grid(True, alpha=0.3, axis='y')
    
    # X-Achsen-Labels für MI-Bereiche
    valid_indices = mi_counts.index
    axes[0,1].set_xticks(range(len(mi_counts)))
    axes[0,1].set_xticklabels([mi_labels[i] if i < len(mi_labels) else f">{mi_bins[-2]:.1f}" 
                              for i in valid_indices], rotation=45, ha='right')
    
    # Werte auf Balken anzeigen
    for bar in bars2:
        height = bar.get_height()
        if height > 0:  # Nur anzeigen wenn Wert > 0
            axes[0,1].annotate(f'{int(height)}',
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 3), textcoords="offset points",
                              ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 3. Komplexität pro Datei (Top 10)
    top_complex_files = df_complexity.groupby('file')['complexity'].max().sort_values(ascending=False).head(10)
    axes[1,0].barh(range(len(top_complex_files)), top_complex_files.values, color='coral')
    axes[1,0].set_yticks(range(len(top_complex_files)))
    axes[1,0].set_yticklabels([f[:20] + '...' if len(f) > 20 else f for f in top_complex_files.index])
    axes[1,0].set_title('Top 10 - Komplexeste Dateien')
    axes[1,0].set_xlabel('Max. Complexity Score')
    
    # 4. Zusammenfassungsstatistiken
    stats_data = {
        'Komplexität': [
            df_complexity['complexity'].mean(),
            df_complexity['complexity'].median(),
            df_complexity['complexity'].max(),
            df_complexity['complexity'].std()
        ],
        'Maintainability': [
            df_maintainability['maintainability'].mean(),
            df_maintainability['maintainability'].median(),
            df_maintainability['maintainability'].max(),
            df_maintainability['maintainability'].std()
        ]
    }
    
    x = np.arange(4)
    width = 0.35
    
    axes[1,1].bar(x - width/2, stats_data['Komplexität'], width, label='Komplexität', alpha=0.8)
    axes[1,1].bar(x + width/2, stats_data['Maintainability'], width, label='Maintainability', alpha=0.8)
    axes[1,1].set_title('Statistische Kennwerte')
    axes[1,1].set_ylabel('Score')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(['Mittelwert', 'Median', 'Maximum', 'Std.abw.'])
    axes[1,1].legend()
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    fig.savefig(f'{output_dir}/{model_name.lower().replace(" ", "_")}_analysis.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{output_dir}/{model_name.lower().replace(" ", "_")}_analysis.pdf', bbox_inches='tight')
    print(f"Plots gespeichert in: {output_dir}/")
    plt.show()
    
    # Detaillierte Statistische Zusammenfassung
    print(f"\n=== DETAILLIERTE ANALYSE: {model_name.upper()} ===")
    print(f"\n📊 KOMPLEXITÄTS-ANALYSE:")
    print(f"  Anzahl Funktionen/Methoden: {len(df_complexity)}")
    print(f"  Mittelwert: {df_complexity['complexity'].mean():.2f}")
    print(f"  Median: {df_complexity['complexity'].median():.2f}")
    print(f"  Standardabweichung: {df_complexity['complexity'].std():.2f}")
    print(f"  Minimum: {df_complexity['complexity'].min()}")
    print(f"  Maximum: {df_complexity['complexity'].max()}")
    
    # Komplexitätskategorien (nach Radon-Standards)
    a_rank = len(df_complexity[df_complexity['complexity'] <= 5])  # A: 1-5
    b_rank = len(df_complexity[(df_complexity['complexity'] >= 6) & (df_complexity['complexity'] <= 10)])  # B: 6-10
    c_rank = len(df_complexity[(df_complexity['complexity'] >= 11) & (df_complexity['complexity'] <= 20)])  # C: 11-20
    d_rank = len(df_complexity[(df_complexity['complexity'] >= 21) & (df_complexity['complexity'] <= 30)])  # D: 21-30
    e_rank = len(df_complexity[(df_complexity['complexity'] >= 31) & (df_complexity['complexity'] <= 40)])  # E: 31-40
    f_rank = len(df_complexity[df_complexity['complexity'] >= 41])  # F: 41+
    
    print(f"\n  Radon-Komplexitätskategorien:")
    print(f"    A (1-5): {a_rank} ({a_rank/len(df_complexity)*100:.1f}%) - low risk")
    print(f"    B (6-10): {b_rank} ({b_rank/len(df_complexity)*100:.1f}%) - low risk")
    print(f"    C (11-20): {c_rank} ({c_rank/len(df_complexity)*100:.1f}%) - moderate risk")
    print(f"    D (21-30): {d_rank} ({d_rank/len(df_complexity)*100:.1f}%) - more than moderate")
    print(f"    E (31-40): {e_rank} ({e_rank/len(df_complexity)*100:.1f}%) - high risk")
    print(f"    F (41+): {f_rank} ({f_rank/len(df_complexity)*100:.1f}%) - very high risk")
    
    print(f"\n🔧 MAINTAINABILITY-ANALYSE:")
    print(f"  Anzahl Dateien: {len(df_maintainability)}")
    print(f"  Mittelwert: {df_maintainability['maintainability'].mean():.2f}")
    print(f"  Median: {df_maintainability['maintainability'].median():.2f}")
    print(f"  Standardabweichung: {df_maintainability['maintainability'].std():.2f}")
    print(f"  Minimum: {df_maintainability['maintainability'].min():.2f}")
    print(f"  Maximum: {df_maintainability['maintainability'].max():.2f}")
    
    # Maintainability-Kategorien (basierend auf MI-Standards)
    excellent = len(df_maintainability[df_maintainability['maintainability'] >= 85])
    good = len(df_maintainability[(df_maintainability['maintainability'] >= 65) & (df_maintainability['maintainability'] < 85)])
    moderate = len(df_maintainability[(df_maintainability['maintainability'] >= 25) & (df_maintainability['maintainability'] < 65)])
    poor = len(df_maintainability[df_maintainability['maintainability'] < 25])
    
    print(f"\n  Maintainability-Kategorien:")
    print(f"    Exzellent (≥85): {excellent} ({excellent/len(df_maintainability)*100:.1f}%)")
    print(f"    Gut (65-84): {good} ({good/len(df_maintainability)*100:.1f}%)")
    print(f"    Moderat (25-64): {moderate} ({moderate/len(df_maintainability)*100:.1f}%)")
    print(f"    Problematisch (<25): {poor} ({poor/len(df_maintainability)*100:.1f}%)")
    
    # Radon-Ranking-Verteilung
    if 'rank' in df_maintainability.columns:
        print(f"\n  Radon-Ranking-Verteilung:")
        rank_counts = df_maintainability['rank'].value_counts()
        for rank, count in rank_counts.items():
            print(f"    Rang {rank}: {count} ({count/len(df_maintainability)*100:.1f}%)")
    
    # Detaillierte Datei-Bewertungen
    print(f"\n  📋 DETAILLIERTE DATEI-BEWERTUNGEN:")
    df_sorted = df_maintainability.sort_values('maintainability', ascending=False)
    for idx, row in df_sorted.iterrows():
        rank_info = f" (Rang: {row['rank']})" if 'rank' in row and pd.notna(row['rank']) else ""
        print(f"    {row['file']}: {row['maintainability']:.2f}{rank_info}")
    
    return df_complexity, df_maintainability


def main():
    """Hauptfunktion - Analyse-Optionen"""
    
    # HIER PFADE ANPASSEN:
    raschka_path = r"E:\01 Uni\02 Bachelor 2025\Codemetriken2\Raschka"
    karpathy_path = r"E:\01 Uni\02 Bachelor 2025\Codemetriken2\Karpathy"
    
    print("Code-Metriken Analyse Tool")
    print("=" * 50)
    print("1. Nur Raschka Modell analysieren")
    print("2. Nur Karpathy Modell analysieren") 
    print("3. Beide Modelle vergleichen")
    print("=" * 50)
    
    choice = input("Bitte wählen Sie eine Option (1-3): ").strip()
    
    # Überprüfen ob radon installiert ist
    try:
        subprocess.run(['radon', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Fehler: 'radon' ist nicht installiert oder nicht im PATH.")
        print("Installation mit: pip install radon")
        return
    
    if choice == "1":
        # Nur Raschka
        if not Path(raschka_path).exists():
            print(f"Fehler: Pfad {raschka_path} existiert nicht!")
            return
        
        print("Starte Analyse für Raschkas Modell...")
        analyze_single_model(raschka_path, "Raschka Model")
        
    elif choice == "2":
        # Nur Karpathy
        if not Path(karpathy_path).exists():
            print(f"Fehler: Pfad {karpathy_path} existiert nicht!")
            return
        
        print("Starte Analyse für Karpathys Modell...")
        analyze_single_model(karpathy_path, "Karpathy Model")
        
    elif choice == "3":
        # Vergleich beider Modelle
        if not Path(raschka_path).exists():
            print(f"Fehler: Pfad {raschka_path} existiert nicht!")
            return
        if not Path(karpathy_path).exists():
            print(f"Fehler: Pfad {karpathy_path} existiert nicht!")
            return
        
        print("Starte Vergleichsanalyse beider Modelle...")
        create_comparison_plots(raschka_path, karpathy_path)
        
    else:
        print("Ungültige Auswahl. Bitte 1, 2 oder 3 wählen.")
        return
    
    print("\nAnalyse abgeschlossen!")
    print("Die Ergebnisse wurden als PNG und PDF im 'plots/' Ordner gespeichert.")

if __name__ == "__main__":
    main()