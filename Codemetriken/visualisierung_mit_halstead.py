import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

'''
Visualisierung erstellt mit Claude.ai
'''

# Daten definieren
karpathy_data = {
    'file': 'train_gpt2_final.py',
    'h1': 20,
    'h2': 179,
    'N1': 149,
    'N2': 271,
    'vocabulary': 199,
    'length': 420,
    'calculated_length': 1426.041586028049,
    'volume': 3207.3823406283323,
    'difficulty': 15.139664804469273,
    'effort': 48558.69353688704,
    'time': 2697.7051964937245,
    'bugs': 1.0691274468761107
}

raschka_data = [
    {
        'file': 'c2_data_preparation_sampling.py',
        'h1': 4, 'h2': 8, 'N1': 7, 'N2': 14,
        'vocabulary': 12, 'length': 21, 'calculated_length': 32.0,
        'volume': 75.28421251514429, 'difficulty': 3.5,
        'effort': 263.494743803005, 'time': 14.638596877944723,
        'bugs': 0.025094737505048096
    },
    {
        'file': 'c3_coding_attention_mechanismus.py',
        'h1': 7, 'h2': 16, 'N1': 11, 'N2': 19,
        'vocabulary': 23, 'length': 30, 'calculated_length': 83.65148445440323,
        'volume': 135.7068586817104, 'difficulty': 4.15625,
        'effort': 564.0316313958589, 'time': 31.33509063310327,
        'bugs': 0.04523561956057014
    },
    {
        'file': 'c4_implementing_gpt_model.py',
        'h1': 6, 'h2': 32, 'N1': 24, 'N2': 42,
        'vocabulary': 38, 'length': 66, 'calculated_length': 175.50977500432694,
        'volume': 346.3632158872766, 'difficulty': 3.9375,
        'effort': 1363.8051625561518, 'time': 75.76695347534177,
        'bugs': 0.11545440529575887
    },
    {
        'file': 'c5_gpt_download.py',
        'h1': 6, 'h2': 11, 'N1': 8, 'N2': 13,
        'vocabulary': 17, 'length': 21, 'calculated_length': 53.563522809337215,
        'volume': 85.83671966625714, 'difficulty': 3.5454545454545454,
        'effort': 304.33018790763896, 'time': 16.9072326615355,
        'bugs': 0.02861223988875238
    },
    {
        'file': 'c5_pretraining_unlabeld_data_erweitert.py',
        'h1': 13, 'h2': 49, 'N1': 42, 'N2': 76,
        'vocabulary': 62, 'length': 118, 'calculated_length': 323.22649869747937,
        'volume': 702.5951646256514, 'difficulty': 10.081632653061224,
        'effort': 7083.3063535728925, 'time': 393.51701964293846,
        'bugs': 0.23419838820855046
    },
    {
        'file': 'c7_finetuning_follow_instruction_model.py',
        'h1': 9, 'h2': 26, 'N1': 17, 'N2': 31,
        'vocabulary': 35, 'length': 48, 'calculated_length': 150.74075768464922,
        'volume': 246.20558481335837, 'difficulty': 5.365384615384615,
        'effort': 1320.987656979365, 'time': 73.38820316552028,
        'bugs': 0.08206852827111946
    }
]

def create_halstead_visualization():
    """Erstellt umfassende Halstead-Metriken Visualisierung"""
    
    # DataFrames erstellen
    df_karpathy = pd.DataFrame([karpathy_data])
    df_karpathy['model'] = 'Karpathy (Monolithisch)'
    
    df_raschka = pd.DataFrame(raschka_data)
    df_raschka['model'] = 'Raschka (Modular)'
    
    # Kombiniertes DataFrame
    df_combined = pd.concat([df_karpathy, df_raschka], ignore_index=True)
    
    # Plotting
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 16))
    
    # Layout: 3x3 Grid mit mehr Zwischenraum
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)
    
    # 1. Volume Vergleich
    ax1 = fig.add_subplot(gs[0, 0])
    karpathy_vol = df_karpathy['volume'].iloc[0]
    raschka_vols = df_raschka['volume'].values
    
    positions = [0] + list(range(2, 2 + len(raschka_vols)))
    values = [karpathy_vol] + list(raschka_vols)
    colors = ['red'] + ['blue'] * len(raschka_vols)
    
    bars1 = ax1.bar(positions, values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_title('Program Volume', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Volume')
    ax1.set_xticks(positions)
    ax1.set_xticklabels(['Karpathy'] + [f'C{i+2}' for i in range(len(raschka_vols))], rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Mehr Platz nach oben für bessere Lesbarkeit
    ax1.set_ylim(0, max(values) * 1.25)
    
    # Werte auf Balken
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. Difficulty Vergleich
    ax2 = fig.add_subplot(gs[0, 1])
    karpathy_diff = df_karpathy['difficulty'].iloc[0]
    raschka_diffs = df_raschka['difficulty'].values
    
    values_diff = [karpathy_diff] + list(raschka_diffs)
    bars2 = ax2.bar(positions, values_diff, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_title('Program Difficulty', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Difficulty')
    ax2.set_xticks(positions)
    ax2.set_xticklabels(['Karpathy'] + [f'C{i+2}' for i in range(len(raschka_diffs))], rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Mehr Platz nach oben
    ax2.set_ylim(0, max(values_diff) * 1.25)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 3. Effort Vergleich (logarithmische Skala)
    ax3 = fig.add_subplot(gs[0, 2])
    karpathy_eff = df_karpathy['effort'].iloc[0]
    raschka_effs = df_raschka['effort'].values
    
    values_eff = [karpathy_eff] + list(raschka_effs)
    bars3 = ax3.bar(positions, values_eff, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_title('Programming Effort', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Effort (log scale)')
    ax3.set_yscale('log')
    ax3.set_xticks(positions)
    ax3.set_xticklabels(['Karpathy'] + [f'C{i+2}' for i in range(len(raschka_effs))], rotation=45)
    ax3.grid(True, alpha=0.3)

    # Erweiterte Y-Achse für logarithmische Skala
    ax3.set_ylim(min(values_eff) * 0.5, max(values_eff) * 2.5)

    for bar in bars3:
        height = bar.get_height()
        ax3.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 8), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 4. Vocabulary Vergleich
    ax4 = fig.add_subplot(gs[1, 0])
    karpathy_vocab = df_karpathy['vocabulary'].iloc[0]
    raschka_vocabs = df_raschka['vocabulary'].values
    
    values_vocab = [karpathy_vocab] + list(raschka_vocabs)
    bars4 = ax4.bar(positions, values_vocab, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_title('Program Vocabulary', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Vocabulary Size')
    ax4.set_xticks(positions)
    ax4.set_xticklabels(['Karpathy'] + [f'C{i+2}' for i in range(len(raschka_vocabs))], rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Mehr Platz nach oben
    ax4.set_ylim(0, max(values_vocab) * 1.25)
    
    for bar in bars4:
        height = bar.get_height()
        ax4.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 5. Time Vergleich (logarithmische Skala)
    ax5 = fig.add_subplot(gs[1, 1])
    karpathy_time = df_karpathy['time'].iloc[0]
    raschka_times = df_raschka['time'].values
    
    values_time = [karpathy_time] + list(raschka_times)
    bars5 = ax5.bar(positions, values_time, color=colors, alpha=0.7, edgecolor='black')
    ax5.set_title('Estimated Programming Time', fontweight='bold', fontsize=12)
    ax5.set_ylabel('Time (seconds, log scale)')
    ax5.set_yscale('log')
    ax5.set_xticks(positions)
    ax5.set_xticklabels(['Karpathy'] + [f'C{i+2}' for i in range(len(raschka_times))], rotation=45)
    ax5.grid(True, alpha=0.3)
    
    # Erweiterte Y-Achse für logarithmische Skala
    ax5.set_ylim(min(values_time) * 0.5, max(values_time) * 2.5)

    for bar in bars5:
        height = bar.get_height()
        ax5.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 8), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 6. Bugs Vergleich
    ax6 = fig.add_subplot(gs[1, 2])
    karpathy_bugs = df_karpathy['bugs'].iloc[0]
    raschka_bugs_list = df_raschka['bugs'].values
    
    values_bugs = [karpathy_bugs] + list(raschka_bugs_list)
    bars6 = ax6.bar(positions, values_bugs, color=colors, alpha=0.7, edgecolor='black')
    ax6.set_title('Estimated Bugs', fontweight='bold', fontsize=12)
    ax6.set_ylabel('Number of Bugs')
    ax6.set_xticks(positions)
    ax6.set_xticklabels(['Karpathy'] + [f'C{i+2}' for i in range(len(raschka_bugs_list))], rotation=45)
    ax6.grid(True, alpha=0.3)
    
    # Mehr Platz nach oben
    ax6.set_ylim(0, max(values_bugs) * 1.25)
    
    for bar in bars6:
        height = bar.get_height()
        ax6.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 7. Durchschnittswerte Vergleich
    ax7 = fig.add_subplot(gs[2, 0])
    metrics = ['Volume', 'Difficulty', 'Effort/1000', 'Vocabulary', 'Time/100']
    karpathy_means = [
        karpathy_vol,
        karpathy_diff,
        karpathy_eff / 1000,
        karpathy_vocab,
        karpathy_time / 100
    ]
    raschka_means = [
        np.mean(raschka_vols),
        np.mean(raschka_diffs),
        np.mean(raschka_effs) / 1000,
        np.mean(raschka_vocabs),
        np.mean(raschka_times) / 100
    ]
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    bars7a = ax7.bar(x_pos - width/2, karpathy_means, width, 
                     label='Karpathy (Monolithisch)', color='red', alpha=0.7)
    bars7b = ax7.bar(x_pos + width/2, raschka_means, width, 
                     label='Raschka (Modular Ø)', color='blue', alpha=0.7)
    
    ax7.set_title('Durchschnittswerte Vergleich', fontweight='bold', fontsize=12)
    ax7.set_ylabel('Skalierte Werte')
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(metrics, rotation=45)
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Erweiterte Y-Achse für bessere Sichtbarkeit
    ax7.set_ylim(0, max(max(karpathy_means), max(raschka_means)) * 1.2)

    # Werte auf Balken
    for bar in bars7a:
        height = bar.get_height()
        ax7.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 8), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    for bar in bars7b:
        height = bar.get_height()
        ax7.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 8. Volume vs Difficulty Scatter Plot
    ax8 = fig.add_subplot(gs[2, 1])
    
    # Karpathy Punkt
    ax8.scatter(karpathy_vol, karpathy_diff, color='red', s=200, 
               alpha=0.8, edgecolors='black', label='Karpathy')
    
    # Raschka Punkte
    ax8.scatter(raschka_vols, raschka_diffs, color='blue', s=100, 
               alpha=0.8, edgecolors='black', label='Raschka')
    
    ax8.set_title('Volume vs. Difficulty', fontweight='bold', fontsize=12)
    ax8.set_xlabel('Program Volume')
    ax8.set_ylabel('Program Difficulty')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Datenpunkt-Labels für Raschka
    for i, (vol, diff) in enumerate(zip(raschka_vols, raschka_diffs)):
        ax8.annotate(f'C{i+2}', (vol, diff), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8)
    
    # 9. Statistik-Tabelle
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    stats_data = [
        ['Metrik', 'Karpathy', 'Raschka Ø', 'Verhältnis'],
        ['Volume', f'{karpathy_vol:.0f}', f'{np.mean(raschka_vols):.0f}', 
         f'{karpathy_vol/np.mean(raschka_vols):.1f}x'],
        ['Difficulty', f'{karpathy_diff:.1f}', f'{np.mean(raschka_diffs):.1f}', 
         f'{karpathy_diff/np.mean(raschka_diffs):.1f}x'],
        ['Effort', f'{karpathy_eff:.0f}', f'{np.mean(raschka_effs):.0f}', 
         f'{karpathy_eff/np.mean(raschka_effs):.1f}x'],
        ['Vocabulary', f'{karpathy_vocab}', f'{np.mean(raschka_vocabs):.0f}', 
         f'{karpathy_vocab/np.mean(raschka_vocabs):.1f}x'],
        ['Bugs', f'{karpathy_bugs:.3f}', f'{np.mean(raschka_bugs_list):.3f}', 
         f'{karpathy_bugs/np.mean(raschka_bugs_list):.1f}x']
    ]
    
    table = ax9.table(cellText=stats_data[1:], colLabels=stats_data[0], 
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Header-Stil
    for i in range(len(stats_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax9.set_title('Statistische Übersicht', fontweight='bold', fontsize=12)
    
    # Haupttitel
    fig.suptitle('Halstead-Metriken Analyse: Monolithischer vs. Modularer Ansatz', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Speichern und Anzeigen
    plt.savefig('halstead_complete_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('halstead_complete_analysis.pdf', bbox_inches='tight')
    plt.show()
    
    # Zusammenfassung ausgeben
    print_summary_statistics(df_karpathy, df_raschka)

def print_summary_statistics(df_karpathy, df_raschka):
    """Gibt eine wissenschaftliche Zusammenfassung aus"""
    
    print("\n" + "="*80)
    print("HALSTEAD-METRIKEN WISSENSCHAFTLICHE ANALYSE")
    print("="*80)
    
    karpathy_vol = df_karpathy['volume'].iloc[0]
    karpathy_diff = df_karpathy['difficulty'].iloc[0]
    karpathy_eff = df_karpathy['effort'].iloc[0]
    
    raschka_vol_mean = df_raschka['volume'].mean()
    raschka_diff_mean = df_raschka['difficulty'].mean()
    raschka_eff_mean = df_raschka['effort'].mean()
    
    print(f"\nPROGRAM VOLUME (Informationsgehalt):")
    print(f"  Karpathy (Monolithisch):  {karpathy_vol:.1f}")
    print(f"  Raschka (Modular Ø):      {raschka_vol_mean:.1f}")
    print(f"  Verhältnis:               {karpathy_vol/raschka_vol_mean:.1f}:1")
    print(f"  Raschka Spanne:           {df_raschka['volume'].min():.1f} - {df_raschka['volume'].max():.1f}")
    
    print(f"\nPROGRAM DIFFICULTY (Wartbarkeitskomplexität):")
    print(f"  Karpathy (Monolithisch):  {karpathy_diff:.1f}")
    print(f"  Raschka (Modular Ø):      {raschka_diff_mean:.1f}")
    print(f"  Verhältnis:               {karpathy_diff/raschka_diff_mean:.1f}:1")
    print(f"  Raschka Spanne:           {df_raschka['difficulty'].min():.1f} - {df_raschka['difficulty'].max():.1f}")
    
    print(f"\nPROGRAMMING EFFORT (Änderungsaufwand):")
    print(f"  Karpathy (Monolithisch):  {karpathy_eff:.0f}")
    print(f"  Raschka (Modular Ø):      {raschka_eff_mean:.0f}")
    print(f"  Verhältnis:               {karpathy_eff/raschka_eff_mean:.1f}:1")
    print(f"  Raschka Spanne:           {df_raschka['effort'].min():.0f} - {df_raschka['effort'].max():.0f}")
    
    print(f"\nMODULARITÄT-ANALYSE:")
    print(f"  Anzahl Raschka-Module:    {len(df_raschka)}")
    print(f"  Gesamtaufwand Raschka:    {df_raschka['effort'].sum():.0f}")
    print(f"  Effizienzfaktor:          {karpathy_eff/df_raschka['effort'].sum():.1f}")
    
    print(f"\nKOMPLEXITÄTSVERTEILUNG RASCHKA:")
    low_complexity = len(df_raschka[df_raschka['difficulty'] <= 5])
    med_complexity = len(df_raschka[(df_raschka['difficulty'] > 5) & (df_raschka['difficulty'] <= 10)])
    high_complexity = len(df_raschka[df_raschka['difficulty'] > 10])
    
    print(f"  Niedrige Komplexität (≤5):   {low_complexity} Module")
    print(f"  Mittlere Komplexität (6-10): {med_complexity} Module") 
    print(f"  Hohe Komplexität (>10):      {high_complexity} Module")

if __name__ == "__main__":
    create_halstead_visualization()