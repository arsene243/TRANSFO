"""
Modele de Regression Lineaire Simple pour Prediction de Temperature des Transformateurs
====================================================================================

Ce script cree un modele de regression lineaire pour predire la temperature des enroulements
d'un transformateur electrique et genere un graphique de prediction avec recommandations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuration matplotlib
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 8)

def charger_donnees(fichier):
    """
    Charge et pretraite les donnees du transformateur depuis un fichier Excel.
    
    Cette fonction :
    - Lit le fichier Excel
    - Adapte automatiquement les noms de colonnes
    - Nettoie les donnees (valeurs manquantes, doublons, outliers)
    - Retourne un DataFrame pret pour l'analyse
    """
    print("=== CHARGEMENT DES DONNEES ===")
    
    try:
        # Lecture du fichier Excel
        df = pd.read_excel(fichier)
        print(f"Donnees chargees : {len(df)} lignes, {len(df.columns)} colonnes")
        print(f"Colonnes disponibles : {list(df.columns)}")
        
        # Mapping automatique des noms de colonnes
        mapping_colonnes = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'temp' in col_lower and ('enroul' in col_lower or 'winding' in col_lower):
                mapping_colonnes[col] = 'temperature_enroulements'
            elif 'temp' in col_lower and ('huile' in col_lower or 'oil' in col_lower):
                mapping_colonnes[col] = 'temperature_huile'
            elif 'tension' in col_lower or 'voltage' in col_lower:
                mapping_colonnes[col] = 'tension'
            elif 'courant' in col_lower or 'current' in col_lower:
                mapping_colonnes[col] = 'courant'
            elif 'puissance' in col_lower or 'power' in col_lower:
                mapping_colonnes[col] = 'puissance'
            elif 'pression' in col_lower or 'pressure' in col_lower:
                mapping_colonnes[col] = 'pression_gaz'
        
        # Application du mapping
        if mapping_colonnes:
            df.rename(columns=mapping_colonnes, inplace=True)
            print(f"Colonnes renommees : {mapping_colonnes}")
        
        # Verification de la variable cible
        if 'temperature_enroulements' not in df.columns:
            print("Attention : Colonne 'temperature_enroulements' non trouvee")
            colonnes_numeriques = df.select_dtypes(include=[np.number]).columns
            if len(colonnes_numeriques) > 0:
                df['temperature_enroulements'] = df[colonnes_numeriques[0]]
                print(f"Utilisation de '{colonnes_numeriques[0]}' comme variable cible")
        
        # Nettoyage des donnees
        print("\n=== NETTOYAGE DES DONNEES ===")
        
        # Suppression des doublons
        taille_initiale = len(df)
        df.drop_duplicates(inplace=True)
        print(f"Doublons supprimes : {taille_initiale - len(df)} lignes")
        
        # Selection des colonnes numeriques
        colonnes_numeriques = df.select_dtypes(include=[np.number]).columns
        df_numerique = df[colonnes_numeriques].copy()
        
        # Gestion des valeurs manquantes
        valeurs_manquantes_avant = df_numerique.isnull().sum().sum()
        df_numerique = df_numerique.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
        valeurs_manquantes_apres = df_numerique.isnull().sum().sum()
        print(f"Valeurs manquantes traitees : {valeurs_manquantes_avant} -> {valeurs_manquantes_apres}")
        
        # Traitement des valeurs aberrantes (methode IQR)
        for col in df_numerique.columns:
            Q1 = df_numerique[col].quantile(0.25)
            Q3 = df_numerique[col].quantile(0.75)
            IQR = Q3 - Q1
            limite_inf = Q1 - 1.5 * IQR
            limite_sup = Q3 + 1.5 * IQR
            
            outliers = ((df_numerique[col] < limite_inf) | (df_numerique[col] > limite_sup)).sum()
            if outliers > 0:
                print(f"Outliers detectes dans {col} : {outliers} valeurs")
                df_numerique[col] = np.clip(df_numerique[col], limite_inf, limite_sup)
        
        print(f"Dataset final : {len(df_numerique)} lignes x {len(df_numerique.columns)} colonnes")
        return df_numerique
        
    except Exception as e:
        print(f"Erreur lors du chargement : {e}")
        return None

def creer_variables_derivees(df):
    """
    Cree des variables derivees pour ameliorer la performance du modele.
    
    Variables creees :
    - Ratio tension/courant (impedance approximative)
    - Puissance calculee si non presente
    - Ecart temperature huile-enroulements
    - Moyennes mobiles pour capturer les tendances
    - Charge relative
    """
    print("\n=== CREATION DE VARIABLES DERIVEES ===")
    df_enrichi = df.copy()
    
    # Variable 1 : Ratio tension/courant
    if 'tension' in df.columns and 'courant' in df.columns:
        df_enrichi['ratio_tension_courant'] = df['tension'] / (df['courant'] + 1e-6)
        print("Ratio tension/courant calcule")
    
    # Variable 2 : Puissance calculee
    if 'tension' in df.columns and 'courant' in df.columns and 'puissance' not in df.columns:
        df_enrichi['puissance_calculee'] = df['tension'] * df['courant']
        print("Puissance calculee (VxI)")
    
    # Variable 3 : Ecart de temperature
    if 'temperature_huile' in df.columns and 'temperature_enroulements' in df.columns:
        df_enrichi['ecart_temp_huile_enroul'] = df['temperature_enroulements'] - df['temperature_huile']
        print("Ecart temperature huile-enroulements calcule")
    
    # Variable 4 : Moyennes mobiles
    if 'temperature_enroulements' in df.columns:
        df_enrichi['temp_moyenne_mobile_3'] = df['temperature_enroulements'].rolling(window=3, min_periods=1).mean()
        df_enrichi['temp_moyenne_mobile_5'] = df['temperature_enroulements'].rolling(window=5, min_periods=1).mean()
        print("Moyennes mobiles calculees (fenetres 3 et 5)")
    
    # Variable 5 : Charge relative
    if 'puissance' in df_enrichi.columns and 'tension' in df.columns:
        puissance_max = df_enrichi['puissance'].max()
        df_enrichi['charge_relative'] = df_enrichi['puissance'] / puissance_max
        print("Charge relative calculee")
    
    print(f"Variables ajoutees : {len(df_enrichi.columns) - len(df.columns)}")
    return df_enrichi

def entrainer_modele_regression(df, variable_cible='temperature_enroulements'):
    """
    Entraine un modele de regression lineaire pour predire la temperature des enroulements.
    
    Etapes :
    1. Separation features/target
    2. Division train/test (80%/20%)
    3. Normalisation des donnees
    4. Entrainement du modele
    5. Evaluation des performances
    6. Analyse de l'importance des variables
    """
    print("\n=== ENTRAINEMENT DU MODELE ===")
    
    # Separation des variables
    variables_explicatives = [col for col in df.columns if col != variable_cible]
    X = df[variables_explicatives]
    y = df[variable_cible]
    
    print(f"Variables explicatives : {len(variables_explicatives)} variables")
    print(f"Variables : {variables_explicatives}")
    print(f"Variable cible : {variable_cible}")
    
    # Division train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    print(f"Division des donnees : {len(X_train)} train, {len(X_test)} test")
    
    # Normalisation des variables
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Normalisation des variables appliquee")
    
    # Entrainement du modele
    modele = LinearRegression()
    modele.fit(X_train_scaled, y_train)
    print("Modele de regression lineaire entraine")
    
    # Predictions sur l'ensemble de test
    y_pred = modele.predict(X_test_scaled)
    
    # Calcul des metriques de performance
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metriques = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    print(f"\n=== PERFORMANCE DU MODELE ===")
    print(f"R2 (coefficient de determination) : {r2:.4f}")
    print(f"  Interpretation : Le modele explique {r2*100:.1f}% de la variance")
    print(f"RMSE (erreur quadratique moyenne) : {rmse:.2f} C")
    print(f"MAE (erreur absolue moyenne) : {mae:.2f} C")
    
    # Analyse de l'importance des variables
    importance_variables = pd.DataFrame({
        'Variable': variables_explicatives,
        'Coefficient': modele.coef_,
        'Importance_Abs': np.abs(modele.coef_)
    }).sort_values('Importance_Abs', ascending=False)
    
    print(f"\n=== IMPORTANCE DES VARIABLES ===")
    for i, row in importance_variables.head(5).iterrows():
        print(f"{row['Variable']:<25} : {row['Coefficient']:>8.3f}")
    
    return modele, scaler, metriques, (X_test, y_test, y_pred), importance_variables

def generer_graphique_prediction(donnees_test, metriques, importance_variables):
    """
    Genere un graphique complet de visualisation des predictions.
    
    Le graphique comprend 4 sous-graphiques :
    1. Predictions vs Realite avec seuils d'alerte
    2. Distribution des erreurs (residus)
    3. Importance des variables (top 8)
    4. Metriques de performance et evaluation
    """
    X_test, y_test, y_pred = donnees_test
    
    print("\n=== GENERATION DU GRAPHIQUE ===")
    
    # Configuration de la figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Analyse de Performance du Modele de Prediction de Temperature', 
                 fontsize=16, fontweight='bold')
    
    # Graphique 1 : Predictions vs Valeurs Reelles
    ax1.scatter(y_test, y_pred, alpha=0.6, color='blue', s=50)
    
    # Ligne de prediction parfaite
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Prediction parfaite')
    
    ax1.set_xlabel('Temperature Reelle (C)')
    ax1.set_ylabel('Temperature Predite (C)')
    ax1.set_title(f'Predictions vs Realite (R2 = {metriques["R2"]:.3f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Ajout des seuils de temperature critiques
    seuils = {'Normal': 65, 'Alerte': 75, 'Critique': 85}
    couleurs_seuils = {'Normal': 'green', 'Alerte': 'orange', 'Critique': 'red'}
    
    for nom_seuil, temp_seuil in seuils.items():
        if temp_seuil <= max_val:
            ax1.axhline(y=temp_seuil, color=couleurs_seuils[nom_seuil], 
                       linestyle=':', alpha=0.7, label=f'Seuil {nom_seuil}')
            ax1.axvline(x=temp_seuil, color=couleurs_seuils[nom_seuil], 
                       linestyle=':', alpha=0.7)
    
    # Graphique 2 : Distribution des Erreurs
    residus = y_test - y_pred
    ax2.hist(residus, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Erreur nulle')
    ax2.set_xlabel('Erreur de Prediction (C)')
    ax2.set_ylabel('Frequence')
    ax2.set_title(f'Distribution des Erreurs (MAE = {metriques["MAE"]:.2f}C)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Statistiques des residus
    ax2.text(0.05, 0.95, f'Moyenne: {residus.mean():.3f}C\nEcart-type: {residus.std():.3f}C', 
             transform=ax2.transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Graphique 3 : Importance des Variables
    top_variables = importance_variables.head(8)
    couleurs_barres = plt.cm.viridis(np.linspace(0, 1, len(top_variables)))
    
    barres = ax3.barh(range(len(top_variables)), top_variables['Importance_Abs'], 
                     color=couleurs_barres)
    ax3.set_yticks(range(len(top_variables)))
    ax3.set_yticklabels(top_variables['Variable'], fontsize=9)
    ax3.set_xlabel('Importance Absolue (|Coefficient|)')
    ax3.set_title('Variables les Plus Influentes')
    ax3.grid(True, axis='x', alpha=0.3)
    
    # Annotation des valeurs
    for i, (barre, coef) in enumerate(zip(barres, top_variables['Coefficient'])):
        largeur = barre.get_width()
        ax3.text(largeur + largeur*0.01, barre.get_y() + barre.get_height()/2, 
                f'{coef:+.2f}', ha='left', va='center', fontsize=8)
    
    # Graphique 4 : Metriques et Evaluation
    ax4.axis('off')
    
    # Texte des metriques
    texte_metriques = f"""
METRIQUES DE PERFORMANCE :
========================
R2 Score : {metriques['R2']:.4f}
RMSE : {metriques['RMSE']:.2f} C
MAE : {metriques['MAE']:.2f} C

EVALUATION DE LA QUALITE :
=========================
"""
    
    # Evaluation qualitative
    if metriques['R2'] >= 0.8:
        evaluation = "EXCELLENT\nModele tres fiable"
    elif metriques['R2'] >= 0.6:
        evaluation = "BON\nModele acceptable"
    elif metriques['R2'] >= 0.4:
        evaluation = "MOYEN\nAmeliorations necessaires"
    else:
        evaluation = "FAIBLE\nModele peu fiable"
    
    texte_metriques += evaluation
    
    ax4.text(0.05, 0.95, texte_metriques, transform=ax4.transAxes, 
             verticalalignment='top', fontsize=11, fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('prediction_temperature_transformateur.png', dpi=300, bbox_inches='tight')
    print("Graphique sauvegarde : prediction_temperature_transformateur.png")
    plt.show()

def generer_recommandations(metriques, importance_variables, donnees_test):
    """
    Genere des recommandations techniques basees sur l'analyse du modele.
    
    Recommandations couvrant :
    - Amelioration du modele
    - Surveillance prioritaire
    - Seuils d'alerte
    - Maintenance preventive
    - Optimisation operationnelle
    - Deploiement en production
    """
    X_test, y_test, y_pred = donnees_test
    
    print("\n" + "="*60)
    print("           RECOMMANDATIONS TECHNIQUES")
    print("="*60)
    
    # 1. Recommandations sur la Performance du Modele
    print("\nAMELIORATION DU MODELE :")
    print("-" * 30)
    
    if metriques['R2'] < 0.6:
        print("PRIORITE HAUTE : R2 faible - Considerer :")
        print("  Ajouter plus de variables explicatives")
        print("  Essayer des modeles non-lineaires (Random Forest, SVM)")
        print("  Verifier la qualite des donnees d'entree")
    
    if metriques['RMSE'] > 5:
        print("ATTENTION : RMSE elevee - Actions recommandees :")
        print("  Reviser la strategie de nettoyage des donnees")
        print("  Augmenter la taille de l'echantillon d'entrainement")
        print("  Appliquer des techniques de regularisation (Ridge, Lasso)")
    
    # 2. Surveillance Prioritaire
    print("\nSURVEILLANCE PRIORITAIRE :")
    print("-" * 32)
    
    top_3_variables = importance_variables.head(3)['Variable'].tolist()
    print("Variables critiques a surveiller en priorite :")
    for i, var in enumerate(top_3_variables, 1):
        print(f"  {i}. {var}")
    
    print("\nRecommandations de monitoring :")
    for var in top_3_variables:
        if 'temp' in var.lower():
            print(f"  {var} : Installer capteurs redondants + alarmes")
        elif 'puissance' in var.lower() or 'courant' in var.lower():
            print(f"  {var} : Monitoring en temps reel + historiques")
        elif 'tension' in var.lower():
            print(f"  {var} : Regulation automatique recommandee")
    
    # 3. Seuils d'Alerte
    print("\nSEUILS D'ALERTE RECOMMANDES :")
    print("-" * 35)
    
    temp_moyenne = y_test.mean()
    temp_std = y_test.std()
    
    seuil_attention = temp_moyenne + temp_std
    seuil_alerte = temp_moyenne + 2*temp_std
    seuil_critique = temp_moyenne + 3*temp_std
    
    print(f"NORMAL     : < {seuil_attention:.1f} C")
    print(f"ATTENTION  : {seuil_attention:.1f} C - {seuil_alerte:.1f} C")
    print(f"ALERTE     : {seuil_alerte:.1f} C - {seuil_critique:.1f} C")
    print(f"CRITIQUE   : > {seuil_critique:.1f} C")
    
    # 4. Maintenance Preventive
    print("\nMAINTENANCE PREVENTIVE :")
    print("-" * 29)
    
    predictions_critiques = (y_pred > seuil_alerte).sum()
    pourcentage_critique = (predictions_critiques / len(y_pred)) * 100
    
    if pourcentage_critique > 10:
        print("URGENCE : > 10% des predictions en zone critique")
        print("  Inspection immediate du systeme de refroidissement")
        print("  Revision de la charge operationnelle")
        print("  Maintenance extraordinaire recommandee")
    elif pourcentage_critique > 5:
        print("ATTENTION : 5-10% des predictions en zone d'alerte")
        print("  Programmer maintenance dans les 30 jours")
        print("  Renforcer la surveillance")
    else:
        print("Situation normale - Maintenance preventive standard")
    
    # 5. Optimisation Operationnelle
    print("\nOPTIMISATION OPERATIONNELLE :")
    print("-" * 36)
    
    if 'puissance' in [var.lower() for var in importance_variables.head(3)['Variable']]:
        print("La puissance est un facteur critique :")
        print("  Optimiser la repartition de charge")
        print("  Eviter les pics de puissance prolonges")
        print("  Considerer un delestage automatique")
    
    if any('temp' in var.lower() and 'huile' in var.lower() for var in importance_variables['Variable']):
        print("Temperature huile influente :")
        print("  Verifier efficacite du systeme de refroidissement")
        print("  Controler la qualite de l'huile isolante")
    
    # 6. Deploiement
    print("\nDEPLOIEMENT EN PRODUCTION :")
    print("-" * 34)
    
    if metriques['R2'] >= 0.7:
        print("Modele pret pour deploiement")
        print("  Integrer dans systeme SCADA")
        print("  Mettre en place alertes automatiques")
        print("  Formation equipes maintenance")
    else:
        print("Ameliorer modele avant deploiement")
        print("  Collecter plus de donnees")
        print("  Tester modeles alternatifs")
    
    print("\n" + "="*60)

def main():
    """Fonction principale qui orchestre l'ensemble du processus d'analyse predictive."""
    print("SYSTEME DE PREDICTION DE TEMPERATURE - TRANSFORMATEURS ELECTRIQUES")
    print("=" * 80)
    
    # Etape 1 : Chargement des donnees
    df = charger_donnees('DATASET_TRANSFO.xlsx')
    if df is None:
        print("Impossible de poursuivre sans donnees")
        return
    
    # Etape 2 : Enrichissement des donnees
    df_enrichi = creer_variables_derivees(df)
    
    # Etape 3 : Entrainement du modele
    modele, scaler, metriques, donnees_test, importance_variables = entrainer_modele_regression(df_enrichi)
    
    # Etape 4 : Generation du graphique
    generer_graphique_prediction(donnees_test, metriques, importance_variables)
    
    # Etape 5 : Recommandations
    generer_recommandations(metriques, importance_variables, donnees_test)
    
    print("\nANALYSE TERMINEE AVEC SUCCES")
    print("Graphique sauvegarde : prediction_temperature_transformateur.png")

if __name__ == "__main__":
    main()