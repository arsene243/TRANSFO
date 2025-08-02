# Modèle de Régression Linéaire pour Transformateurs Électriques

## Description

Ce script utilise la régression linéaire pour prédire la température des enroulements des transformateurs électriques et génère un graphique d'analyse avec recommandations techniques.

## Utilisation

```bash
python regression_clean.py
```

## Résultats

- **Graphique de prédiction** : `prediction_temperature_transformateur.png`
- **Analyse complète** avec métriques de performance
- **Recommandations techniques** pour la maintenance

## Structure des Données

Le fichier `DATASET_TRANSFO.xlsx` doit contenir des colonnes de mesures du transformateur (tension, courant, puissance, températures, etc.)

## 🚀 Mise en Production du Modèle

### 1. Architecture de Déploiement

#### A. Environnement de Production
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Capteurs      │────│   Acquisition    │────│   Modèle IA     │
│   Transformateur│    │   Données SCADA  │    │   Prédictif     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Opérateurs    │────│   Interface      │────│   Système       │
│   Maintenance   │    │   Visualisation  │    │   d'Alertes     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

#### B. Infrastructure Technique
- **Serveur de calcul** : CPU/GPU pour inférence temps réel
- **Base de données** : Stockage historique des mesures et prédictions
- **API REST** : Interface pour intégration SCADA
- **Système d'alertes** : Email, SMS, notifications push
- **Dashboard web** : Monitoring temps réel

### 2. Étapes de Déploiement

#### Étape 1: Préparation du Modèle
```python
# Sauvegarde du modèle entraîné
import joblib

# Après entraînement dans regression_clean.py
joblib.dump(modele, 'modele_temperature.pkl')
joblib.dump(scaler, 'scaler_temperature.pkl')

# Sauvegarde des métadonnées
metadata = {
    'variables': variables_explicatives,
    'seuils': {'normal': 38.1, 'alerte': 40.4, 'critique': 42.6},
    'performance': {'r2': metriques['R2'], 'rmse': metriques['RMSE']}
}
joblib.dump(metadata, 'metadata_modele.pkl')
```

#### Étape 2: API de Prédiction
```python
# api_prediction.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Chargement du modèle au démarrage
modele = joblib.load('modele_temperature.pkl')
scaler = joblib.load('scaler_temperature.pkl')
metadata = joblib.load('metadata_modele.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupération des données SCADA
        data = request.json
        
        # Création du DataFrame
        df = pd.DataFrame([data])
        
        # Normalisation
        df_scaled = scaler.transform(df[metadata['variables']])
        
        # Prédiction
        temperature_pred = modele.predict(df_scaled)[0]
        
        # Détermination du niveau d'alerte
        if temperature_pred >= metadata['seuils']['critique']:
            niveau = 'CRITIQUE'
        elif temperature_pred >= metadata['seuils']['alerte']:
            niveau = 'ALERTE'
        else:
            niveau = 'NORMAL'
        
        return jsonify({
            'temperature_predite': float(temperature_pred),
            'niveau_alerte': niveau,
            'timestamp': pd.Timestamp.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'erreur': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

#### Étape 3: Intégration SCADA
```python
# Integration avec systemes existants
import requests
import time

def monitorer_transformateur():
    while True:
        # Récupération des données SCADA
        donnees_scada = obtenir_donnees_transformateur()
        
        # Appel API prédiction
        response = requests.post('http://api-server:5000/predict', 
                               json=donnees_scada)
        
        if response.status_code == 200:
            resultat = response.json()
            
            # Action selon le niveau d'alerte
            if resultat['niveau_alerte'] == 'CRITIQUE':
                declencher_alerte_critique(resultat)
            elif resultat['niveau_alerte'] == 'ALERTE':
                declencher_alerte_preventive(resultat)
            
            # Stockage en base
            sauvegarder_prediction(resultat)
        
        time.sleep(60)  # Vérification chaque minute
```

### 3. Configuration Système

#### A. Serveur Linux (Recommandé)
```bash
# Installation des dépendances
sudo apt-get update
sudo apt-get install python3-pip nginx redis-server

# Installation packages Python
pip3 install flask pandas scikit-learn joblib gunicorn

# Configuration service systemd
sudo nano /etc/systemd/system/prediction-api.service
```

```ini
[Unit]
Description=API Prediction Transformateur
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/opt/transformer-prediction
Environment=PATH=/opt/transformer-prediction/venv/bin
ExecStart=/opt/transformer-prediction/venv/bin/gunicorn --workers 4 --bind 0.0.0.0:5000 api_prediction:app
Restart=always

[Install]
WantedBy=multi-user.target
```

#### B. Configuration Nginx (Proxy inverse)
```nginx
server {
    listen 80;
    server_name votre-serveur.com;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 4. Monitoring et Maintenance

#### A. Surveillance des Performances
```python
# monitoring.py
import logging
from datetime import datetime, timedelta

def verifier_performance_modele():
    # Récupération des prédictions des dernières 24h
    predictions = obtenir_predictions_recentes()
    mesures_reelles = obtenir_mesures_reelles()
    
    # Calcul de l'erreur courante
    erreur_courante = calculer_erreur(predictions, mesures_reelles)
    
    # Alerte si dégradation
    if erreur_courante > seuil_erreur_acceptable:
        envoyer_alerte_maintenance_modele()
        
    # Log des performances
    logging.info(f"Performance modèle : RMSE={erreur_courante:.2f}")
```

#### B. Mise à Jour Automatique
```python
# Réentraînement périodique
def reentrainer_modele():
    # Récupération nouvelles données
    nouvelles_donnees = obtenir_donnees_periode(days=90)
    
    # Réentraînement
    nouveau_modele = entrainer_modele(nouvelles_donnees)
    
    # Validation sur données récentes
    if valider_nouveau_modele(nouveau_modele):
        deployer_nouveau_modele(nouveau_modele)
        notifier_mise_a_jour()
```

### 5. Sécurité et Fiabilité

#### A. Sécurité
- **Authentification API** : Tokens JWT ou clés API
- **Chiffrement** : HTTPS/TLS pour toutes communications
- **Firewall** : Restriction accès IP
- **Logs d'audit** : Traçabilité des prédictions

#### B. Haute Disponibilité
- **Load Balancer** : Répartition de charge
- **Redondance** : Plusieurs instances API
- **Backup** : Sauvegarde modèles et données
- **Monitoring** : Alertes système (CPU, mémoire, réseau)

### 6. Tests en Production

#### A. Déploiement Progressif
```python
# Phase 1: Mode observation (pas d'actions automatiques)
# Phase 2: Alertes préventives uniquement
# Phase 3: Actions automatiques avec supervision
# Phase 4: Fonctionnement autonome
```

#### B. Validation Continue
- **A/B Testing** : Comparaison ancien/nouveau modèle
- **Validation croisée** : Vérification sur données historiques
- **Feedback opérateurs** : Retours terrains

### 7. Documentation Opérationnelle

#### Manuel d'Exploitation
- **Procédures de démarrage/arrêt**
- **Résolution des incidents courants**
- **Contacts d'escalade**
- **Paramètres de configuration**

#### Formation Équipes
- **Interprétation des prédictions**
- **Actions à entreprendre selon niveaux d'alerte**
- **Maintenance préventive basée sur IA**

Cette approche garantit un déploiement robuste et fiable du modèle prédictif en environnement industriel.