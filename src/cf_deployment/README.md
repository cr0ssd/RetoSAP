# GUIA DE DESPLIEGUE EN SAP BTP CLOUD FOUNDRY (CRON-JOB WORKER)

Este directorio contiene los archivos necesarios para empaquetar y desplegar el **Worker en Segundo Plano (Daemon)** en SAP BTP Cloud Foundry utilizando el Buildpack oficial de Python.

---

## 1. Archivos de Configuracion

- **`manifest.yml`**: Define las cuotas del contenedor (`768M` RAM, `2048M` disco), sin ruta HTTP externa (`no-route: true`), healthcheck por proceso y variables de entorno base (`TABLA_DEFAULT`, `GEMINI_MODEL`, `CRON_INTERVAL_MINUTES`).
- **`Procfile`**: Especifica el comando de inicio para Cloud Foundry (`worker: python src/cron_job_worker.py`).
- **`runtime.txt`**: Fija la version del interprete de Python (`python-3.12.x`).
- **`requirements.txt`**: Dependencias ligeras requeridas exclusivamente para inferencia ML, MCP, Gemini y PDFs (`hana-ml`, `hdbcli`, `schedule`, `reportlab`, `google-genai`, etc.).
- **`deploy_cf.sh`**: Script de empaquetado limpio que sube solo `src/` (sin `dashboard/` ni notebooks) junto con este `requirements.txt` ligero.

---

## 2. Como Funciona el Worker Daemon

1. **Arranque Inmediato:** Al encenderse (`cf start sap-soc-cron-worker`), ejecuta una corrida inicial sobre `TABLA_DEFAULT`.
2. **Ciclo Periodico:** Utiliza `schedule` para despertar cada intervalo configurado (por defecto: cada `60` minutos, parametrizable via `CRON_INTERVAL_MINUTES`).
3. **Pipeline End-to-End:**
   - Ejecuta `src/workflow.py` (extraccion, ventanas de `5min` a `3h`, Isolation Forest PAL, clustering y persistencia en tablas `RESULT_*`).
   - Activa el Agente Forense (`src/mcp_hana_tools.py` + `src/agente_forense_postmortem.py`) con deduplicacion multi-escala y muestreo de logs crudos MCP.
   - Persiste los reportes Postmortem en `RESULT_POSTMORTEMS_<mes>` y compila el PDF consolidado del mes.
4. **Proteccion de Memoria:** Llama a `gc.collect()` tras cada corrida para mantenerse dentro de la cuota de Cloud Foundry.

---

## 3. Pasos para Desplegar a SAP BTP

1. Iniciar sesion en tu org y space de SAP BTP:
   ```bash
   cf login --sso -a https://api.cf.us10-001.hana.ondemand.com
   ```
2. Subir paquete ligero sin arrancar (`STOPPED`, para no procesar lotes antes de la demo):
   ```bash
   bash src/cf_deployment/deploy_cf.sh
   ```
3. Configurar credenciales por variable de entorno (si es primera vez):
   ```bash
   cf set-env sap-soc-cron-worker HANA_ADDRESS "<HOST>"
   cf set-env sap-soc-cron-worker HANA_PORT "443"
   cf set-env sap-soc-cron-worker HANA_USER "<USER>"
   cf set-env sap-soc-cron-worker HANA_PASSWORD "<PASSWORD>"
   cf set-env sap-soc-cron-worker GEMINI_API_KEY "<KEY>"
   ```
4. Pre-compilar (*stage*) el droplet sin iniciar la app (`cf packages sap-soc-cron-worker`, luego `cf stage-package ...` y `cf set-droplet ...`).
5. El dia de la demo, apuntar a la tabla deseada e iniciar:
   ```bash
   cf set-env sap-soc-cron-worker TABLA_DEFAULT "LOGS_2026_02"
   cf start sap-soc-cron-worker
   cf logs sap-soc-cron-worker --recent
   ```
