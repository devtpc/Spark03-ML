CONFIG = ./configscripts/config.conf
include ${CONFIG}


#propagateing config settings to the respective folders/files
refresh-confs:
	@cd configscripts && \
	sh refresh_confs.sh

#create infra with terraform - Note: you should be logged in with 'az login'
planinfra:
	@cd terraform && \
	terraform init --backend-config=backend.conf && \
	terraform plan -out terraform.plan

createinfra: planinfra
	@cd terraform && \
	terraform apply -auto-approve terraform.plan 

#leave it here, although for this task it's not needed
databricks-ws-config-export:
	@cd configscripts && \
	sh update_terraform_ws_configs.sh


#destroy databricks workspace with terraform.
destroy-databricks-ws:
	@cd terraform && \
	terraform destroy -auto-approve

#create the databrics workspace from scratch
create-all:
	refresh-confs createinfra
